"""
ALICE — Pure TRL GRPO training script (no Unsloth dependency).

Usage:
    python training/train_trl.py \
        --model_id Qwen/Qwen2.5-0.5B-Instruct \
        --episodes 200 \
        --steps_per_update 4

Supported benchmark models (pass any of these as --model_id):
    Qwen/Qwen2.5-0.5B-Instruct
    Qwen/Qwen2.5-1.5B-Instruct
    Qwen/Qwen2.5-3B-Instruct
    HuggingFaceTB/SmolLM2-1.7B-Instruct
    google/gemma-3-1b-it

Environment variables:
    ALICE_ENV_URL   — base URL of the ALICE server (default http://localhost:7860)
    HF_TOKEN        — Hugging Face write token (optional, for checkpoint push)
    HF_REPO_ID      — HF repo to push checkpoints (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
log = logging.getLogger("alice.train_trl")

ENV_URL     = os.getenv("ALICE_ENV_URL", "http://localhost:7860")
HF_TOKEN    = os.getenv("HF_TOKEN", "")
HF_REPO_ID  = os.getenv("HF_REPO_ID", "")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="ALICE pure-TRL GRPO trainer")
    p.add_argument("--model_id",          default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--episodes",          type=int,   default=200)
    p.add_argument("--max_turns",         type=int,   default=5)
    p.add_argument("--group_size",        type=int,   default=8,
                   help="GRPO group size G — actions sampled per step")
    p.add_argument("--lr",                type=float, default=1e-5)
    p.add_argument("--kl_coef",           type=float, default=0.04)
    p.add_argument("--steps_per_update",  type=int,   default=4)
    p.add_argument("--max_new_tokens",    type=int,   default=256)
    p.add_argument("--load_in_4bit",      action="store_true")
    p.add_argument("--push_to_hub",       action="store_true")
    p.add_argument("--checkpoint_every",  type=int,   default=50)
    p.add_argument("--update_leaderboard", action="store_true", default=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Environment client
# ---------------------------------------------------------------------------

def env_reset() -> dict:
    r = httpx.post(f"{ENV_URL}/reset", timeout=30.0)
    r.raise_for_status()
    return r.json()


def env_step(episode_id: str, action: str) -> dict:
    r = httpx.post(f"{ENV_URL}/step",
                   json={"episode_id": episode_id, "action": action},
                   timeout=30.0)
    r.raise_for_status()
    return r.json()


def leaderboard_update(model_id: str, avg_reward: float, success_rate: float,
                        disc_cov: float, episodes: int):
    try:
        httpx.post(
            f"{ENV_URL}/leaderboard/update",
            json={"model_id": model_id, "avg_reward": avg_reward,
                  "success_rate": success_rate,
                  "discrimination_coverage": disc_cov,
                  "episodes_run": episodes},
            timeout=10.0,
        )
    except Exception as exc:
        log.warning("Leaderboard update failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Model loader (standard transformers, no Unsloth)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str, load_in_4bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = None
    if load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    log.info("Loading model: %s (4-bit=%s)", model_id, load_in_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Attach LoRA adapter via PEFT for fine-tuning
    from peft import LoraConfig, get_peft_model, TaskType
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# GRPO core (pure PyTorch, no TRL GRPOTrainer to keep this self-contained)
# ---------------------------------------------------------------------------

@dataclass
class RolloutBatch:
    prompts:     List[str]
    responses:   List[str]
    rewards:     List[float]
    episode_ids: List[str]


def _sample_response(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def collect_rollouts(model, tokenizer, group_size: int,
                     max_turns: int, max_new_tokens: int) -> RolloutBatch:
    prompts, responses, rewards, episode_ids = [], [], [], []

    for _ in range(group_size):
        try:
            ep = env_reset()
            ep_id = ep["episode_id"]
            task  = ep["task"]
            total_reward = 0.0
            last_action  = ""

            for turn in range(max_turns):
                # Chain-of-thought prompt: model reasons before answering
                cot_prompt = (
                    f"<task>{task}</task>\n"
                    f"Think step by step, then give your final answer.\n"
                    f"<reasoning>"
                )
                action = _sample_response(model, tokenizer, cot_prompt, max_new_tokens)
                result = env_step(ep_id, action)
                total_reward += result["reward"]
                last_action   = action
                if result.get("done"):
                    break

            prompts.append(task)
            responses.append(last_action)
            rewards.append(total_reward)
            episode_ids.append(ep_id)
        except Exception as exc:
            log.warning("Rollout error (skipping): %s", exc)

    return RolloutBatch(prompts=prompts, responses=responses,
                        rewards=rewards, episode_ids=episode_ids)


def grpo_update(model, tokenizer, optimizer, batch: RolloutBatch,
                kl_coef: float) -> float:
    """Single GRPO gradient step — group-normalised advantages, KL penalty."""
    if not batch.rewards:
        return 0.0

    rewards_t = torch.tensor(batch.rewards, dtype=torch.float32)
    mean_r    = rewards_t.mean()
    std_r     = rewards_t.std().clamp(min=1e-6)
    advantages = (rewards_t - mean_r) / std_r   # group-normalised

    device  = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device)

    for adv, prompt, response in zip(advantages, batch.prompts, batch.responses):
        full_text = prompt + "\n" + response
        inputs    = tokenizer(full_text, return_tensors="pt",
                              truncation=True, max_length=768).to(device)
        labels    = inputs["input_ids"].clone()
        prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        labels[0, :prompt_len] = -100   # mask prompt tokens from loss

        out      = model(**inputs, labels=labels)
        log_prob = -out.loss                    # average token log-prob
        kl_pen   = kl_coef * (log_prob ** 2)   # simplified KL proxy
        loss     = -(adv.to(device) * log_prob) + kl_pen
        total_loss = total_loss + loss

    total_loss = total_loss / max(len(batch.rewards), 1)
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(total_loss.detach())


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.load_in_4bit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    all_rewards:  List[float] = []
    all_successes: List[float] = []
    disc_coverage: float = 0.0

    log.info("Starting TRL-style GRPO training | model=%s | episodes=%d",
             args.model_id, args.episodes)

    for ep in range(1, args.episodes + 1):
        batch = collect_rollouts(
            model, tokenizer,
            group_size=args.group_size,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
        )
        if not batch.rewards:
            continue

        loss = grpo_update(model, tokenizer, optimizer, batch, args.kl_coef)

        all_rewards.extend(batch.rewards)
        # Treat reward > 0.3 as success (mirrors ALICE's threshold logic)
        all_successes.extend([1.0 if r > 0.3 else 0.0 for r in batch.rewards])

        # Fetch live discrimination coverage
        try:
            state = httpx.get(f"{ENV_URL}/health", timeout=5.0).json()
            disc_coverage = state.get("disc_coverage", disc_coverage)
        except Exception:
            pass

        avg_r     = sum(all_rewards[-50:]) / max(len(all_rewards[-50:]), 1)
        avg_succ  = sum(all_successes[-50:]) / max(len(all_successes[-50:]), 1)

        log.info("Episode %4d | loss=%.4f | avg_reward=%.4f | success=%.2f%%",
                 ep, loss, avg_r, avg_succ * 100)

        if args.update_leaderboard and ep % 10 == 0:
            leaderboard_update(args.model_id, avg_r, avg_succ, disc_coverage, ep)

        if args.checkpoint_every and ep % args.checkpoint_every == 0:
            ckpt = f"checkpoints/{args.model_id.replace('/', '_')}_ep{ep}"
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            log.info("Checkpoint saved: %s", ckpt)
            if args.push_to_hub and HF_REPO_ID and HF_TOKEN:
                from huggingface_hub import HfApi
                HfApi().upload_folder(folder_path=ckpt, repo_id=HF_REPO_ID,
                                      token=HF_TOKEN, commit_message=f"checkpoint ep{ep}")
                log.info("Pushed to HF Hub: %s", HF_REPO_ID)

    # Final leaderboard update
    if all_rewards and args.update_leaderboard:
        avg_r    = sum(all_rewards) / len(all_rewards)
        avg_succ = sum(all_successes) / len(all_successes)
        leaderboard_update(args.model_id, avg_r, avg_succ, disc_coverage, len(all_rewards))
        log.info("Final | avg_reward=%.4f | success=%.2f%% | episodes=%d",
                 avg_r, avg_succ * 100, len(all_rewards))

    return model, tokenizer


if __name__ == "__main__":
    args = _parse_args()
    train(args)