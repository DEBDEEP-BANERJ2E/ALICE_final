"""
ALICE — Unsloth + TRL GRPO training script.

Uses Unsloth's 4-bit QLoRA (2× faster, ~60% less VRAM vs standard transformers)
combined with TRL's GRPOTrainer for policy optimisation.

Usage:
    python training/train_unsloth.py \
        --model_id Qwen/Qwen2.5-1.5B-Instruct \
        --episodes 200

Supported benchmark models:
    Qwen/Qwen2.5-0.5B-Instruct
    Qwen/Qwen2.5-1.5B-Instruct
    Qwen/Qwen2.5-3B-Instruct
    HuggingFaceTB/SmolLM2-1.7B-Instruct
    google/gemma-3-1b-it

Environment variables:
    ALICE_ENV_URL   — ALICE server URL (default http://localhost:7860)
    HF_TOKEN        — HF write token for checkpoint push
    HF_REPO_ID      — HF repo to push (optional)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import List

import httpx
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
log = logging.getLogger("alice.train_unsloth")

ENV_URL    = os.getenv("ALICE_ENV_URL", "http://localhost:7860")
HF_TOKEN   = os.getenv("HF_TOKEN", "")
HF_REPO_ID = os.getenv("HF_REPO_ID", "")


def _parse_args():
    p = argparse.ArgumentParser(description="ALICE Unsloth+TRL GRPO trainer")
    p.add_argument("--model_id",          default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--episodes",          type=int,   default=200)
    p.add_argument("--max_turns",         type=int,   default=5)
    p.add_argument("--group_size",        type=int,   default=8)
    p.add_argument("--lr",                type=float, default=5e-6)
    p.add_argument("--kl_coef",           type=float, default=0.04)
    p.add_argument("--max_seq_length",    type=int,   default=1024)
    p.add_argument("--max_new_tokens",    type=int,   default=256)
    p.add_argument("--lora_rank",         type=int,   default=16)
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
# Unsloth model loader with transformers fallback
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str, max_seq_length: int, lora_rank: int):
    try:
        from unsloth import FastLanguageModel
        log.info("Loading with Unsloth 4-bit QLoRA: %s", model_id)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,         # auto-detect (bfloat16 on Ampere+)
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_rank * 2,
            lora_dropout=0.0,   # Unsloth optimised for 0
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        log.info("Unsloth model loaded — 4-bit QLoRA active")
        return model, tokenizer, True    # True = unsloth active

    except ImportError:
        log.warning("Unsloth not available — falling back to standard transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=lora_rank,
            lora_alpha=lora_rank * 2, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        return model, tokenizer, False   # False = standard transformers


# ---------------------------------------------------------------------------
# GRPO rollout + update (mirrors train_trl.py but uses Unsloth-trained model)
# ---------------------------------------------------------------------------

def _sample_response(model, tokenizer, prompt: str, max_new_tokens: int,
                      unsloth_active: bool) -> str:
    device  = next(model.parameters()).device
    inputs  = tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=512).to(device)

    if unsloth_active:
        # Unsloth inference mode — 2× faster generation
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)

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


def collect_rollouts(model, tokenizer, group_size: int, max_turns: int,
                      max_new_tokens: int, unsloth_active: bool):
    prompts, responses, rewards, episode_ids = [], [], [], []
    for _ in range(group_size):
        try:
            ep    = env_reset()
            ep_id = ep["episode_id"]
            task  = ep["task"]
            total_reward = 0.0
            last_action  = ""

            for _ in range(max_turns):
                # Chain-of-thought: model reasons before answering (CoT feature)
                cot_prompt = (
                    f"<task>{task}</task>\n"
                    "Reason step by step, then state your final answer.\n"
                    "<reasoning>"
                )
                action = _sample_response(model, tokenizer, cot_prompt,
                                           max_new_tokens, unsloth_active)
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
            log.warning("Rollout error (skipped): %s", exc)

    return prompts, responses, rewards, episode_ids


def grpo_update(model, tokenizer, optimizer, prompts, responses, rewards,
                kl_coef: float, unsloth_active: bool) -> float:
    if not rewards:
        return 0.0

    if unsloth_active:
        # Switch back to training mode after inference pass
        from unsloth import FastLanguageModel
        FastLanguageModel.for_training(model)

    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std().clamp(min=1e-6) )

    device     = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device)

    for adv, prompt, response in zip(advantages, prompts, responses):
        full_text = prompt + "\n" + response
        inputs    = tokenizer(full_text, return_tensors="pt",
                              truncation=True, max_length=768).to(device)
        labels    = inputs["input_ids"].clone()
        prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        labels[0, :prompt_len] = -100

        out      = model(**inputs, labels=labels)
        log_prob = -out.loss
        kl_pen   = kl_coef * (log_prob ** 2)
        loss     = -(adv.to(device) * log_prob) + kl_pen
        total_loss = total_loss + loss

    total_loss = total_loss / max(len(rewards), 1)
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(total_loss.detach())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    model, tokenizer, unsloth_active = load_model_and_tokenizer(
        args.model_id, args.max_seq_length, args.lora_rank
    )
    log.info("Unsloth active: %s", unsloth_active)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    all_rewards:   List[float] = []
    all_successes: List[float] = []
    disc_coverage: float = 0.0

    log.info("Starting Unsloth GRPO | model=%s | episodes=%d", args.model_id, args.episodes)

    for ep in range(1, args.episodes + 1):
        prompts, responses, rewards, _ = collect_rollouts(
            model, tokenizer, args.group_size, args.max_turns,
            args.max_new_tokens, unsloth_active,
        )
        if not rewards:
            continue

        loss = grpo_update(model, tokenizer, optimizer, prompts, responses,
                            rewards, args.kl_coef, unsloth_active)

        all_rewards.extend(rewards)
        all_successes.extend([1.0 if r > 0.3 else 0.0 for r in rewards])

        try:
            state = httpx.get(f"{ENV_URL}/health", timeout=5.0).json()
            disc_coverage = state.get("disc_coverage", disc_coverage)
        except Exception:
            pass

        avg_r    = sum(all_rewards[-50:]) / max(len(all_rewards[-50:]), 1)
        avg_succ = sum(all_successes[-50:]) / max(len(all_successes[-50:]), 1)

        log.info("Episode %4d | loss=%.4f | avg_reward=%.4f | success=%.2f%%",
                 ep, loss, avg_r, avg_succ * 100)

        if args.update_leaderboard and ep % 10 == 0:
            leaderboard_update(args.model_id, avg_r, avg_succ, disc_coverage, ep)

        if args.checkpoint_every and ep % args.checkpoint_every == 0:
            ckpt = f"checkpoints/{args.model_id.replace('/', '_')}_unsloth_ep{ep}"
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            log.info("Checkpoint: %s", ckpt)
            if args.push_to_hub and HF_REPO_ID and HF_TOKEN:
                from huggingface_hub import HfApi
                HfApi().upload_folder(folder_path=ckpt, repo_id=HF_REPO_ID,
                                      token=HF_TOKEN, commit_message=f"unsloth ep{ep}")

    if all_rewards and args.update_leaderboard:
        avg_r    = sum(all_rewards) / len(all_rewards)
        avg_succ = sum(all_successes) / len(all_successes)
        leaderboard_update(args.model_id, avg_r, avg_succ, disc_coverage, len(all_rewards))
        log.info("Final | avg_reward=%.4f | success=%.2f%% | unsloth=%s",
                 avg_r, avg_succ * 100, unsloth_active)

    return model, tokenizer


if __name__ == "__main__":
    args = _parse_args()
    train(args)