# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "torch",
#   "transformers",
#   "peft",
#   "accelerate",
#   "bitsandbytes",
#   "huggingface_hub",
#   "numpy",
# ]
# ///
"""
ALICE — HF Jobs training script (UV script format).

Runs inside a HF Job container:
  1. Clones the ALICE repo from the HF Space
  2. Starts the ALICE FastAPI server in a background thread
  3. Runs GRPO training against it
  4. Pushes the trained model + leaderboard update to HF Hub

Environment variables (set as Job secrets):
  HF_TOKEN        — HF write token (required)
  HF_SPACE_ID     — e.g. rohanjain1648/alice-rl-environment
  MODEL_ID        — model to train (default Qwen/Qwen2.5-0.5B-Instruct)
  EPISODES        — number of training episodes (default 100)
  GROUP_SIZE      — GRPO group size (default 4)
  MAX_TURNS       — turns per episode (default 3)
  LR              — learning rate (default 1e-5)
  KL_COEF         — KL penalty coefficient (default 0.04)
  MAX_NEW_TOKENS  — max tokens per generation (default 128)
  HUB_REPO_ID     — HF repo to push trained model (optional)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List

import httpx
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("alice.hf_job")

# ── Config from env ───────────────────────────────────────────────────────────
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
HF_SPACE_ID    = os.environ.get("HF_SPACE_ID", "rohanjain1648/alice-rl-environment")
MODEL_ID       = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
EPISODES       = int(os.environ.get("EPISODES", "30"))
GROUP_SIZE     = int(os.environ.get("GROUP_SIZE", "4"))
MAX_TURNS      = int(os.environ.get("MAX_TURNS", "1"))
LR             = float(os.environ.get("LR", "1e-5"))
KL_COEF        = float(os.environ.get("KL_COEF", "0.04"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "32"))
HUB_REPO_ID    = os.environ.get("HUB_REPO_ID", "")
CKPT_DIR       = "/tmp/alice_checkpoint"
REPO_DIR       = "/tmp/alice_repo"
# Always use the live HF Space — no local server needed in a Job container
ENV_URL        = f"https://{HF_SPACE_ID.replace('/', '-')}.hf.space"
_client        = httpx.Client(timeout=30.0)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Clone repo
# ─────────────────────────────────────────────────────────────────────────────

def clone_repo():
    log.info("Cloning ALICE repo from HF Space: %s", HF_SPACE_ID)
    if os.path.exists(REPO_DIR):
        log.info("Repo already exists at %s", REPO_DIR)
        return
    clone_url = f"https://huggingface.co/spaces/{HF_SPACE_ID}"
    if HF_TOKEN:
        user = HF_SPACE_ID.split("/")[0]
        clone_url = f"https://{user}:{HF_TOKEN}@huggingface.co/spaces/{HF_SPACE_ID}"
    result = subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, REPO_DIR],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.warning("git clone failed (%s) — trying snapshot_download", result.stderr[:200])
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_SPACE_ID, repo_type="space",
            local_dir=REPO_DIR, token=HF_TOKEN or None,
        )
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    log.info("Repo ready at %s", REPO_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Start ALICE server in background thread
# ─────────────────────────────────────────────────────────────────────────────

_server_thread: threading.Thread | None = None

def start_server():
    global _server_thread

    def _run():
        env = os.environ.copy()
        env["PORT"] = "7860"
        env["PYTHONPATH"] = REPO_DIR
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "alice_server:api",
             "--host", "0.0.0.0", "--port", "7860", "--log-level", "warning"],
            cwd=REPO_DIR, env=env,
        )

    _server_thread = threading.Thread(target=_run, daemon=True)
    _server_thread.start()

    log.info("Waiting for ALICE server to be ready...")
    for attempt in range(30):
        time.sleep(3)
        try:
            r = httpx.get(f"{ENV_URL}/health", timeout=4.0)
            if r.status_code == 200:
                log.info("Server ready: %s", r.json())
                return
        except Exception:
            pass
        log.info("  attempt %d/30...", attempt + 1)

    raise RuntimeError("ALICE server did not start within 90 seconds")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load model
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    log.info("Loading tokenizer: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_in_4bit = torch.cuda.is_available()
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if load_in_4bit else None

    log.info("Loading model (4-bit=%s, device=%s)", load_in_4bit,
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        # Speed: skip slow safetensors verification
        low_cpu_mem_usage=True,
    )

    # Smallest LoRA possible — r=4 trains 4× fewer params → faster backward
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Enable gradient checkpointing to save VRAM (allows larger batch)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    log.info("Model ready")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — GRPO training
# ─────────────────────────────────────────────────────────────────────────────

def sample_response(model, tokenizer, prompt: str) -> str:
    device = next(model.parameters()).device
    # Short direct prompt — no CoT wrapper, saves ~30 tokens of input overhead
    inputs = tokenizer(
        f"Task: {prompt}\nAnswer:",
        return_tensors="pt", truncation=True, max_length=128,  # 128 not 512
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,   # 32 tokens
            do_sample=False,                  # greedy — 3-5× faster than sampling
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def collect_rollouts(model, tokenizer) -> tuple[list, list, list]:
    prompts, responses, rewards = [], [], []
    for _ in range(GROUP_SIZE):
        try:
            ep    = _client.post(f"{ENV_URL}/reset").json()
            ep_id = ep["episode_id"]
            task  = ep["task"]
            action = sample_response(model, tokenizer, task)
            # MAX_TURNS=1: single step, no loop overhead
            result = _client.post(
                f"{ENV_URL}/step",
                json={"episode_id": ep_id, "action": action},
            ).json()
            prompts.append(task)
            responses.append(action)
            rewards.append(result["reward"])
        except Exception as exc:
            log.warning("Rollout error (skipped): %s", exc)
    return prompts, responses, rewards


def grpo_update(model, tokenizer, optimizer, prompts, responses, rewards) -> float:
    if not rewards:
        return 0.0
    rewards_t  = torch.tensor(rewards, dtype=torch.float32)
    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std().clamp(min=1e-6))
    device     = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device)

    for adv, prompt, response in zip(advantages, prompts, responses):
        full_text  = f"Task: {prompt}\nAnswer: {response}"
        inputs     = tokenizer(full_text, return_tensors="pt",
                               truncation=True, max_length=256).to(device)  # 256 not 768
        labels     = inputs["input_ids"].clone()
        prompt_len = len(tokenizer(f"Task: {prompt}\nAnswer:", return_tensors="pt")["input_ids"][0])
        labels[0, :prompt_len] = -100
        out        = model(**inputs, labels=labels)
        log_prob   = -out.loss
        kl_pen     = KL_COEF * (log_prob ** 2)
        total_loss = total_loss + (-(adv.to(device) * log_prob) + kl_pen)

    total_loss = total_loss / max(len(rewards), 1)
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(total_loss.detach())


def push_leaderboard(avg_r: float, avg_succ: float, disc: float, ep: int):
    payload = {"model_id": MODEL_ID, "avg_reward": avg_r,
               "success_rate": avg_succ, "discrimination_coverage": disc,
               "episodes_run": ep}
    for url in [ENV_URL, f"https://{HF_SPACE_ID.replace('/', '-')}.hf.space"]:
        try:
            _client.post(f"{url}/leaderboard/update", json=payload)
        except Exception as exc:
            log.warning("Leaderboard push to %s failed: %s", url, exc)
    log.info("Leaderboard pushed: ep=%d avg_reward=%.4f success=%.2f%%", ep, avg_r, avg_succ*100)


def train(model, tokenizer):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    all_rewards, all_successes, all_losses = [], [], []
    disc = 0.0
    t0 = time.time()

    log.info("Starting GRPO training: %d episodes × %d rollouts (1 turn, greedy, 32 tokens)",
             EPISODES, GROUP_SIZE)

    for ep in range(1, EPISODES + 1):
        prompts, responses, rewards = collect_rollouts(model, tokenizer)
        if not rewards:
            continue

        loss = grpo_update(model, tokenizer, optimizer, prompts, responses, rewards)
        all_rewards.extend(rewards)
        all_successes.extend([1.0 if r > 0.3 else 0.0 for r in rewards])
        all_losses.append(loss)

        avg_r    = float(np.mean(all_rewards[-50:]))
        avg_succ = float(np.mean(all_successes[-50:]))

        log.info("Ep %3d/%d | loss=%.4f | avg_reward=%.4f | success=%.0f%% | %.0fs",
                 ep, EPISODES, loss, avg_r, avg_succ * 100, time.time() - t0)

        # Push every 5 episodes (not 10) so leaderboard updates faster
        if ep % 5 == 0:
            push_leaderboard(avg_r, avg_succ, disc, ep)

    avg_r    = float(np.mean(all_rewards)) if all_rewards else 0.0
    avg_succ = float(np.mean(all_successes)) if all_successes else 0.0
    push_leaderboard(avg_r, avg_succ, disc, EPISODES)

    log.info("Done | avg_reward=%.4f | success=%.0f%% | rollouts=%d | time=%.0fs",
             avg_r, avg_succ * 100, len(all_rewards), time.time() - t0)
    return avg_r, avg_succ


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Save + push checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_and_push(model, tokenizer, avg_r: float, avg_succ: float):
    import json
    os.makedirs(CKPT_DIR, exist_ok=True)
    log.info("Saving checkpoint to %s", CKPT_DIR)
    model.save_pretrained(CKPT_DIR)
    tokenizer.save_pretrained(CKPT_DIR)

    meta = {
        "model_id": MODEL_ID, "episodes": EPISODES, "group_size": GROUP_SIZE,
        "max_turns": MAX_TURNS, "lr": LR, "kl_coef": KL_COEF,
        "final_avg_reward": round(avg_r, 4), "final_success_rate": round(avg_succ, 4),
        "hf_space": f"https://{HF_SPACE_ID.replace('/', '-')}.hf.space",
    }
    with open(f"{CKPT_DIR}/alice_training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata saved")

    if HUB_REPO_ID and HF_TOKEN:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(HUB_REPO_ID, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=CKPT_DIR,
            repo_id=HUB_REPO_ID,
            commit_message=f"ALICE GRPO — {MODEL_ID.split('/')[-1]} — {EPISODES} eps — reward={avg_r:.4f}",
        )
        log.info("Model pushed to https://huggingface.co/%s", HUB_REPO_ID)
    else:
        log.info("HUB_REPO_ID not set — checkpoint saved locally only at %s", CKPT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("ALICE HF Job Training (direct to HF Space)")
    log.info("  Model:    %s", MODEL_ID)
    log.info("  Episodes: %d | Group: %d | Turns: %d | Tokens: %d",
             EPISODES, GROUP_SIZE, MAX_TURNS, MAX_NEW_TOKENS)
    log.info("  Env URL:  %s", ENV_URL)
    log.info("=" * 60)

    # Verify the Space is reachable before loading the model
    log.info("Checking HF Space health...")
    for attempt in range(10):
        try:
            r = _client.get(f"{ENV_URL}/health")
            if r.status_code == 200:
                log.info("Space healthy: %s", r.json())
                break
        except Exception as exc:
            log.warning("Health check attempt %d/10 failed: %s", attempt + 1, exc)
            time.sleep(5)
    else:
        raise RuntimeError(f"HF Space {ENV_URL} not reachable after 10 attempts")

    model, tokenizer = load_model()
    avg_r, avg_succ  = train(model, tokenizer)
    save_and_push(model, tokenizer, avg_r, avg_succ)

    log.info("Job complete.")
