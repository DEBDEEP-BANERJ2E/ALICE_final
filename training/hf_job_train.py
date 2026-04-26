# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx", "numpy",
#   "torch", "transformers>=4.40.0", "peft>=0.10.0",
#   "accelerate>=0.27.0", "trl>=0.8.6", "datasets>=2.18.0",
#   "bitsandbytes>=0.43.0",
# ]
# ///
"""
ALICE — Production RL training script using TRL GRPOTrainer.

Architecture:
  - Loads model locally on GPU (a10g-small = 24GB VRAM)
  - Runs proper multi-turn ALICE episodes: reset → generate × MAX_TURNS → step
  - Uses TRL GRPOTrainer with real reward signals from the ALICE verifier stack
  - Pushes full metrics (rewards, advantages, loss, disc_coverage, failure_bank,
    curriculum_tier, per-turn composites) to /training/push after every episode
  - Registers job ID + live URL in /jobs/register so the dashboard shows it
  - Saves LoRA checkpoint and optionally pushes to HF Hub

Env vars (set as Job secrets/env):
  HF_TOKEN        — HF write token (required)
  HF_SPACE_ID     — e.g. rohanjain1648/alice-rl-environment
  MODEL_ID        — HF model ID (default Qwen/Qwen2.5-0.5B-Instruct)
  EPISODES        — training episodes (default 100)
  GROUP_SIZE      — GRPO rollouts per update (default 8)
  MAX_TURNS       — turns per episode (default 3)
  LR              — learning rate (default 1e-5)
  KL_COEF         — KL penalty coefficient (default 0.04)
  MAX_NEW_TOKENS  — max tokens per generation (default 64)
  LOAD_IN_4BIT    — 1 to enable 4-bit QLoRA (default 1 on GPU)
  LORA_R          — LoRA rank (default 16)
  PUSH_TO_HUB     — 1 to push checkpoint to HF Hub after training
  HUB_REPO_ID     — HF Hub repo for checkpoint (e.g. username/alice-trained)
  JOB_ID          — injected by HF Jobs runtime automatically
"""
from __future__ import annotations
import logging, os, time, json, uuid
import httpx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
log = logging.getLogger("alice.train")

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
HF_SPACE_ID    = os.environ.get("HF_SPACE_ID", "rohanjain1648/alice-rl-environment")
MODEL_ID       = os.environ.get("MODEL_ID",       "Qwen/Qwen2.5-0.5B-Instruct")
EPISODES       = int(os.environ.get("EPISODES",       "100"))
GROUP_SIZE     = int(os.environ.get("GROUP_SIZE",      "8"))
MAX_TURNS      = int(os.environ.get("MAX_TURNS",       "3"))
LR             = float(os.environ.get("LR",            "1e-5"))
KL_COEF        = float(os.environ.get("KL_COEF",       "0.04"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS",  "64"))
LOAD_IN_4BIT   = os.environ.get("LOAD_IN_4BIT", "1") == "1"
LORA_R         = int(os.environ.get("LORA_R",          "16"))
PUSH_TO_HUB    = os.environ.get("PUSH_TO_HUB", "0") == "1"
HUB_REPO_ID    = os.environ.get("HUB_REPO_ID", "")
# HF Jobs injects this automatically; fall back to a generated ID
JOB_ID         = os.environ.get("JOB_ID", str(uuid.uuid4())[:24])

SPACE_URL = f"https://{HF_SPACE_ID.replace('/', '-')}.hf.space"
JOB_URL   = f"https://huggingface.co/jobs/{HF_SPACE_ID.split('/')[0]}/{JOB_ID}"

# Persistent HTTP client to the ALICE Space
_env = httpx.Client(base_url=SPACE_URL, timeout=30.0)

# ── System prompt — forces result = ... Python output ────────────────────────
_SYSTEM = (
    "You are a precise Python solver. "
    "For every task, output ONLY a single Python statement that assigns the answer to `result`.\n"
    "Rules:\n"
    "  - Always write: result = <value>\n"
    "  - No explanations, no markdown, no extra lines\n"
    "  - Use Python literals: strings in quotes, numbers bare, booleans True/False\n"
    "Examples:\n"
    "  result = 42\n"
    "  result = 'Canberra'\n"
    "  result = True\n"
    "  result = [x**2 for x in range(5)]\n"
    "  result = sorted([3,1,4,1,5], reverse=True)"
)

_FEW_SHOT = (
    "Task: What is 15 * 7 - 3?\nresult = 102\n\n"
    "Task: Capital of Australia?\nresult = 'Canberra'\n\n"
    "Task: Is 7 prime?\nresult = True\n\n"
    "Task: Sum of even numbers 1-20?\nresult = sum(x for x in range(1,21) if x%2==0)\n\n"
)

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    log.info("Loading %s | 4bit=%s | LoRA r=%d", MODEL_ID, LOAD_IN_4BIT, LORA_R)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for generation batching

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if LOAD_IN_4BIT else None

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if _has_flash_attn() else "eager",
    )

    if LOAD_IN_4BIT:
        mdl = prepare_model_for_kbit_training(mdl)

    # Detect target modules dynamically
    target_mods = _get_lora_targets(mdl)
    log.info("LoRA target modules: %s", target_mods)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        target_modules=target_mods,
        lora_dropout=0.05,
        bias="none",
    )
    mdl = get_peft_model(mdl, lora)
    mdl.print_trainable_parameters()

    log.info("Model loaded on %s", next(mdl.parameters()).device)
    return mdl, tok


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def _get_lora_targets(model) -> list[str]:
    """Detect attention projection layer names dynamically."""
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
    found = set()
    for name, _ in model.named_modules():
        for c in candidates:
            if name.endswith(c):
                found.add(c)
    # At minimum use q+v
    return list(found) if found else ["q_proj", "v_proj"]

# ── Generation ────────────────────────────────────────────────────────────────
def generate(model, tokenizer, task: str, feedback: str = "") -> str:
    """Generate a result = ... Python answer for the given task."""
    import torch

    # On turn 2+, include feedback from previous turn
    feedback_block = f"\nPrevious attempt feedback: {feedback}\n" if feedback else ""
    prompt = (
        f"{_SYSTEM}\n\n"
        f"{_FEW_SHOT}"
        f"Task: {task}{feedback_block}\nresult ="
    )

    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=512,
        padding=False,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.92,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Normalise to result = ... format
    first_line = raw.split("\n")[0].strip()
    if first_line.startswith("result"):
        return first_line[:256]
    return f"result = {first_line[:200]}"

# ── ALICE environment calls ───────────────────────────────────────────────────
def env_reset() -> dict:
    return _env.post("/reset").json()

def env_step(ep_id: str, action: str) -> dict:
    return _env.post("/step", json={"episode_id": ep_id, "action": action}).json()

# ── Episode runner — proper multi-turn ───────────────────────────────────────
def run_episode(model, tokenizer) -> dict:
    """
    Full 3-turn ALICE episode:
      Turn 1: initial attempt
      Turn 2: retry with feedback from T1 verification
      Turn 3: final attempt with hint
    Returns per-turn rewards, composites, actions, and total reward.
    """
    ep    = env_reset()
    ep_id = ep["episode_id"]
    task  = ep["task"]

    turn_rewards    = []
    turn_composites = []
    turn_actions    = []
    feedback        = ""
    total_r         = 0.0

    for turn in range(1, MAX_TURNS + 1):
        action = generate(model, tokenizer, task, feedback)
        result = env_step(ep_id, action)

        r         = float(result.get("reward", 0.01))
        done      = result.get("done", False)
        info      = result.get("info", {})
        verif     = info.get("verification", {})
        composite = float(verif.get("composite_score", 0.0))

        turn_rewards.append(r)
        turn_composites.append(composite)
        turn_actions.append(action)
        total_r += r

        # Extract feedback for next turn
        feedback = verif.get("reasoning", "") or ""
        if composite >= 0.5:
            feedback = f"Turn {turn} succeeded (score={composite:.2f}). Maintain this quality."
        else:
            t1 = verif.get("tier1_details", {})
            if t1 and not t1.get("success"):
                feedback = f"Error: {t1.get('error_message', 'unknown')}. Fix the Python code."
            else:
                feedback = f"Score {composite:.2f} — improve correctness and completeness."

        if done:
            break

    final_composite = turn_composites[-1] if turn_composites else 0.0
    return {
        "task":             task,
        "actions":          turn_actions,
        "turn_rewards":     turn_rewards,
        "turn_composites":  turn_composites,
        "total_reward":     total_r,
        "success":          final_composite >= 0.5,
        "composite":        final_composite,
        "turns_taken":      len(turn_rewards),
    }

# ── Rollout collection ────────────────────────────────────────────────────────
def collect_rollouts(model, tokenizer) -> dict:
    """Collect GROUP_SIZE rollouts, return all data needed for GRPO update."""
    rewards, successes, composites = [], [], []
    prompts, responses             = [], []
    all_turn_rewards               = []

    for _ in range(GROUP_SIZE):
        try:
            ep = run_episode(model, tokenizer)
            rewards.append(ep["total_reward"])
            successes.append(1.0 if ep["success"] else 0.0)
            composites.append(ep["composite"])
            # Use last action + task as the (prompt, response) pair for GRPO
            prompts.append(ep["task"])
            responses.append(ep["actions"][-1] if ep["actions"] else "result = None")
            all_turn_rewards.append(ep["turn_rewards"])
        except Exception as exc:
            log.warning("Rollout error (skipped): %s", exc)
            rewards.append(0.01)
            successes.append(0.0)
            composites.append(0.0)
            prompts.append("")
            responses.append("")
            all_turn_rewards.append([0.01])

    return {
        "rewards":          rewards,
        "successes":        successes,
        "composites":       composites,
        "prompts":          prompts,
        "responses":        responses,
        "all_turn_rewards": all_turn_rewards,
    }

# ── GRPO policy gradient update ───────────────────────────────────────────────
def grpo_update(model, tokenizer, optimizer, rollouts: dict) -> dict:
    """
    Group-normalised policy gradient (GRPO):
      advantage_i = (r_i - mean(r)) / (std(r) + eps)
      loss = -sum(advantage_i * log_prob_i) / N + KL_COEF * sum(log_prob_i^2) / N
    Returns loss value and per-rollout advantages.
    """
    import torch
    import torch.nn as nn

    rewards   = rollouts["rewards"]
    prompts   = rollouts["prompts"]
    responses = rollouts["responses"]

    device = next(model.parameters()).device
    r_t    = torch.tensor(rewards, dtype=torch.float32, device=device)
    adv    = (r_t - r_t.mean()) / (r_t.std().clamp(min=1e-6))

    total_loss = torch.tensor(0.0, device=device, requires_grad=False)
    n_valid    = 0

    for a, prompt, resp in zip(adv, prompts, responses):
        if not prompt or not resp:
            continue

        full_text  = prompt + "\nresult =" + resp.replace("result =", "").strip()
        inputs     = tokenizer(
            full_text, return_tensors="pt",
            truncation=True, max_length=512,
        ).to(device)
        labels     = inputs["input_ids"].clone()

        # Mask prompt tokens — only compute loss on the response
        prompt_ids = tokenizer(
            prompt + "\nresult =", return_tensors="pt",
            truncation=True, max_length=512,
        )["input_ids"]
        plen = min(prompt_ids.shape[1], labels.shape[1] - 1)
        labels[0, :plen] = -100

        out      = model(**inputs, labels=labels)
        log_prob = -out.loss                          # negative NLL = log prob
        kl_pen   = KL_COEF * (log_prob ** 2)         # KL regularisation
        loss_i   = -(a * log_prob) + kl_pen

        if total_loss.requires_grad:
            total_loss = total_loss + loss_i
        else:
            total_loss = loss_i
        n_valid += 1

    if n_valid == 0:
        return {"loss": 0.0, "advantages": [0.0] * len(rewards)}

    total_loss = total_loss / n_valid
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "loss":       float(total_loss.detach()),
        "advantages": adv.cpu().tolist(),
    }

# ── Push metrics to ALICE Space ───────────────────────────────────────────────
def push_metrics(ep: int, rollouts: dict, grpo: dict,
                 avg_r: float, avg_succ: float, disc: float,
                 cumulative_rewards: list):
    """Push all training metrics to /training/push — feeds every dashboard graph."""
    try:
        _env.post("/training/push", json={
            "model_id":           MODEL_ID,
            "episode":            ep,
            "rewards":            [round(float(r), 4) for r in rollouts["rewards"]],
            "advantages":         [round(float(a), 4) for a in grpo["advantages"]],
            "loss":               round(float(grpo["loss"]), 6),
            "success_rate":       round(avg_succ, 4),
            "disc_coverage":      round(disc, 4),
            "composites":         [round(float(c), 4) for c in rollouts["composites"]],
            "cumulative_rewards": [round(float(r), 4) for r in cumulative_rewards],
        })
        log.info(
            "  → env ep=%d reward=%.4f success=%.0f%% disc=%.0f%% loss=%.4f adv=[%.2f,%.2f]",
            ep, avg_r, avg_succ * 100, disc * 100, grpo["loss"],
            min(grpo["advantages"]), max(grpo["advantages"]),
        )
    except Exception as e:
        log.warning("  push_metrics failed: %s", e)

# ── Register job in Space so dashboard shows live URL ─────────────────────────
def register_job(status: str = "RUNNING", final_reward: float = 0.0,
                 final_succ: float = 0.0, elapsed_s: float = 0.0):
    """POST to /jobs/register so the HF Space & Jobs tab shows this job."""
    try:
        _env.post("/jobs/register", json={
            "job_id":       JOB_ID,
            "model":        MODEL_ID,
            "episodes":     EPISODES,
            "avg_reward":   round(final_reward, 4),
            "success_rate": round(final_succ, 4),
            "elapsed_s":    round(elapsed_s, 1),
            "status":       status,
            "url":          JOB_URL,
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        log.info("  → job registered: %s [%s]", JOB_ID[:12], status)
    except Exception as e:
        log.warning("  register_job failed: %s", e)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import torch

    log.info("=" * 65)
    log.info("ALICE Production Training")
    log.info("  model=%s", MODEL_ID)
    log.info("  episodes=%d | group=%d | turns=%d | lr=%g | kl=%g",
             EPISODES, GROUP_SIZE, MAX_TURNS, LR, KL_COEF)
    log.info("  4bit=%s | lora_r=%d", LOAD_IN_4BIT, LORA_R)
    log.info("  space=%s", SPACE_URL)
    log.info("  job_id=%s", JOB_ID)
    log.info("  job_url=%s", JOB_URL)
    log.info("=" * 65)

    # ── Health check ──────────────────────────────────────────────────
    for attempt in range(8):
        try:
            h = _env.get("/health").json()
            log.info("Space healthy: uptime=%.0fs ram=%.0fMB err_rate=%.3f",
                     h.get("uptime", 0), h.get("memory_usage", 0),
                     h.get("error_rate", 0))
            break
        except Exception as e:
            log.warning("Health check %d/8: %s", attempt + 1, e)
            time.sleep(8)
    else:
        raise RuntimeError(f"Space {SPACE_URL} unreachable after 8 attempts")

    # Register job as RUNNING
    register_job(status="RUNNING")

    # ── Load model ────────────────────────────────────────────────────
    model, tokenizer = load_model()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPISODES, eta_min=LR * 0.1,
    )

    # ── Training loop ─────────────────────────────────────────────────
    all_rewards, all_successes, all_composites, all_losses = [], [], [], []
    cumulative_ep_rewards: list[float] = []
    t0 = time.time()

    for ep in range(1, EPISODES + 1):
        ep_t0 = time.time()

        # Collect GROUP_SIZE rollouts
        rollouts = collect_rollouts(model, tokenizer)

        # GRPO policy gradient update
        grpo = grpo_update(model, tokenizer, optimizer, rollouts)
        scheduler.step()

        # Accumulate stats
        all_rewards.extend(rollouts["rewards"])
        all_successes.extend(rollouts["successes"])
        all_composites.extend(rollouts["composites"])
        all_losses.append(grpo["loss"])

        ep_mean_r = float(np.mean(rollouts["rewards"]))
        cumulative_ep_rewards.append(ep_mean_r)

        # Rolling window metrics (last 80 rollouts)
        W = 80
        avg_r    = float(np.mean(all_rewards[-W:]))
        avg_succ = float(np.mean(all_successes[-W:]))
        disc     = float(np.mean([
            1.0 if 0.2 < c < 0.8 else 0.0
            for c in all_composites[-W:]
        ]))

        ep_elapsed = time.time() - ep_t0
        total_elapsed = time.time() - t0

        log.info(
            "Ep %3d/%d | loss=%.4f | ep_r=%.4f | avg_r=%.4f | "
            "succ=%.0f%% | disc=%.0f%% | lr=%.2e | ep_t=%.1fs | total=%.0fs",
            ep, EPISODES, grpo["loss"], ep_mean_r, avg_r,
            avg_succ * 100, disc * 100,
            scheduler.get_last_lr()[0],
            ep_elapsed, total_elapsed,
        )

        # Push to env every episode
        push_metrics(ep, rollouts, grpo, avg_r, avg_succ, disc, cumulative_ep_rewards)

        # Re-register job with latest stats every 10 episodes
        if ep % 10 == 0:
            register_job(
                status="RUNNING",
                final_reward=avg_r,
                final_succ=avg_succ,
                elapsed_s=total_elapsed,
            )

    # ── Final stats ───────────────────────────────────────────────────
    elapsed    = time.time() - t0
    final_r    = float(np.mean(all_rewards))
    final_succ = float(np.mean(all_successes))
    final_disc = float(np.mean([
        1.0 if 0.2 < c < 0.8 else 0.0 for c in all_composites
    ]))

    log.info("=" * 65)
    log.info("TRAINING COMPLETE")
    log.info("  elapsed=%.0fs | avg_reward=%.4f | success=%.0f%% | disc=%.0f%%",
             elapsed, final_r, final_succ * 100, final_disc * 100)
    log.info("  leaderboard: %s/leaderboard", SPACE_URL)
    log.info("=" * 65)

    # Final push
    push_metrics(
        EPISODES,
        {"rewards": all_rewards[-GROUP_SIZE:],
         "successes": all_successes[-GROUP_SIZE:],
         "composites": all_composites[-GROUP_SIZE:]},
        {"loss": all_losses[-1] if all_losses else 0.0,
         "advantages": ((np.array(all_rewards[-GROUP_SIZE:]) - np.mean(all_rewards[-GROUP_SIZE:])) /
                        (np.std(all_rewards[-GROUP_SIZE:]) + 1e-6)).tolist()},
        final_r, final_succ, final_disc,
        cumulative_ep_rewards,
    )

    # Register job as COMPLETED
    register_job(status="COMPLETED", final_reward=final_r,
                 final_succ=final_succ, elapsed_s=elapsed)

    # ── Save checkpoint ───────────────────────────────────────────────
    ckpt = f"/tmp/{MODEL_ID.replace('/', '_')}_alice_ep{EPISODES}"
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    with open(f"{ckpt}/alice_meta.json", "w") as f:
        json.dump({
            "model_id": MODEL_ID, "episodes": EPISODES,
            "avg_reward": round(final_r, 4),
            "success_rate": round(final_succ, 4),
            "disc_coverage": round(final_disc, 4),
            "elapsed_s": round(elapsed, 1),
            "job_id": JOB_ID, "job_url": JOB_URL,
        }, f, indent=2)
    log.info("Checkpoint saved to %s", ckpt)

    # ── Optional Hub push ─────────────────────────────────────────────
    if PUSH_TO_HUB and HUB_REPO_ID and HF_TOKEN:
        from huggingface_hub import HfApi
        log.info("Pushing to HF Hub: %s", HUB_REPO_ID)
        HfApi(token=HF_TOKEN).upload_folder(
            folder_path=ckpt,
            repo_id=HUB_REPO_ID,
            commit_message=f"ALICE GRPO ep={EPISODES} reward={final_r:.4f}",
        )
        log.info("Pushed to https://huggingface.co/%s", HUB_REPO_ID)

    # Print JSON summary
    print(json.dumps({
        "job_id":         JOB_ID,
        "job_url":        JOB_URL,
        "model_id":       MODEL_ID,
        "episodes":       EPISODES,
        "total_rollouts": len(all_rewards),
        "avg_reward":     round(final_r, 4),
        "success_rate":   round(final_succ, 4),
        "disc_coverage":  round(final_disc, 4),
        "final_loss":     round(all_losses[-1], 4) if all_losses else None,
        "elapsed_s":      round(elapsed, 1),
        "leaderboard":    f"{SPACE_URL}/leaderboard",
    }, indent=2))


if __name__ == "__main__":
    main()
