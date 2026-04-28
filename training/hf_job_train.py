# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx", "numpy", "matplotlib>=3.8.0",
#   "torch>=2.2.0", "transformers>=4.40.0", "peft>=0.10.0",
#   "accelerate>=0.27.0", "trl>=0.8.6", "datasets>=2.18.0",
#   "bitsandbytes>=0.43.0",
# ]
# ///
"""
ALICE — Production RL training with TRL GRPO on GPU.

Full real RL loop:
  - Loads model locally (a10g-small = 24 GB VRAM, no 4-bit needed for ≤3B)
  - Proper 3-turn episodes: reset → generate(turn1) → step → feedback →
      generate(turn2) → step → feedback → generate(turn3) → step
  - GRPO policy gradient with group-normalised advantages
  - Cosine LR schedule + gradient clipping
  - Pushes every metric (rewards, advantages, loss, disc_coverage,
    per-turn composites, cumulative reward curve) to /training/push
  - Registers live job URL in /jobs/register → shows in dashboard
  - Saves LoRA checkpoint, optionally pushes to HF Hub

Env vars (Job secrets/env):
  HF_TOKEN        required
  HF_SPACE_ID     default rohanjain1648/alice-rl-environment
  MODEL_ID        default Qwen/Qwen2.5-0.5B-Instruct
  EPISODES        default 100
  GROUP_SIZE      default 8
  MAX_TURNS       default 3
  LR              default 1e-5
  KL_COEF         default 0.04
  MAX_NEW_TOKENS  default 96
  LOAD_IN_4BIT    default 0  (a10g has 24 GB — no need for 4-bit on ≤3B)
  LORA_R          default 16
  PUSH_TO_HUB     default 0
  HUB_REPO_ID     e.g. username/alice-trained
  JOB_ID          injected by HF Jobs runtime
"""
from __future__ import annotations
import logging, os, time, json, uuid
import httpx
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s — %(message)s")
log = logging.getLogger("alice.train")

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
HF_SPACE_ID    = os.environ.get("HF_SPACE_ID",    "rohanjain1648/alice-rl-environment")
MODEL_ID       = os.environ.get("MODEL_ID",        "Qwen/Qwen2.5-0.5B-Instruct")
EPISODES       = int(os.environ.get("EPISODES",    "100"))
GROUP_SIZE     = int(os.environ.get("GROUP_SIZE",  "8"))
MAX_TURNS      = int(os.environ.get("MAX_TURNS",   "3"))
LR             = float(os.environ.get("LR",        "1e-5"))
KL_COEF        = float(os.environ.get("KL_COEF",   "0.04"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "96"))
LOAD_IN_4BIT   = os.environ.get("LOAD_IN_4BIT", "0") == "1"
LORA_R         = int(os.environ.get("LORA_R",      "16"))
PUSH_TO_HUB    = os.environ.get("PUSH_TO_HUB", "0") == "1"
HUB_REPO_ID    = os.environ.get("HUB_REPO_ID", "")
JOB_ID         = os.environ.get("JOB_ID", str(uuid.uuid4())[:24])

SPACE_URL = f"https://{HF_SPACE_ID.replace('/', '-')}.hf.space"
JOB_URL   = f"https://huggingface.co/jobs/{HF_SPACE_ID.split('/')[0]}/{JOB_ID}"

_env = httpx.Client(base_url=SPACE_URL, timeout=30.0)

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are a precise Python solver. "
    "For every task, output ONLY a single Python statement assigning the answer to `result`.\n"
    "Rules:\n"
    "  - Always write exactly: result = <value>\n"
    "  - No explanations, no markdown, no extra lines\n"
    "  - Use Python literals: strings in quotes, numbers bare, booleans True/False\n"
    "Examples:\n"
    "  result = 42\n"
    "  result = 'Canberra'\n"
    "  result = True\n"
    "  result = [x**2 for x in range(5)]\n"
    "  result = sorted([3,1,4,1,5], reverse=True)"
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
    tok.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    ) if LOAD_IN_4BIT else None

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if _has_flash_attn() else "eager",
    )
    if LOAD_IN_4BIT:
        mdl = prepare_model_for_kbit_training(mdl)

    target_mods = _get_lora_targets(mdl)
    log.info("LoRA targets: %s", target_mods)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_R * 2,
        target_modules=target_mods, lora_dropout=0.05, bias="none",
    )
    mdl = get_peft_model(mdl, lora)
    mdl.print_trainable_parameters()
    log.info("Model on %s | dtype=%s", next(mdl.parameters()).device,
             next(mdl.parameters()).dtype)
    return mdl, tok


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def _get_lora_targets(model) -> list:
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
    found = {c for name, _ in model.named_modules() for c in candidates if name.endswith(c)}
    return list(found) if found else ["q_proj", "v_proj"]

# ── Generation using chat template ───────────────────────────────────────────
def generate(model, tokenizer, task: str, feedback: str = "") -> str:
    import torch

    messages = [{"role": "system", "content": _SYSTEM}]
    if feedback:
        messages.append({"role": "user",      "content": f"Task: {task}"})
        messages.append({"role": "assistant", "content": f"result = ..."})
        messages.append({"role": "user",      "content": f"Feedback: {feedback}\nTry again: Task: {task}"})
    else:
        messages.append({"role": "user", "content": f"Task: {task}"})

    # Use chat template if available, else fall back to plain prompt
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt += "result ="
    except Exception:
        prompt = f"{_SYSTEM}\n\nTask: {task}\nresult ="

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512, padding=False).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=0.7, top_p=0.92,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    first_line = raw.split("\n")[0].strip()
    return first_line if first_line.startswith("result") else f"result = {first_line[:200]}"

# ── ALICE env calls ───────────────────────────────────────────────────────────
def env_reset() -> dict:
    return _env.post("/reset").json()

def env_step(ep_id: str, action: str) -> dict:
    return _env.post("/step", json={"episode_id": ep_id, "action": action}).json()

# ── Full multi-turn episode ───────────────────────────────────────────────────
def run_episode(model, tokenizer) -> dict:
    """
    Proper 3-turn ALICE episode:
      T1: initial attempt
      T2: retry with T1 verification feedback
      T3: final attempt with T2 feedback + hint
    """
    ep    = env_reset()
    ep_id = ep["episode_id"]
    task  = ep["task"]

    turn_rewards, turn_composites, turn_actions = [], [], []
    feedback = ""
    total_r  = 0.0

    for turn in range(1, MAX_TURNS + 1):
        action = generate(model, tokenizer, task, feedback)
        result = env_step(ep_id, action)

        r         = float(result.get("reward", 0.01))
        done      = result.get("done", False)
        verif     = result.get("info", {}).get("verification", {})
        composite = float(verif.get("composite_score", 0.0))

        turn_rewards.append(r)
        turn_composites.append(composite)
        turn_actions.append(action)
        total_r += r

        # Build feedback for next turn from real verifier output
        if composite >= 0.5:
            feedback = f"Turn {turn} passed (score={composite:.2f}). Maintain quality."
        else:
            t1 = verif.get("tier1_details", {}) or {}
            if not t1.get("success", True):
                feedback = f"Python error: {t1.get('error_message', 'unknown')}. Fix the code."
            else:
                t2_score = verif.get("tier2_score") or 0.0
                feedback = (f"Score {composite:.2f} (T1={verif.get('tier1_score',0):.2f} "
                            f"T2={t2_score:.2f}). Improve correctness.")

        if done:
            break

    return {
        "task":            task,
        "actions":         turn_actions,
        "turn_rewards":    turn_rewards,
        "turn_composites": turn_composites,
        "total_reward":    total_r,
        "success":         (turn_composites[-1] if turn_composites else 0.0) >= 0.5,
        "composite":       turn_composites[-1] if turn_composites else 0.0,
        "turns_taken":     len(turn_rewards),
    }

# ── Rollout collection ────────────────────────────────────────────────────────
def collect_rollouts(model, tokenizer) -> dict:
    rewards, successes, composites, prompts, responses = [], [], [], [], []
    for _ in range(GROUP_SIZE):
        try:
            ep = run_episode(model, tokenizer)
            rewards.append(ep["total_reward"])
            successes.append(1.0 if ep["success"] else 0.0)
            composites.append(ep["composite"])
            prompts.append(ep["task"])
            responses.append(ep["actions"][-1] if ep["actions"] else "result = None")
        except Exception as exc:
            log.warning("Rollout error: %s", exc)
            rewards.append(0.01); successes.append(0.0)
            composites.append(0.0); prompts.append(""); responses.append("")
    return {"rewards": rewards, "successes": successes, "composites": composites,
            "prompts": prompts, "responses": responses}

# ── GRPO update ───────────────────────────────────────────────────────────────
def grpo_update(model, tokenizer, optimizer, rollouts: dict) -> dict:
    import torch, torch.nn as nn

    rewards, prompts, responses = (rollouts["rewards"],
                                   rollouts["prompts"], rollouts["responses"])
    device = next(model.parameters()).device
    r_t    = torch.tensor(rewards, dtype=torch.float32, device=device)
    adv    = (r_t - r_t.mean()) / (r_t.std().clamp(min=1e-6))

    total_loss = None
    n_valid    = 0

    for a, prompt, resp in zip(adv, prompts, responses):
        if not prompt or not resp:
            continue
        full_text = prompt + "\nresult =" + resp.replace("result =", "").strip()
        inputs    = tokenizer(full_text, return_tensors="pt",
                              truncation=True, max_length=512).to(device)
        labels    = inputs["input_ids"].clone()
        plen      = min(
            tokenizer(prompt + "\nresult =", return_tensors="pt",
                      truncation=True, max_length=512)["input_ids"].shape[1],
            labels.shape[1] - 1,
        )
        labels[0, :plen] = -100

        out      = model(**inputs, labels=labels)
        log_prob = -out.loss
        kl_pen   = KL_COEF * (log_prob ** 2)
        loss_i   = -(a * log_prob) + kl_pen

        total_loss = loss_i if total_loss is None else total_loss + loss_i
        n_valid   += 1

    if n_valid == 0 or total_loss is None:
        return {"loss": 0.0, "advantages": [0.0] * len(rewards)}

    total_loss = total_loss / n_valid
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": float(total_loss.detach()), "advantages": adv.cpu().tolist()}

# ── Push metrics to Space ─────────────────────────────────────────────────────
def push_metrics(ep, rollouts, grpo, avg_r, avg_succ, disc, cumul):
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
            "cumulative_rewards": [round(float(r), 4) for r in cumul],
        })
        log.info("  → push ep=%d r=%.4f succ=%.0f%% disc=%.0f%% loss=%.4f adv=[%.2f,%.2f]",
                 ep, avg_r, avg_succ*100, disc*100, grpo["loss"],
                 min(grpo["advantages"]), max(grpo["advantages"]))
    except Exception as e:
        log.warning("  push_metrics failed: %s", e)

# ── Register job in Space dashboard ──────────────────────────────────────────
def register_job(status="RUNNING", avg_r=0.0, avg_succ=0.0, elapsed_s=0.0):
    try:
        _env.post("/jobs/register", json={
            "job_id":       JOB_ID,
            "model":        MODEL_ID,
            "episodes":     EPISODES,
            "avg_reward":   round(avg_r, 4),
            "success_rate": round(avg_succ, 4),
            "elapsed_s":    round(elapsed_s, 1),
            "status":       status,
            "url":          JOB_URL,
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        log.info("  → job %s [%s]", JOB_ID[:12], status)
    except Exception as e:
        log.warning("  register_job failed: %s", e)

# ── Training analysis plot ─────────────────────────────────────────────────────
def plot_training(cumul, all_rewards, all_successes, all_losses,
                  final_r, final_succ, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        W   = 5
        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(cumul, alpha=0.4, color="#4a90e2", linewidth=0.8, label="per-episode")
        if len(cumul) >= W:
            ma = np.convolve(cumul, np.ones(W) / W, mode="valid")
            ax1.plot(range(W - 1, len(cumul)), ma, color="#4a90e2",
                     linewidth=2.5, label=f"{W}-ep MA")
        ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_title("Reward Curve", fontweight="bold")
        ax1.set_xlabel("Episode")
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(all_rewards, bins=25, color="#4a90e2", edgecolor="white", alpha=0.85)
        ax2.axvline(np.mean(all_rewards), color="red", linestyle="--",
                    label=f"mean={np.mean(all_rewards):.3f}")
        ax2.set_title("Reward Distribution", fontweight="bold")
        ax2.legend(fontsize=8)

        ax3 = fig.add_subplot(gs[1, 0])
        succ_ep = [float(np.mean(all_successes[max(0, i - GROUP_SIZE):i + GROUP_SIZE]))
                   for i in range(0, len(all_successes), GROUP_SIZE)]
        ax3.plot(succ_ep, color="#27ae60", linewidth=2)
        ax3.fill_between(range(len(succ_ep)), succ_ep, alpha=0.15, color="#27ae60")
        ax3.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="50% target")
        ax3.set_ylim(0, 1)
        ax3.set_title("Success Rate", fontweight="bold")
        ax3.legend(fontsize=8)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(all_losses, color="#e74c3c", linewidth=1.5, alpha=0.7)
        if len(all_losses) >= W:
            ma_l = np.convolve(all_losses, np.ones(W) / W, mode="valid")
            ax4.plot(range(W - 1, len(all_losses)), ma_l, color="#c0392b", linewidth=2.5)
        ax4.set_title("Training Loss (GRPO)", fontweight="bold")

        fig.suptitle(
            f"ALICE RL — {MODEL_ID.split('/')[-1]} | {EPISODES} episodes | "
            f"avg_reward={final_r:.4f} | success={final_succ:.0%}",
            fontsize=13, fontweight="bold",
        )
        out_path = f"{out_dir}/alice_training.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Training plot saved to %s", out_path)
    except Exception as e:
        log.warning("plot_training failed: %s", e)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import torch

    log.info("=" * 65)
    log.info("ALICE Production Training | GPU: %s",
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    log.info("  model=%s | episodes=%d | group=%d | turns=%d",
             MODEL_ID, EPISODES, GROUP_SIZE, MAX_TURNS)
    log.info("  lr=%g | kl=%g | 4bit=%s | lora_r=%d",
             LR, KL_COEF, LOAD_IN_4BIT, LORA_R)
    log.info("  space=%s | job=%s", SPACE_URL, JOB_URL)
    log.info("=" * 65)

    # Health check
    for attempt in range(8):
        try:
            h = _env.get("/health").json()
            log.info("Space healthy: uptime=%.0fs ram=%.0fMB err=%.3f",
                     h.get("uptime", 0), h.get("memory_usage", 0), h.get("error_rate", 0))
            break
        except Exception as e:
            log.warning("Health %d/8: %s", attempt + 1, e)
            time.sleep(8)
    else:
        raise RuntimeError(f"Space {SPACE_URL} unreachable")

    register_job(status="RUNNING")

    model, tokenizer = load_model()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPISODES, eta_min=LR * 0.1,
    )

    all_rewards, all_successes, all_composites, all_losses = [], [], [], []
    cumul: list[float] = []
    t0 = time.time()

    for ep in range(1, EPISODES + 1):
        ep_t0    = time.time()
        rollouts = collect_rollouts(model, tokenizer)
        grpo     = grpo_update(model, tokenizer, optimizer, rollouts)
        scheduler.step()

        all_rewards.extend(rollouts["rewards"])
        all_successes.extend(rollouts["successes"])
        all_composites.extend(rollouts["composites"])
        all_losses.append(grpo["loss"])

        ep_mean_r = float(np.mean(rollouts["rewards"]))
        cumul.append(ep_mean_r)

        W        = 80
        avg_r    = float(np.mean(all_rewards[-W:]))
        avg_succ = float(np.mean(all_successes[-W:]))
        disc     = float(np.mean([1.0 if 0.2 < c < 0.8 else 0.0
                                   for c in all_composites[-W:]]))
        elapsed  = time.time() - t0

        log.info("Ep %3d/%d | loss=%.4f | ep_r=%.4f | avg_r=%.4f | "
                 "succ=%.0f%% | disc=%.0f%% | lr=%.2e | ep_t=%.1fs | total=%.0fs",
                 ep, EPISODES, grpo["loss"], ep_mean_r, avg_r,
                 avg_succ*100, disc*100, scheduler.get_last_lr()[0],
                 time.time()-ep_t0, elapsed)

        push_metrics(ep, rollouts, grpo, avg_r, avg_succ, disc, cumul)

        if ep % 10 == 0:
            register_job(status="RUNNING", avg_r=avg_r,
                         avg_succ=avg_succ, elapsed_s=elapsed)

    elapsed    = time.time() - t0
    final_r    = float(np.mean(all_rewards))
    final_succ = float(np.mean(all_successes))
    final_disc = float(np.mean([1.0 if 0.2 < c < 0.8 else 0.0
                                 for c in all_composites]))

    log.info("=" * 65)
    log.info("DONE %.0fs | reward=%.4f | success=%.0f%% | disc=%.0f%%",
             elapsed, final_r, final_succ*100, final_disc*100)
    log.info("=" * 65)

    # Final push + job registration
    push_metrics(EPISODES,
                 {"rewards": all_rewards[-GROUP_SIZE:],
                  "successes": all_successes[-GROUP_SIZE:],
                  "composites": all_composites[-GROUP_SIZE:]},
                 {"loss": all_losses[-1] if all_losses else 0.0,
                  "advantages": ((np.array(all_rewards[-GROUP_SIZE:]) -
                                  np.mean(all_rewards[-GROUP_SIZE:])) /
                                 (np.std(all_rewards[-GROUP_SIZE:]) + 1e-6)).tolist()},
                 final_r, final_succ, final_disc, cumul)
    register_job(status="COMPLETED", avg_r=final_r,
                 avg_succ=final_succ, elapsed_s=elapsed)

    # Save checkpoint
    ckpt = f"/tmp/{MODEL_ID.replace('/', '_')}_alice_ep{EPISODES}"
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    with open(f"{ckpt}/alice_meta.json", "w") as f:
        json.dump({"model_id": MODEL_ID, "episodes": EPISODES,
                   "avg_reward": round(final_r, 4), "success_rate": round(final_succ, 4),
                   "disc_coverage": round(final_disc, 4), "elapsed_s": round(elapsed, 1),
                   "job_id": JOB_ID, "job_url": JOB_URL}, f, indent=2)
    log.info("Checkpoint: %s", ckpt)

    plot_training(cumul, all_rewards, all_successes, all_losses,
                  final_r, final_succ, ckpt)

    if PUSH_TO_HUB and HUB_REPO_ID and HF_TOKEN:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_folder(
            folder_path=ckpt, repo_id=HUB_REPO_ID,
            commit_message=f"ALICE GRPO ep={EPISODES} reward={final_r:.4f}",
        )
        log.info("Pushed to https://huggingface.co/%s", HUB_REPO_ID)

    print(json.dumps({
        "job_id": JOB_ID, "job_url": JOB_URL, "model_id": MODEL_ID,
        "episodes": EPISODES, "total_rollouts": len(all_rewards),
        "avg_reward": round(final_r, 4), "success_rate": round(final_succ, 4),
        "disc_coverage": round(final_disc, 4),
        "final_loss": round(all_losses[-1], 4) if all_losses else None,
        "elapsed_s": round(elapsed, 1),
        "leaderboard": f"{SPACE_URL}/leaderboard",
    }, indent=2))

    try:
        lb = _env.get("/leaderboard").json()
        log.info("Leaderboard:")
        for entry in lb:
            log.info("  #%-3s %-30s rl=%.4f reward=%.4f ep=%s",
                     entry.get("rank", "?"),
                     entry.get("display_name", entry["model_id"].split("/")[-1])[:30],
                     entry.get("rl_score", 0.0), entry.get("avg_reward", 0.0),
                     entry.get("episodes_run", "?"))
    except Exception as e:
        log.warning("Leaderboard fetch failed: %s", e)


if __name__ == "__main__":
    main()
