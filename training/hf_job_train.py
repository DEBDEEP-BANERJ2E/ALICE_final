# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx", "numpy",
#   "torch", "transformers", "peft", "accelerate",
#   "trl>=0.8.0",
# ]
# ///
"""
ALICE — HF Jobs training script.
Loads a tiny local model, runs TRL GRPO against the live ALICE HF Space,
pushes rewards/success/disc_coverage to the env after every episode.

Lightweight models are used so that they:
Run on t4-small in ~3-5 minutes.
Run on cpu-basic in ~8-10 minutes (slower generation, same logic).

Env vars:
  HF_TOKEN       — HF write token
  HF_SPACE_ID    — e.g. rohanjain1648/alice-rl-environment
  MODEL_ID       — default HuggingFaceTB/SmolLM2-135M-Instruct  (~270MB, fastest)
  EPISODES       — default 20
  GROUP_SIZE     — default 4
  MAX_TURNS      — default 2
  LOAD_IN_4BIT   — default 0  (set 1 on GPU to halve VRAM)
"""
from __future__ import annotations
import logging, os, time, json
import httpx
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s — %(message)s")
log = logging.getLogger("alice.job")

# ── Config ────────────────────────────────────────────────────────────
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HF_SPACE_ID  = os.environ.get("HF_SPACE_ID", "rohanjain1648/alice-rl-environment")
MODEL_ID     = os.environ.get("MODEL_ID",   "HuggingFaceTB/SmolLM2-135M-Instruct")
EPISODES     = int(os.environ.get("EPISODES",    "300"))
GROUP_SIZE   = int(os.environ.get("GROUP_SIZE",   "4"))
MAX_TURNS    = int(os.environ.get("MAX_TURNS",    "2"))
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"
LR           = float(os.environ.get("LR",       "2e-5"))
KL_COEF      = float(os.environ.get("KL_COEF",  "0.04"))
MAX_NEW_TOK  = int(os.environ.get("MAX_NEW_TOKENS", "48"))

SPACE_URL = f"https://{HF_SPACE_ID.replace('/', '-')}.hf.space"
_env = httpx.Client(base_url=SPACE_URL, timeout=20.0)

# ── Load model ────────────────────────────────────────────────────────
def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    log.info("Loading %s (4bit=%s)...", MODEL_ID, LOAD_IN_4BIT)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    ) if LOAD_IN_4BIT else None

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16 if LOAD_IN_4BIT else torch.float32,
    )
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none",
    )
    mdl = get_peft_model(mdl, lora)
    mdl.print_trainable_parameters()
    log.info("Model ready")
    return mdl, tok
# ── Generate response ─────────────────────────────────────────────────
# System prompt that forces the model to always output valid Python
_SYSTEM = (
    "You are a Python solver. For every task, respond with ONLY a single Python "
    "expression that assigns the answer to `result`. Examples:\n"
    "  result = 42\n"
    "  result = 'Canberra'\n"
    "  result = True\n"
    "  result = [x**2 for x in range(5)]\n"
    "Never explain. Never use markdown. Just: result = <value>"
)

def generate(model, tokenizer, task: str) -> str:
    import torch
    device = next(model.parameters()).device
    # Few-shot prompt that strongly biases toward result = ... format
    prompt = (
        f"{_SYSTEM}\n\n"
        f"Task: What is 2+2?\nresult = 4\n\n"
        f"Task: Capital of France?\nresult = 'Paris'\n\n"
        f"Task: {task}\nresult ="
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=384).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=32,
            do_sample=True, temperature=0.6, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
    # Ensure output always starts with result =
    first_line = raw.split("\n")[0].strip()
    return f"result = {first_line}" if not first_line.startswith("result") else first_line
    return tokenizer.decode(gen, skip_special_tokens=True).strip()[:200]

# ── ALICE env calls ───────────────────────────────────────────────────
def env_reset() -> dict:
    return _env.post("/reset").json()

def env_step(ep_id: str, action: str) -> dict:
    return _env.post("/step", json={"episode_id": ep_id, "action": action}).json()

def push_to_env(avg_r: float, avg_succ: float, disc: float, ep: int,
                rewards: list, advantages: list, loss: float, composites: list,
                cumulative_rewards: list):
    """Push all training metrics — feeds reward/advantage/loss/cumulative graphs."""
    try:
        _env.post("/training/push", json={
            "model_id":           MODEL_ID,
            "episode":            ep,
            "rewards":            [round(float(r), 4) for r in rewards],
            "advantages":         [round(float(a), 4) for a in advantages],
            "loss":               round(float(loss), 6),
            "success_rate":       round(avg_succ, 4),
            "disc_coverage":      round(disc, 4),
            "composites":         [round(float(c), 4) for c in composites],
            "cumulative_rewards": [round(float(r), 4) for r in cumulative_rewards],
        })
        log.info("  → env: reward=%.4f success=%.0f%% disc=%.0f%% loss=%.4f ep=%d",
                 avg_r, avg_succ * 100, disc * 100, loss, ep)
    except Exception as e:
        log.warning("  env push failed: %s", e)

# ── Episode + rollout ─────────────────────────────────────────────────
def run_episode(model, tokenizer) -> dict:
    ep      = env_reset()
    ep_id   = ep["episode_id"]
    task    = ep["task"]
    total_r = 0.0
    action  = ""
    composite = 0.0
    for _ in range(MAX_TURNS):
        action = generate(model, tokenizer, task)
        result = env_step(ep_id, action)
        total_r  += float(result.get("reward", 0.01))
        composite = result.get("info", {}).get("verification", {}).get("composite_score", 0.0)
        if result.get("done"):
            break
    return {"reward": total_r, "success": composite >= 0.5, "composite": composite,
            "task": task, "action": action}

def collect_rollouts(model, tokenizer):
    rewards, successes, composites, prompts, responses = [], [], [], [], []
    for _ in range(GROUP_SIZE):
        try:
            ep = run_episode(model, tokenizer)
            rewards.append(ep["reward"])
            successes.append(1.0 if ep["success"] else 0.0)
            composites.append(ep["composite"])
            prompts.append(ep["task"])
            responses.append(ep["action"])
        except Exception as exc:
            log.warning("Rollout error: %s", exc)
            rewards.append(0.01); successes.append(0.0)
            composites.append(0.0); prompts.append(""); responses.append("")
    return rewards, successes, composites, prompts, responses

# ── TRL GRPO update ───────────────────────────────────────────────────
def grpo_update(model, tokenizer, optimizer, prompts, responses, rewards):
    """Minimal GRPO: group-normalised advantage × log-prob loss + KL pen."""
    import torch
    device = next(model.parameters()).device
    r_t    = torch.tensor(rewards, dtype=torch.float32)
    adv    = (r_t - r_t.mean()) / (r_t.std().clamp(min=1e-6))

    total_loss = torch.tensor(0.0, device=device)
    for a, prompt, resp in zip(adv, prompts, responses):
        if not prompt or not resp:
            continue
        text   = prompt + "\n" + resp
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
        labels = inputs["input_ids"].clone()
        plen   = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        labels[0, :plen] = -100
        out      = model(**inputs, labels=labels)
        log_prob = -out.loss
        kl_pen   = KL_COEF * (log_prob ** 2)
        total_loss = total_loss + (-(a.to(device) * log_prob) + kl_pen)

    total_loss = total_loss / max(len(rewards), 1)
    optimizer.zero_grad()
    total_loss.backward()
    import torch.nn as nn
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(total_loss.detach())

# ── Main ──────────────────────────────────────────────────────────────
def main():
    import torch
    log.info("=" * 60)
    log.info("ALICE HF Job | model=%s | episodes=%d | group=%d",
             MODEL_ID, EPISODES, GROUP_SIZE)
    log.info("space=%s", SPACE_URL)
    log.info("=" * 60)

    # Health check
    for attempt in range(6):
        try:
            h = _env.get("/health").json()
            log.info("Space healthy: uptime=%.0fs ram=%.0fMB",
                     h.get("uptime", 0), h.get("memory_usage", 0))
            break
        except Exception as e:
            log.warning("Health %d/6: %s", attempt + 1, e)
            time.sleep(5)
    else:
        raise RuntimeError(f"Space {SPACE_URL} unreachable")

    model, tokenizer = load_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    all_rewards, all_successes, all_composites, all_losses = [], [], [], []
    cumulative_ep_rewards = []   # mean reward per episode — should trend upward
    t0 = time.time()

    for ep in range(1, EPISODES + 1):
        rewards, successes, composites, prompts, responses = collect_rollouts(model, tokenizer)

        loss = grpo_update(model, tokenizer, optimizer, prompts, responses, rewards)

        all_rewards.extend(rewards)
        all_successes.extend(successes)
        all_composites.extend(composites)
        all_losses.append(loss)

        ep_mean_r = float(np.mean(rewards))
        cumulative_ep_rewards.append(ep_mean_r)

        avg_r    = float(np.mean(all_rewards[-40:]))
        avg_succ = float(np.mean(all_successes[-40:]))
        disc     = float(np.mean([1.0 if 0.2 < c < 0.8 else 0.0
                                   for c in all_composites[-40:]]))

        # GRPO advantages for this episode
        r_arr = np.array(rewards, dtype=np.float32)
        adv   = ((r_arr - r_arr.mean()) / (r_arr.std() + 1e-6)).tolist()

        log.info("Ep %2d/%d | loss=%.4f | ep_reward=%.4f | avg_reward=%.4f | "
                 "success=%.0f%% | disc=%.0f%% | adv=[%.2f,%.2f] | %.0fs",
                 ep, EPISODES, loss, ep_mean_r, avg_r, avg_succ * 100, disc * 100,
                 min(adv), max(adv), time.time() - t0)

        push_to_env(avg_r, avg_succ, disc, ep, rewards, adv, loss, composites,
                    cumulative_ep_rewards)

    elapsed = time.time() - t0
    final_r    = float(np.mean(all_rewards))
    final_succ = float(np.mean(all_successes))
    final_disc = float(np.mean([1.0 if 0.2 < c < 0.8 else 0.0
                                 for c in all_composites]))

    log.info("=" * 60)
    log.info("DONE %.0fs | reward=%.4f | success=%.0f%% | disc=%.0f%%",
             elapsed, final_r, final_succ * 100, final_disc * 100)
    log.info("=" * 60)

    push_to_env(final_r, final_succ, final_disc, EPISODES,
                all_rewards[-GROUP_SIZE:],
                ((np.array(all_rewards[-GROUP_SIZE:]) - np.mean(all_rewards[-GROUP_SIZE:])) /
                 (np.std(all_rewards[-GROUP_SIZE:]) + 1e-6)).tolist(),
                all_losses[-1] if all_losses else 0.0,
                all_composites[-GROUP_SIZE:],
                cumulative_ep_rewards)

    print(json.dumps({
        "model_id":       MODEL_ID,
        "episodes":       EPISODES,
        "total_rollouts": len(all_rewards),
        "avg_reward":     round(final_r,    4),
        "success_rate":   round(final_succ, 4),
        "disc_coverage":  round(final_disc, 4),
        "final_loss":     round(all_losses[-1], 4) if all_losses else None,
        "elapsed_s":      round(elapsed, 1),
        "leaderboard":    f"{SPACE_URL}/leaderboard",
    }, indent=2))

if __name__ == "__main__":
    main()
