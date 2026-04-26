# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "numpy"]
# ///
"""
ALICE — CPU-friendly eval/training using HF Inference API.
No local model loading. Runs on cpu-basic (free HF Jobs tier).

Runs multi-turn ALICE episodes via the HF Inference API, pushes
per-episode metrics to the Space dashboard, and updates the leaderboard.

Env vars (Job secrets/env):
  HF_TOKEN        required (for Inference API auth)
  HF_SPACE_ID     default rohanjain1648/alice-rl-environment
  MODEL_ID        default Qwen/Qwen2.5-0.5B-Instruct
  EPISODES        default 50
  MAX_TURNS       default 3
  JOB_ID          injected by HF Jobs runtime
"""
from __future__ import annotations
import logging, os, time, json, uuid
import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
log = logging.getLogger("alice.cpu_train")

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
HF_SPACE_ID = os.environ.get("HF_SPACE_ID", "rohanjain1648/alice-rl-environment")
MODEL_ID    = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
EPISODES    = int(os.environ.get("EPISODES", "50"))
MAX_TURNS   = int(os.environ.get("MAX_TURNS", "3"))
JOB_ID      = os.environ.get("JOB_ID", str(uuid.uuid4())[:24])

SPACE_URL = f"https://{HF_SPACE_ID.replace('/', '-')}.hf.space"
JOB_URL   = f"https://huggingface.co/jobs/{HF_SPACE_ID.split('/')[0]}/{JOB_ID}"
INFER_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}/v1/chat/completions"

_env   = httpx.Client(base_url=SPACE_URL, timeout=30.0)
_infer = httpx.Client(timeout=45.0)
_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

_SYSTEM = (
    "You are a precise Python solver. "
    "Output ONLY a single Python statement assigning the answer to `result`.\n"
    "Rules:\n"
    "  - Always write exactly: result = <value>\n"
    "  - No explanations, no markdown, no extra lines\n"
    "  - Use Python literals: strings in quotes, numbers bare, booleans True/False\n"
    "Examples:\n"
    "  result = 42\n"
    "  result = 'Canberra'\n"
    "  result = True\n"
    "  result = [x**2 for x in range(5)]"
)


def infer(task: str, feedback: str = "") -> str:
    messages = [{"role": "system", "content": _SYSTEM}]
    if feedback:
        messages += [
            {"role": "user",      "content": f"Task: {task}"},
            {"role": "assistant", "content": "result = ..."},
            {"role": "user",      "content": f"Feedback: {feedback}\nRetry: {task}"},
        ]
    else:
        messages.append({"role": "user", "content": f"Task: {task}"})
    try:
        r    = _infer.post(INFER_URL, headers=_HEADERS,
                           json={"model": MODEL_ID, "max_tokens": 64, "temperature": 0.7,
                                 "messages": messages})
        raw  = r.json()["choices"][0]["message"]["content"].strip()
        first = raw.split("\n")[0].strip()
        return first if first.startswith("result") else f"result = {first[:200]}"
    except Exception as e:
        log.warning("infer failed: %s", e)
        return "result = None"


def run_episode() -> dict:
    ep    = _env.post("/reset").json()
    ep_id = ep["episode_id"]
    task  = ep["task"]
    feedback = ""
    turn_rewards, turn_composites = [], []
    for turn in range(1, MAX_TURNS + 1):
        action = infer(task, feedback)
        result = _env.post("/step", json={"episode_id": ep_id, "action": action}).json()
        r         = float(result.get("reward", 0.0))
        verif     = result.get("info", {}).get("verification", {})
        composite = float(verif.get("composite_score", 0.0))
        turn_rewards.append(r)
        turn_composites.append(composite)
        if composite >= 0.5:
            feedback = f"Turn {turn} passed (score={composite:.2f}). Maintain quality."
        else:
            t1 = verif.get("tier1_details", {}) or {}
            if not t1.get("success", True):
                feedback = f"Python error: {t1.get('error_message', 'unknown')}. Fix the code."
            else:
                t2 = verif.get("tier2_score") or 0.0
                feedback = (f"Score {composite:.2f} (T1={verif.get('tier1_score', 0):.2f} "
                            f"T2={t2:.2f}). Improve correctness.")
        if result.get("done", False):
            break
    return {
        "rewards":    turn_rewards,
        "composites": turn_composites,
        "success":    (turn_composites[-1] if turn_composites else 0.0) >= 0.5,
        "composite":  turn_composites[-1] if turn_composites else 0.0,
    }


def push_metrics(ep, rewards, composites, avg_r, avg_succ, disc, cumul):
    try:
        adv_arr = np.array(rewards, dtype=float)
        if adv_arr.std() > 1e-6:
            adv = ((adv_arr - adv_arr.mean()) / adv_arr.std()).tolist()
        else:
            adv = [0.0] * len(rewards)
        _env.post("/training/push", json={
            "model_id":           MODEL_ID,
            "episode":            ep,
            "rewards":            [round(float(r), 4) for r in rewards],
            "advantages":         [round(float(a), 4) for a in adv],
            "loss":               0.0,
            "success_rate":       round(avg_succ, 4),
            "disc_coverage":      round(disc, 4),
            "composites":         [round(float(c), 4) for c in composites],
            "cumulative_rewards": [round(float(r), 4) for r in cumul],
        })
    except Exception as e:
        log.warning("push_metrics failed: %s", e)


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
        log.warning("register_job failed: %s", e)


def main():
    log.info("=" * 65)
    log.info("ALICE CPU Eval | model=%s | episodes=%d | turns=%d",
             MODEL_ID, EPISODES, MAX_TURNS)
    log.info("  space=%s", SPACE_URL)
    log.info("  job=%s", JOB_URL)
    log.info("=" * 65)

    for attempt in range(8):
        try:
            h = _env.get("/health").json()
            log.info("Space healthy: uptime=%.0fs ram=%.0fMB",
                     h.get("uptime", 0), h.get("memory_usage", 0))
            break
        except Exception as e:
            log.warning("Health %d/8: %s", attempt + 1, e)
            time.sleep(8)
    else:
        raise RuntimeError(f"Space {SPACE_URL} unreachable after 8 attempts")

    register_job(status="RUNNING")

    all_rewards, all_composites, all_successes = [], [], []
    cumul: list[float] = []
    t0 = time.time()

    for ep in range(1, EPISODES + 1):
        try:
            ep_data       = run_episode()
            ep_rewards    = ep_data["rewards"]
            ep_composites = ep_data["composites"]
            ep_success    = ep_data["success"]
        except Exception as exc:
            log.warning("Episode %d failed: %s", ep, exc)
            ep_rewards    = [0.0]
            ep_composites = [0.0]
            ep_success    = False

        all_rewards.extend(ep_rewards)
        all_composites.extend(ep_composites)
        all_successes.append(1.0 if ep_success else 0.0)
        cumul.append(float(np.mean(ep_rewards)))

        W        = min(len(all_rewards), 80)
        avg_r    = float(np.mean(all_rewards[-W:]))
        sw       = min(len(all_successes), 20)
        avg_succ = float(np.mean(all_successes[-sw:]))
        disc     = float(np.mean([1.0 if 0.2 < c < 0.8 else 0.0
                                   for c in all_composites[-W:]]))
        elapsed  = time.time() - t0

        log.info("Ep %3d/%d | avg_r=%.4f | succ=%.0f%% | disc=%.0f%% | %.0fs",
                 ep, EPISODES, avg_r, avg_succ * 100, disc * 100, elapsed)

        push_metrics(ep, ep_rewards, ep_composites, avg_r, avg_succ, disc, cumul)

        if ep % 10 == 0:
            register_job(status="RUNNING", avg_r=avg_r,
                         avg_succ=avg_succ, elapsed_s=elapsed)

    elapsed    = time.time() - t0
    final_r    = float(np.mean(all_rewards))    if all_rewards    else 0.0
    final_succ = float(np.mean(all_successes))  if all_successes  else 0.0
    final_disc = float(np.mean([1.0 if 0.2 < c < 0.8 else 0.0
                                 for c in all_composites])) if all_composites else 0.0

    log.info("=" * 65)
    log.info("DONE %.0fs | reward=%.4f | success=%.0f%% | disc=%.0f%%",
             elapsed, final_r, final_succ * 100, final_disc * 100)
    log.info("=" * 65)

    push_metrics(EPISODES,
                 all_rewards[-8:] if all_rewards else [0.0],
                 all_composites[-8:] if all_composites else [0.0],
                 final_r, final_succ, final_disc, cumul)
    register_job(status="COMPLETED", avg_r=final_r,
                 avg_succ=final_succ, elapsed_s=elapsed)

    print(json.dumps({
        "job_id":       JOB_ID,
        "job_url":      JOB_URL,
        "model_id":     MODEL_ID,
        "episodes":     EPISODES,
        "avg_reward":   round(final_r,    4),
        "success_rate": round(final_succ, 4),
        "disc_coverage": round(final_disc, 4),
        "elapsed_s":    round(elapsed, 1),
    }))


if __name__ == "__main__":
    main()
