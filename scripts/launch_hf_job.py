"""
Launch ALICE training as a Hugging Face Job.

Usage:
    python scripts/launch_hf_job.py [--model MODEL_ID] [--episodes N] [--flavor FLAVOR]

Examples:
    python scripts/launch_hf_job.py
    python scripts/launch_hf_job.py --model Qwen/Qwen2.5-0.5B-Instruct --episodes 50 --flavor t4-small
    python scripts/launch_hf_job.py --model Qwen/Qwen2.5-1.5B-Instruct --episodes 200 --flavor a10g-small
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
HF_SPACE_ID = os.environ.get("HF_SPACE_ID", "rohanjain1648/alice-rl-environment")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set. Add it to .env or export HF_TOKEN=hf_...")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Launch ALICE training on HF Jobs")
    p.add_argument("--model",    default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HF model ID to train")
    p.add_argument("--episodes", type=int, default=100,
                   help="Number of training episodes")
    p.add_argument("--group",    type=int, default=4,
                   help="GRPO group size")
    p.add_argument("--turns",    type=int, default=3,
                   help="Max turns per episode")
    p.add_argument("--flavor",   default="t4-small",
                   choices=["cpu-basic", "cpu-upgrade",
                             "t4-small", "t4-medium",
                             "l4x1", "a10g-small", "a10g-large", "a100-large"],
                   help="HF Jobs hardware flavor")
    p.add_argument("--hub-repo", default="",
                   help="HF Hub repo to push trained model (e.g. username/alice-trained)")
    p.add_argument("--timeout",  default="3h",
                   help="Job timeout (e.g. 2h, 90m, 7200)")
    p.add_argument("--no-wait",  action="store_true",
                   help="Submit job and exit without streaming logs")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import run_uv_job, inspect_job, fetch_job_logs, fetch_job_metrics
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    script_path = str(Path(__file__).parent.parent / "training" / "hf_job_train.py")
    if not Path(script_path).exists():
        print(f"ERROR: Training script not found at {script_path}")
        sys.exit(1)

    secrets = {"HF_TOKEN": HF_TOKEN}
    env = {
        "HF_SPACE_ID":    HF_SPACE_ID,
        "MODEL_ID":       args.model,
        "EPISODES":       str(args.episodes),
        "GROUP_SIZE":     str(args.group),
        "MAX_TURNS":      str(args.turns),
        "LR":             "1e-5",
        "KL_COEF":        "0.04",
        "MAX_NEW_TOKENS": "128",
    }
    if args.hub_repo:
        env["HUB_REPO_ID"] = args.hub_repo

    print("=" * 60)
    print("Submitting ALICE training job to HF Jobs")
    print(f"  Model:   {args.model}")
    print(f"  Episodes:{args.episodes} | Group:{args.group} | Turns:{args.turns}")
    print(f"  Flavor:  {args.flavor}")
    print(f"  Timeout: {args.timeout}")
    if args.hub_repo:
        print(f"  Push to: {args.hub_repo}")
    print("=" * 60)

    job = run_uv_job(
        script_path,
        flavor=args.flavor,
        env=env,
        secrets=secrets,
        timeout=args.timeout,
        token=HF_TOKEN,
    )

    print(f"\n✅ Job submitted!")
    print(f"   Job ID:  {job.id}")
    print(f"   Job URL: {job.url}")
    print(f"   Status:  {job.status.stage}")

    if args.no_wait:
        print("\nRun this to stream logs:")
        print(f"  python -c \"from huggingface_hub import fetch_job_logs; "
              f"[print(l) for l in fetch_job_logs('{job.id}')]\"")
        return

    # ── Stream logs ───────────────────────────────────────────────────────────
    print("\nStreaming logs (Ctrl+C to stop watching, job continues)...\n")
    print("-" * 60)

    try:
        last_stage = job.status.stage
        for log_line in fetch_job_logs(job_id=job.id, token=HF_TOKEN):
            print(log_line, end="", flush=True)

        # Final status
        final = inspect_job(job_id=job.id, token=HF_TOKEN)
        print(f"\n{'=' * 60}")
        print(f"Job finished: {final.status.stage}")
        print(f"Job URL: {final.url}")

        if final.status.stage == "COMPLETED":
            print("\n✅ Training complete!")
            if args.hub_repo:
                print(f"   Model at: https://huggingface.co/{args.hub_repo}")
            space_url = f"https://{HF_SPACE_ID.replace('/', '-')}.hf.space"
            print(f"   Leaderboard: {space_url}/leaderboard")
        elif final.status.stage == "ERROR":
            print(f"\n❌ Job failed: {final.status.message}")
            print("   Check full logs at:", final.url)
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n\nStopped watching. Job is still running.")
        print(f"Job URL: {job.url}")
        print(f"To check status: python -c \"from huggingface_hub import inspect_job; "
              f"print(inspect_job('{job.id}').status)\"")


if __name__ == "__main__":
    main()
