"""
Launch ALICE training as a Hugging Face Job.

Usage:
    python scripts/launch_hf_job.py
    python scripts/launch_hf_job.py --episodes 50
    python scripts/launch_hf_job.py --model Qwen/Qwen2.5-Coder-3B-Instruct --episodes 100

Model/provider defaults:
    model    = Qwen/Qwen2.5-Coder-3B-Instruct  (cheapest: $0.01/M via nscale)
    provider = nscale
    flavor   = cpu-basic  (no GPU needed — inference is remote)
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

# Load .env
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
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--group",    type=int, default=8)
    p.add_argument("--turns",    type=int, default=3)
    p.add_argument("--4bit",     dest="load_4bit", action="store_true",
                   help="Enable 4-bit quantisation (GPU only)")
    p.add_argument("--flavor",   default="a10g-small",
                   choices=["cpu-basic", "cpu-upgrade", "t4-small", "t4-medium",
                             "l4x1", "a10g-small", "a10g-large", "l40sx1"])
    p.add_argument("--no-wait",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import run_uv_job, inspect_job, fetch_job_logs
    except ImportError:
        print("pip install huggingface_hub")
        sys.exit(1)

    # Pre-set namespace to avoid whoami rate-limit call inside run_uv_job
    os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
    namespace = HF_SPACE_ID.split("/")[0]  # "rohanjain1648"

    script_path = str(Path(__file__).parent.parent / "training" / "hf_job_train.py")

    print("=" * 60)
    print("Submitting ALICE training job")
    print(f"  Model:    {args.model}")
    print(f"  Episodes: {args.episodes} | Group: {args.group} | Turns: {args.turns}")
    print(f"  Flavor:   {args.flavor} | 4bit: {args.load_4bit}")
    print("=" * 60)

    job = run_uv_job(
        script_path,
        flavor=args.flavor,
        namespace=namespace,
        env={
            "HF_SPACE_ID":    HF_SPACE_ID,
            "MODEL_ID":       args.model,
            "EPISODES":       str(args.episodes),
            "GROUP_SIZE":     str(args.group),
            "MAX_TURNS":      str(args.turns),
            "LOAD_IN_4BIT":   "1" if args.load_4bit else "0",
        },
        secrets={"HF_TOKEN": HF_TOKEN},
        token=HF_TOKEN,
    )

    print(f"\n✅ Job submitted!")
    print(f"   Job ID:  {job.id}")
    print(f"   Job URL: {job.url}")
    print(f"   Status:  {job.status.stage}")
    print(f"   Model:   {args.model} on {args.flavor}")

    if args.no_wait:
        print(f"\nStream logs: python -c \"from huggingface_hub import fetch_job_logs; "
              f"[print(l,end='') for l in fetch_job_logs(job_id='{job.id}', token='{HF_TOKEN}')]\"")
        return

    print("\nStreaming logs...\n" + "-" * 60)
    try:
        for line in fetch_job_logs(job_id=job.id, token=HF_TOKEN):
            print(line, end="", flush=True)

        final = inspect_job(job_id=job.id, token=HF_TOKEN)
        print(f"\n{'=' * 60}")
        print(f"Job finished: {final.status.stage}")
        space_url = f"https://{HF_SPACE_ID.replace('/', '-')}.hf.space"
        if final.status.stage == "COMPLETED":
            print(f"✅ Done! Leaderboard: {space_url}/leaderboard")
        else:
            print(f"❌ Failed: {getattr(final.status, 'message', '')}")
            print(f"   Logs: {final.url}")
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nStopped watching. Job still running: {job.url}")


if __name__ == "__main__":
    main()
