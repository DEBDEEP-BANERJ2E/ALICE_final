"""
Launch ALICE training as a Hugging Face Job.

Default: Qwen2.5-0.5B on a10g-small (24 GB VRAM) — no 4-bit needed.
For larger models use --4bit flag.

Usage:
    python scripts/launch_hf_job.py                                    # 0.5B, 100 ep
    python scripts/launch_hf_job.py --model Qwen/Qwen2.5-1.5B-Instruct --episodes 100
    python scripts/launch_hf_job.py --model Qwen/Qwen2.5-3B-Instruct --4bit
    python scripts/launch_hf_job.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct
"""
from __future__ import annotations
import argparse, os, sys
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
    p = argparse.ArgumentParser(description="Launch ALICE GRPO training on HF Jobs")
    p.add_argument("--model",    default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HF model ID")
    p.add_argument("--episodes", type=int, default=100,
                   help="Training episodes (default 100)")
    p.add_argument("--group",    type=int, default=8,
                   help="GRPO group size / rollouts per update (default 8)")
    p.add_argument("--turns",    type=int, default=3,
                   help="Max turns per episode (default 3)")
    p.add_argument("--lr",       type=float, default=1e-5)
    p.add_argument("--lora-r",   type=int, default=16)
    p.add_argument("--4bit",     dest="load_4bit", action="store_true",
                   help="Enable 4-bit QLoRA (needed for 7B+ on a10g)")
    p.add_argument("--flavor",   default="cpu-basic",
                   choices=["cpu-basic", "t4-small", "t4-medium", "l4x1",
                             "a10g-small", "a10g-large", "l40sx1"],
                   help="HF Jobs flavor (default cpu-basic)")
    p.add_argument("--push-to-hub", dest="push_to_hub", action="store_true",
                   help="Push trained LoRA to HF Hub after training")
    p.add_argument("--hub-repo", default="",
                   help="HF Hub repo ID for checkpoint push")
    p.add_argument("--no-wait",  action="store_true",
                   help="Submit and exit without streaming logs")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import run_uv_job, inspect_job, fetch_job_logs
    except ImportError:
        print("pip install huggingface_hub>=0.36")
        sys.exit(1)

    os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
    namespace   = HF_SPACE_ID.split("/")[0]
    script_path = str(Path(__file__).parent.parent / "training" / "hf_job_train.py")

    print("=" * 65)
    print("ALICE Training Job")
    print(f"  Model:    {args.model}")
    print(f"  Episodes: {args.episodes} | Group: {args.group} | Turns: {args.turns}")
    print(f"  LR: {args.lr} | LoRA r: {args.lora_r} | 4-bit: {args.load_4bit}")
    print(f"  Flavor:   {args.flavor}")
    print(f"  Space:    https://{HF_SPACE_ID.replace('/', '-')}.hf.space")
    print("=" * 65)

    env = {
        "HF_SPACE_ID":    HF_SPACE_ID,
        "MODEL_ID":       args.model,
        "EPISODES":       str(args.episodes),
        "GROUP_SIZE":     str(args.group),
        "MAX_TURNS":      str(args.turns),
        "LR":             str(args.lr),
        "LORA_R":         str(args.lora_r),
        "LOAD_IN_4BIT":   "1" if args.load_4bit else "0",
        "PUSH_TO_HUB":    "1" if args.push_to_hub else "0",
        "HUB_REPO_ID":    args.hub_repo,
    }

    job = run_uv_job(
        script_path,
        flavor=args.flavor,
        namespace=namespace,
        env=env,
        secrets={"HF_TOKEN": HF_TOKEN},
        token=HF_TOKEN,
    )

    print(f"\n✅ Job submitted!")
    print(f"   Job ID:  {job.id}")
    print(f"   Job URL: {job.url}")
    print(f"   Status:  {job.status.stage}")
    print(f"   Dashboard: https://{HF_SPACE_ID.replace('/', '-')}.hf.space")

    if args.no_wait:
        print(f"\nStream logs:")
        print(f"  python -c \"from huggingface_hub import fetch_job_logs; "
              f"[print(l,end='') for l in fetch_job_logs(job_id='{job.id}', "
              f"token='{HF_TOKEN}')]\"")
        return

    print("\nStreaming logs (Ctrl+C to stop watching, job continues)...\n" + "-" * 65)
    try:
        for line in fetch_job_logs(job_id=job.id, token=HF_TOKEN):
            print(line, end="", flush=True)

        final = inspect_job(job_id=job.id, token=HF_TOKEN)
        space_url = f"https://{HF_SPACE_ID.replace('/', '-')}.hf.space"
        print(f"\n{'=' * 65}")
        print(f"Job finished: {final.status.stage}")
        if final.status.stage == "COMPLETED":
            print(f"✅ Leaderboard: {space_url}/leaderboard")
        else:
            print(f"❌ Failed. Logs: {final.url}")
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nStopped watching. Job still running: {job.url}")


if __name__ == "__main__":
    main()
