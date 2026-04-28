#!/usr/bin/env bash
# Launch ALICE training on Hugging Face Jobs (T4 GPU).
set -euo pipefail

HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set}"
MODEL_ID="${ALICE_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
ENV_URL="${ALICE_ENV_URL:-https://your-username-alice-rl-environment.hf.space}"
HF_REPO_ID="${ALICE_HF_REPO_ID:-your-username/alice-agent}"

echo "Pulling latest agent checkpoint from HF Hub..."
huggingface-cli download "$HF_REPO_ID" --local-dir ./checkpoints || true

echo "Launching training job on HF Jobs (T4, Unsloth 4-bit QLoRA)..."
huggingface-cli jobs run \
    --gpu t4-medium \
    --env "ALICE_MODEL_ID=$MODEL_ID" \
    --env "ALICE_ENV_URL=$ENV_URL" \
    --env "ALICE_HF_REPO_ID=$HF_REPO_ID" \
    --env "HF_TOKEN=$HF_TOKEN" \
    "pip install -q 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' && uv run python training/train.py"

echo "Training job submitted."
