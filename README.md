---
title: ALICE RL Environment
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
hardware: t4-small
secrets:
  - OPENAI_API_KEY
  - HF_TOKEN
  - REFERENCE_MODEL_PRIMARY
  - REFERENCE_MODEL_SECONDARY
---

# ALICE — Adversarial Loop for Inter-model Co-evolutionary Environment

ALICE is a closed-loop RL training environment implementing a **hunt → repair → verify → escalate** cycle for training LLM agents on Hugging Face infrastructure.

## Architecture

- **OpenEnv Server** (`server.py`): FastAPI server exposing `/reset`, `/step`, `/state`, `/health`
- **Episode Handler**: 3-turn trajectory management with CoT reasoning
- **Task Generator**: Hunt mode (adversarial prompts) + Repair mode (training pair synthesis)
- **Curriculum Manager**: Discrimination zone computation and co-evolutionary escalation
- **Oracle Interface**: Benchmark calibration via GPT-4o and Qwen-72B
- **Verifier Stack**: Three-tier verification (Programmatic → LLM Judge → Regression Battery)
- **Failure Bank**: Novelty-indexed failure storage and repair queue
- **Reward Function**: Bellman-shaped composite scoring
- **Anti-Hacking Monitors**: RestrictedPython sandbox, trajectory sampler, DAPO entropy monitor
- **Gradio Dashboard**: Live observability at port 7860

## Quick Start

```bash
# Install dependencies
uv sync

# Start the OpenEnv server
uv run uvicorn server:app --host 0.0.0.0 --port 8000

# Start the dashboard
uv run python dashboard/gradio_app.py

# Run training
uv run python training/train.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ALICE_MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | HF model ID for the agent |
| `ALICE_ENV_URL` | `http://localhost:8000` | OpenEnv server URL |
| `ALICE_HF_REPO_ID` | `` | HF Hub repo for checkpoint saving |
| `ALICE_LR` | `1e-5` | Training learning rate |
| `ALICE_GAMMA` | `0.99` | Discount factor |
| `OPENAI_API_KEY` | `` | OpenAI API key for Oracle/LLM Judge |
| `HF_TOKEN` | `` | Hugging Face token |

## Deployment

See `scripts/deploy_spaces.sh` for HF Spaces deployment and `scripts/launch_training.sh` for HF Jobs training.
