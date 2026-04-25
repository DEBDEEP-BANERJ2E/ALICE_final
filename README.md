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
  - HF_SPACE_ID
  - ALICE_HF_REPO_ID
---

# ALICE — Adversarial Loop for Inter-model Co-evolutionary Environment

[![HF Space](https://img.shields.io/badge/🤗%20Space-ALICE%20RL%20Environment-blue)](https://huggingface.co/spaces/rohanjain1648/alice-rl-environment)
[![Open in Colab (TRL)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohanjain1648/alice-rl-environment/blob/main/notebooks/train_trl_colab.ipynb)
[![Open in Colab (Unsloth)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohanjain1648/alice-rl-environment/blob/main/notebooks/train_unsloth_colab.ipynb)

ALICE is a **closed-loop, adversarial RL training environment** that implements a **hunt → repair → verify → escalate** cycle for training LLM agents using GRPO (Group Relative Policy Optimisation, as in DeepSeek-R1).

It is built on top of **OpenEnv** — the standardised RL environment framework for LLMs — and deployed as a Hugging Face Docker Space.

---

## Live Links

| Resource | URL |
|---|---|
| **Gradio Dashboard** | https://rohanjain1648-alice-rl-environment.hf.space |
| **API Swagger Docs** | https://rohanjain1648-alice-rl-environment.hf.space/docs |
| **Health Endpoint** | https://rohanjain1648-alice-rl-environment.hf.space/health |
| **Leaderboard API** | https://rohanjain1648-alice-rl-environment.hf.space/leaderboard |
| **HF Space** | https://huggingface.co/spaces/rohanjain1648/alice-rl-environment |
| **HF Jobs Dashboard** | https://huggingface.co/spaces |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    ALICE RL Environment                  │
│                                                          │
│  ┌─────────────┐     ┌──────────────────────────────┐   │
│  │  FastAPI /  │     │        Gradio UI (6 tabs)     │   │
│  │  OpenEnv    │     │  Overview | Curriculum |      │   │
│  │  API        │     │  Training | HF Jobs |         │   │
│  │  /reset     │     │  Failure Bank | Leaderboard   │   │
│  │  /step      │     └──────────────────────────────┘   │
│  │  /health    │                                         │
│  │  /state     │                                         │
│  │  /failures  │                                         │
│  │  /leaderboard│                                        │
│  └──────┬──────┘                                         │
│         │                                                │
│  ┌──────▼──────────────────────────────────────────┐    │
│  │              Core Components                     │    │
│  │                                                  │    │
│  │  EpisodeHandler ──▶ CurriculumManager            │    │
│  │       │                    │                     │    │
│  │  TaskGenerator        VerifierStack              │    │
│  │  (Hunt + Repair)    T1: RestrictedPython         │    │
│  │       │             T2: LLM Judge (CoT rubric)   │    │
│  │  RewardFunction      T3: 500+ Regression tests   │    │
│  │  (GRPO-shaped)             │                     │    │
│  │       │             FailureBank                  │    │
│  │       └──────────── (sentence-transformer + kNN) │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. OpenEnv-Compliant API
Standard `POST /reset` → `POST /step` → `GET /state` cycle, fully compatible with any OpenEnv client.

### 2. Chain-of-Thought (CoT) Prompting
All training prompts wrap tasks in `<task>...</task><reasoning>` tags, encouraging models to reason before answering. The verifier T2 judge uses a CoT rubric scoring correctness, reasoning depth, and conciseness.

### 3. Three-Tier Verifier Stack
| Tier | What it does |
|---|---|
| **T1 — RestrictedPython sandbox** | Safe execution of code answers; catches exceptions, infinite loops |
| **T2 — Dual LLM judge with CoT** | Two reference models score correctness + reasoning; composite score |
| **T3 — Regression battery** | 500+ deterministic test cases; hard pass/fail for known questions |

### 4. Curriculum Manager — Discrimination Zone
The curriculum auto-escalates difficulty based on the **discrimination zone**: tasks where the model's success rate is between 20% and 80%. Too-easy tasks are retired; too-hard tasks are deferred. This ensures training always happens at the frontier.

### 5. Failure Bank — Novelty Scoring
Failed episodes are embedded with `sentence-transformers/all-MiniLM-L6-v2` and scored against a k-NN index. Novel failures (novelty > 0.7) are queued for repair. Duplicate failures are suppressed.

### 6. GRPO Training (DeepSeek-R1 style)
Group Relative Policy Optimisation: no value model, no critic. Advantages are normalised within a group of G rollouts from the same prompt. KL penalty keeps the policy close to the reference model.

### 7. Leaderboard
5 benchmark models evaluated in the ALICE RL environment, ranked by composite RL score:
- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `google/gemma-3-1b-it`

Users can submit their own model via the Leaderboard tab or `POST /leaderboard/submit`.

---

## Training Scripts

### Pure TRL GRPO (`training/train_trl.py`)
```bash
python training/train_trl.py \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --episodes 200 \
    --load_in_4bit \
    --update_leaderboard
```

### Unsloth + TRL GRPO (`training/train_unsloth.py`)
```bash
python training/train_unsloth.py \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --episodes 200 \
    --update_leaderboard
```
Unsloth gives ~2× faster training and ~60% less VRAM vs standard transformers. Falls back to standard transformers automatically if Unsloth is not available.

---

## Colab Notebooks

| Notebook | Description |
|---|---|
| [train_trl_colab.ipynb](notebooks/train_trl_colab.ipynb) | Pure TRL GRPO — runs on free T4 Colab GPU |
| [train_unsloth_colab.ipynb](notebooks/train_unsloth_colab.ipynb) | Unsloth 4-bit QLoRA + TRL — fastest option |

---

## Local Quick Start

```bash
cd alice_env
python alice_server.py
# Gradio dashboard: http://localhost:7860
# API docs:         http://localhost:7860/docs
```

Training against local server:
```bash
ALICE_ENV_URL=http://localhost:7860 python training/train_trl.py --model_id Qwen/Qwen2.5-0.5B-Instruct
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ALICE_ENV_URL` | `http://localhost:7860` | ALICE server URL |
| `OPENAI_API_KEY` | — | OpenAI key for T2 LLM judge |
| `HF_TOKEN` | — | HF write token for checkpoint push |
| `HF_SPACE_ID` | — | `username/space-name` for HF Space link |
| `ALICE_HF_REPO_ID` | — | Training Space ID for HF Jobs tab |
| `REFERENCE_MODEL_PRIMARY` | `gpt-4o-mini` | Primary reference model for T2 judge |
| `REFERENCE_MODEL_SECONDARY` | `Qwen/Qwen2.5-72B` | Secondary reference model |
| `LEADERBOARD_PATH` | `data/leaderboard.json` | Leaderboard persistence path |
| `PORT` | `7860` | Server port |

---

## Deploy to HF Spaces

```bash
HF_SPACE_ID=rohanjain1648/alice-rl-environment \
HF_TOKEN=hf_... \
bash scripts/deploy_spaces.sh
```

The Dockerfile uses a two-stage build: builder installs all deps into `/app/.venv`, runtime copies only the virtualenv — keeping the image lean.