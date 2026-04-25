# ALICE — Active Links

## Hugging Face Space (Live)

| Resource | URL |
|---|---|
| **Gradio Dashboard** | https://rohanjain1648-alice-rl-environment.hf.space |
| **API Swagger Docs** | https://rohanjain1648-alice-rl-environment.hf.space/docs |
| **Health Endpoint** | https://rohanjain1648-alice-rl-environment.hf.space/health |
| **Leaderboard API** | https://rohanjain1648-alice-rl-environment.hf.space/leaderboard |
| **Failure Bank API** | https://rohanjain1648-alice-rl-environment.hf.space/failures |
| **HF Space page** | https://huggingface.co/spaces/rohanjain1648/alice-rl-environment |
| **Space file tree** | https://huggingface.co/spaces/rohanjain1648/alice-rl-environment/tree/main |
| **HF Jobs Dashboard** | https://huggingface.co/spaces |

## Environment Server (local)

| Page | URL |
|---|---|
| Gradio Dashboard | http://localhost:7860 |
| API Swagger Docs | http://localhost:7860/docs |
| Health Check | http://localhost:7860/health |
| Current State | http://localhost:7860/state |
| Failure Bank | http://localhost:7860/failures |
| Leaderboard | http://localhost:7860/leaderboard |
| OpenAPI JSON | http://localhost:7860/openapi.json |

## Environment API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode, returns `episode_id` + first task |
| `POST` | `/step` | Submit an action, returns reward + next state |
| `GET`  | `/state` | Current episode state |
| `GET`  | `/health` | Uptime, error rate, latency P95, RAM |
| `GET`  | `/failures` | Failure bank (`?error_type=` / `?agent_version=` filters) |
| `GET`  | `/leaderboard` | Leaderboard (`?model_ids=id1,id2` filter) |
| `POST` | `/leaderboard/update` | Push training scores (used by train scripts) |
| `POST` | `/leaderboard/submit` | Register a user model for comparison |

## Standalone Analytics Dashboard (local)

```bash
ALICE_ENV_URL=http://localhost:7860 python dashboard/gradio_app.py
# → Advanced analytics at http://localhost:7861
```

## Training

### Pure TRL GRPO
```bash
ALICE_ENV_URL=http://localhost:7860 python training/train_trl.py \
    --model_id Qwen/Qwen2.5-1.5B-Instruct --episodes 200 --load_in_4bit
```

### Unsloth + TRL GRPO (2× faster, -60% VRAM)
```bash
ALICE_ENV_URL=http://localhost:7860 python training/train_unsloth.py \
    --model_id Qwen/Qwen2.5-1.5B-Instruct --episodes 200
```

### Against the live HF Space
```bash
ALICE_ENV_URL=https://rohanjain1648-alice-rl-environment.hf.space \
python training/train_trl.py --model_id Qwen/Qwen2.5-0.5B-Instruct
```

## Colab Notebooks

| Notebook | Link |
|---|---|
| TRL GRPO | notebooks/train_trl_colab.ipynb |
| Unsloth GRPO | notebooks/train_unsloth_colab.ipynb |

## Benchmark Models (Leaderboard)

| Model | HF ID |
|---|---|
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B-Instruct` |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B-Instruct` |
| Qwen2.5-3B   | `Qwen/Qwen2.5-3B-Instruct` |
| SmolLM2-1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| Gemma-3-1B   | `google/gemma-3-1b-it` |

## Space Secrets to Configure

| Secret | Value |
|---|---|
| `HF_SPACE_ID` | `rohanjain1648/alice-rl-environment` |
| `ALICE_HF_REPO_ID` | `rohanjain1648/<training-space>` |
| `HF_TOKEN` | your HF write token |
| `OPENAI_API_KEY` | your OpenAI key (for T2 LLM judge) |

## Deploy / Update

```bash
# From ALICE_final/
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path='alice_env',
    repo_id='rohanjain1648/alice-rl-environment',
    repo_type='space',
    ignore_patterns=['.venv/**','__pycache__/**','*.pyc','data/**','checkpoints/**','.pytest_cache/**'],
)
"
```