# ALICE — Active Links

## Environment Server (local)

| Page | URL |
|---|---|
| Gradio Dashboard | http://localhost:7860 |
| API Swagger Docs | http://localhost:7860/docs |
| Health Check | http://localhost:7860/health |
| Current State | http://localhost:7860/state |
| Failure Bank | http://localhost:7860/failures |
| OpenAPI JSON | http://localhost:7860/openapi.json |

## Environment API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode, returns `episode_id` + first task |
| `POST` | `/step` | Submit an action, returns reward + next state |
| `GET`  | `/state` | Current episode state |
| `GET`  | `/health` | Uptime, error rate, latency P95, RAM |
| `GET`  | `/failures` | Failure bank (optional `?error_type=` / `?agent_version=` filters) |

## Standalone Analytics Dashboard

Run separately on port 7861:

```bash
ALICE_ENV_URL=http://localhost:7860 python dashboard/gradio_app.py
```

| Page | URL |
|---|---|
| Analytics Dashboard | http://localhost:7861 |

## Hugging Face

Configure these environment variables (Space secrets) to enable HF links:

| Env var | Purpose |
|---|---|
| `HF_SPACE_ID` | `username/env-space-name` — the environment Space |
| `ALICE_HF_REPO_ID` | `username/training-space-name` — the training Space |
| `HF_TOKEN` | Write token for private spaces and checkpoint pushes |

Once set, the URLs follow this pattern:

| Resource | URL pattern |
|---|---|
| Environment Space | `https://<username>-<space-name>.hf.space` |
| Environment Space Docs | `https://<username>-<space-name>.hf.space/docs` |
| Training Space | `https://<username>-<training-space>.hf.space` |
| HF Space Jobs | https://huggingface.co/spaces |
| HF Hub | https://huggingface.co |

## Starting Everything

### Local (single command)

```bash
cd alice_env
python alice_server.py
# → Gradio at http://localhost:7860
# → API    at http://localhost:7860/docs
```

### Local (server + standalone dashboard)

```bash
# Terminal 1
python alice_server.py

# Terminal 2
ALICE_ENV_URL=http://localhost:7860 python dashboard/gradio_app.py
# → Advanced analytics at http://localhost:7861
```

### Training (local, needs GPU)

```bash
ALICE_ENV_URL=http://localhost:7860 python training/train.py
```

### Deploy to HF Spaces

```bash
HF_SPACE_ID=username/alice-env HF_TOKEN=hf_... bash scripts/deploy_spaces.sh
```