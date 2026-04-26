# ALICE — Project Structure

> **ALICE** (Adversarial Loop for Inter-model Co-evolutionary Environments) is a live RL training environment hosted as a Hugging Face Space. Models are evaluated through multi-turn episodes, graded by a three-tier verifier, and ranked on a public leaderboard.

---

## Root

| File / Folder | What it is |
|---|---|
| `alice_server.py` | **Main entry point.** Single file that wires together FastAPI (REST API) + Gradio (dashboard UI). Runs on `uvicorn` at port 7860. Contains all API endpoints (`/reset`, `/step`, `/health`, `/training/push`, `/leaderboard`, etc.), the dashboard's eight tabs, mock/seed data, and the HF Job launch logic. |
| `server.py` | Older OpenEnv-compliant FastAPI server (the pre-Gradio version). Exposes the same `/reset` and `/step` endpoints against `server/app.py`. No longer the primary entry point but kept for compatibility. |
| `client.py` | Meta OpenEnv reference client stub (Apache-licensed boilerplate). Not actively used by ALICE itself. |
| `models.py` | Meta OpenEnv reference model definitions (Apache-licensed boilerplate). Not actively used by ALICE itself. |
| `alice_env_leaderboard_tmp.py` | Scratch/staging copy of `environment/leaderboard.py` used to prototype the `_load()` fix before merging. Safe to delete once merged. |
| `Dockerfile` | **Primary production Dockerfile.** Two-stage build: installs all Python deps (including optional Unsloth GPU stack) in a builder image, then copies the venv into a slim runtime image. Entry point: `python alice_server.py`. |
| `pyproject.toml` | Python package manifest (`openenv-alice_env`). Declares all runtime dependencies, optional GPU extras, dev extras, and `uv` script shortcuts (`serve`, `train`, `dashboard`). |
| `uv.lock` | Pinned dependency lockfile for `uv`. Ensures reproducible installs across environments. |
| `openenv.yaml` | OpenEnv spec descriptor. Declares this repo as an OpenEnv `space` environment backed by `alice_server:app` on port 7860. |
| `.env` | Local environment variables (not committed to git). Typically holds `HF_TOKEN`, `OPENAI_API_KEY`, `ALICE_ENV_URL`. |
| `.gitattributes` | Git LFS / line-ending rules. |
| `README.md` | HF Space README (shown on the Space landing page). Contains the Space header metadata block. |
| `LINKS.md` | Curated list of relevant links (HF Space, GitHub, research papers). |
| `PROGRESS.md` | Informal dev journal tracking what has been built and what is still planned. |

---

## `environment/` — Core RL Environment Logic

The RL environment is implemented here. Every component is stateless where possible; shared state flows through `episode_handler.py`.

| File | What it does |
|---|---|
| `__init__.py` | Package marker. |
| `episode_handler.py` | Manages the 3-turn episode lifecycle: `reset()` initialises a new episode (generates a task, sets state), `step(action)` runs verification and computes the reward, marks the episode done after 3 turns or a pass. The CoT reflection loop (turn 2 uses turn 1 feedback) lives here. |
| `state.py` | Defines the MDP state vector: 768-d task embedding + 5-d agent capability vector + difficulty tier + turn number + 16×768 failure bank snapshot + discrimination coverage. Used to represent the environment's current observation. |
| `task_generator.py` | Two modes — **Hunt**: prompts an LLM adversarially to generate tasks in the discrimination zone (tasks the model gets right 20–80% of the time). **Repair**: synthesises (task, correct_solution) training pairs from Failure Bank entries for SFT. |
| `verifier_stack.py` | Three-tier verification pipeline: **T1** RestrictedPython sandbox (no `open`/`exec`/`import`, 5 s timeout, 512 MB cap), **T2** LLM judge (checks correctness of intent), **T3** regression battery (20 task variants). Composite score = 0.4×T1 + 0.4×T2 + 0.2×T3. |
| `reward_function.py` | Bellman-shaped reward. Turn 1: R = programmatic pass/fail. Turn 2: R += turn-decay × composite. Turn 3: R += novelty bonus. Also applies a penalty for known-failure patterns. |
| `oracle_interface.py` | Calls the HF Inference API with reference models to calibrate task difficulty. Results are cached (30-day TTL) to avoid redundant API calls. Used to compute discrimination scores. |
| `curriculum_manager.py` | Tracks per-task success rates, computes the discrimination zone (20–80% success), and escalates the difficulty tier when coverage > 70% for 50+ episodes. The co-evolutionary loop controller. |
| `failure_bank.py` | Stores confirmed failure cases (composite < 0.5) with sentence-transformer embeddings. Uses k-NN cosine similarity (k=16) to deduplicate by novelty — only failures with novelty > 0.7 are added. Manages the repair queue for SFT. |
| `leaderboard.py` | In-memory leaderboard backed by `data/leaderboard.json`. Stores per-model `avg_reward`, `success_rate`, `disc_coverage`, `episodes_run`. Computes `rl_score = 0.5×avg_reward + 0.3×success_rate + 0.2×disc_coverage`. Seeded with real benchmark results on first run. |

---

## `training/` — Training Scripts

All scripts are self-contained and can be submitted as HF Jobs directly.

| File | What it does |
|---|---|
| `__init__.py` | Package marker. |
| `hf_cpu_job.py` | **Free-tier HF Job script.** Uses the HF Inference API for model inference (no local model loading, no GPU). Runs multi-turn ALICE episodes, pushes per-episode metrics to `/training/push` (live dashboard updates), and registers job status. Submitted on `cpu-basic` flavor — no HF credits needed. This is what the "Start Training" button and the Leaderboard submit form use. |
| `hf_job_train.py` | **GPU training script.** Loads the model locally with LoRA (via `peft`), runs full GRPO policy gradient updates, cosine LR schedule, gradient clipping. Designed for `a10g-small` (24 GB VRAM, no 4-bit needed for ≤3B models). Pushes metrics to `/training/push` every episode. Optionally pushes the LoRA checkpoint to HF Hub on completion. |
| `train.py` | TRL + Unsloth GRPO trainer. Uses `GRPOTrainer` from TRL directly against the ALICE env API. Earlier version of the training loop. |
| `train_trl.py` | Pure TRL GRPO script (no Unsloth dependency). Drop-in for environments where Unsloth can't be installed. |
| `train_unsloth.py` | Unsloth + TRL GRPO script. Fastest option on compatible NVIDIA GPUs (uses Unsloth's custom CUDA kernels for 2× speed). |

---

## `monitors/` — Runtime Safety Monitors

These run alongside training to detect pathological behaviour.

| File | What it does |
|---|---|
| `__init__.py` | Package marker. |
| `entropy_monitor.py` | **DAPO Entropy Monitor.** Tracks policy entropy over a 100-episode sliding window. Flags `potential_collapse` if entropy drops > 20% — a sign the model is converging to a single degenerate output (e.g. always `result = 42`). |
| `sandbox.py` | Programmatic verifier sandbox (also used standalone). Executes agent-generated Python in RestrictedPython with `open`, `exec`, `eval`, `__import__`, `compile` blocked. Captures stdout/stderr and enforces a 5 s / 512 MB limit. |
| `trajectory_sampler.py` | Randomly samples 5% of trajectories for anomaly detection. Detects four failure modes: `reward_hacking`, `exploration_collapse`, `output_repetition`, `policy_divergence`. |

---

## `server/` — Legacy OpenEnv Server

An earlier standalone FastAPI server from the OpenEnv scaffolding phase, kept for reference and compatibility.

| File | What it does |
|---|---|
| `__init__.py` | Package marker. |
| `app.py` | FastAPI app (Meta OpenEnv boilerplate). Exposes `/reset`, `/step`, `/health` backed by `alice_env_environment.py`. |
| `alice_env_environment.py` | The environment class used by the legacy server. Earlier version of the episode logic before `environment/episode_handler.py`. |
| `Dockerfile` | Multi-stage Dockerfile for the legacy server. Uses `openenv-base` as the base image and `uvicorn server.app:app` as the entry point. |
| `requirements.txt` | Dependency list for the legacy server (superseded by `pyproject.toml`). |

---

## `dashboard/` — Standalone Analytics Dashboard

| File | What it does |
|---|---|
| `__init__.py` | Package marker. |
| `gradio_app.py` | Standalone Gradio analytics dashboard that connects to a running ALICE server over HTTP. Shows 5 tabs of advanced analytics (reward curves, curriculum heatmap, failure bank, trajectory viewer, leaderboard). Not the primary dashboard — `alice_server.py` embeds the dashboard directly. |

---

## `scripts/` — Deployment & Launch Helpers

| File | What it does |
|---|---|
| `launch_hf_job.py` | CLI wrapper to submit `hf_job_train.py` as a HF Job. Accepts `--model`, `--episodes`, `--4bit`, `--flavor` flags. Default: Qwen2.5-0.5B on `a10g-small`. |
| `launch_training.sh` | Shell script that sets env vars and calls `launch_hf_job.py`. Convenience wrapper for local dev. |
| `deploy_spaces.sh` | Pushes the repo to the HF Space remote (`git push hf hf-deploy:main`). |
| `verify_spaces.sh` | Health-checks the deployed Space by hitting `/health` and asserting a 200 response. |

---

## `notebooks/` — Colab Notebooks

| File | What it does |
|---|---|
| `alice_full_colab.ipynb` | End-to-end ALICE notebook: installs deps, starts the env server, runs training with GRPO, plots results. Designed to run on a free Colab T4. |
| `train_trl_colab.ipynb` | Minimal TRL GRPO notebook focused on the training loop only (assumes the ALICE Space is already running). |
| `train_unsloth_colab.ipynb` | Unsloth-accelerated GRPO notebook for Colab Pro (A100). Fastest training path. |

---

## `tests/` — Test Suite

| File | What it does |
|---|---|
| `__init__.py` | Package marker. |
| `test_server.py` | Unit tests for the API endpoints (`/reset`, `/step`, `/health`). Uses `httpx` against a live server. |
| `test_integration.py` | End-to-end integration tests: full episode lifecycle, reward shaping, verifier stack composition. |
| `test_properties.py` | Property-based tests (hypothesis or manual) for reward function invariants and curriculum logic. |

---

## `images/` — Static Assets

| File | What it is |
|---|---|
| `alice_architecture.png` | Architecture diagram of the ALICE co-evolutionary loop. Shown in the Mission tab of the dashboard. |
| `hf_blog_1.jpeg` | Screenshot from the HF Open LLM Leaderboard v2 blog post (intro & motivation). |
| `hf_blog_2.jpeg` | Screenshot showing benchmark saturation — top scores converging on human baseline. |
| `hf_blog_3.jpeg` | Screenshot showing saturation causes: easy tasks, data contamination, benchmark errors. |

---

## `data/` — Persistent Storage

| File | What it is |
|---|---|
| `.gitkeep` | Keeps the empty `data/` directory tracked by git. |
| `leaderboard.json` | Auto-generated at runtime. Stores all leaderboard entries as JSON. Created on first startup if missing; seeded from benchmark results. **Not committed to git.** |

---

## Key Data Flows

```
User / Training Script
        │
        ▼
  POST /reset  ──►  episode_handler.py  ──►  task_generator.py
                           │
        ◄──── {episode_id, task}
        │
        ▼
  POST /step   ──►  verifier_stack.py (T1→T2→T3)
                    reward_function.py
                    curriculum_manager.py
                    failure_bank.py
                           │
        ◄──── {reward, done, info}
        │
        ▼
  POST /training/push  ──►  dashboard charts (live)
                             leaderboard.py  (score update)
```
