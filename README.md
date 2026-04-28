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
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-orange)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

> 📖 BLOG: **[Read the full technical deep-dive →](BLOG.md)**

---

**ALICE is a closed-loop, adversarial RL training environment for LLMs.** It implements a continuous **hunt → verify → repair → escalate** cycle: the environment generates tasks that sit just at the frontier of the model's capability, grades responses through a three-tier verifier, banks novel failures, and escalates difficulty as the model improves — all without any human labels or static datasets.

Built on top of **[OpenEnv](https://github.com/meta-pytorch/OpenEnv)** and deployed as a Hugging Face Docker Space.

---

## Why ALICE?

Static benchmarks saturate. Once a model solves most of a fixed test set, scores cluster near the ceiling and stop measuring real capability differences.

![HF Open LLM Leaderboard v2 — intro and motivation](https://raw.githubusercontent.com/DEBDEEP-BANERJ2E/ALICE_final/main/images/hf_blog_1.jpeg)

![Benchmark saturation: top scores converging on human baseline](https://raw.githubusercontent.com/DEBDEEP-BANERJ2E/ALICE_final/main/images/hf_blog_2.jpeg)

![Saturation causes: easy tasks, contamination, benchmark errors](https://raw.githubusercontent.com/DEBDEEP-BANERJ2E/ALICE_final/main/images/hf_blog_3.jpeg)

ALICE sidesteps saturation by generating tasks **adversarially** — always targeting the model's discrimination zone (tasks it gets right 20–80% of the time). There is no ceiling because the environment co-evolves with the model.

| Traditional Benchmark | ALICE |
|---|---|
| Fixed task set | Auto-generated, model-specific tasks |
| Human labelling | Oracle discrimination + verifier stack |
| Saturates quickly | Co-evolves with the model — no ceiling |
| Pass/fail on test split | GRPO gradient signal every episode |
| One-shot scoring | Continuous online training loop |

---

## Architecture

![ALICE co-evolutionary architecture](https://raw.githubusercontent.com/DEBDEEP-BANERJ2E/ALICE_final/main/images/alice_architecture.png)

> Note: Model names shown in the architecture diagram (Qwen-72B, GPT-4o) are illustrative examples. Actual reference models used are configurable via environment variables.

```
┌──────────────────────────────────────────────────────────────────┐
│                       ALICE RL Environment                       │
│                                                                  │
│  ┌─────────────────────┐     ┌──────────────────────────────┐   │
│  │  FastAPI / OpenEnv  │     │     Gradio Dashboard (8 tabs) │   │
│  │  POST /reset        │     │  Mission | Overview |          │   │
│  │  POST /step         │     │  Training Metrics | Curriculum │   │
│  │  GET  /state        │     │  HF Jobs | Failure Bank |      │   │
│  │  GET  /health       │     │  Leaderboard | Pro Access      │   │
│  │  POST /training/push│     └──────────────────────────────┘   │
│  │  GET  /leaderboard  │                                         │
│  │  POST /jobs/register│                                         │
│  └──────────┬──────────┘                                         │
│             │                                                    │
│  ┌──────────▼──────────────────────────────────────────────┐    │
│  │                     Core Components                      │    │
│  │                                                          │    │
│  │  EpisodeHandler ──────▶ CurriculumManager                │    │
│  │        │                      │                          │    │
│  │        ▼                      ▼                          │    │
│  │  TaskGenerator          VerifierStack                    │    │
│  │  (Hunt + Repair)      T1: RestrictedPython               │    │
│  │        │              T2: LLM Judge (CoT rubric)         │    │
│  │        ▼              T3: Regression battery             │    │
│  │  RewardFunction               │                          │    │
│  │  (GRPO-shaped)        FailureBank                        │    │
│  │        │              (sentence-transformer + k-NN)      │    │
│  │        └──────────────────────┘                          │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Live Links

| Resource | URL |
|---|---|
| **Gradio Dashboard** | https://rohanjain1648-alice-rl-environment.hf.space |
| **API Swagger Docs** | https://rohanjain1648-alice-rl-environment.hf.space/docs |
| **Health Endpoint** | https://rohanjain1648-alice-rl-environment.hf.space/health |
| **Leaderboard API** | https://rohanjain1648-alice-rl-environment.hf.space/leaderboard |
| **HF Space** | https://huggingface.co/spaces/rohanjain1648/alice-rl-environment |
| **HF Jobs Dashboard** | https://huggingface.co/settings/jobs |
| **Blog — How ALICE Works** | [BLOG.md](BLOG.md) |

---

## Quick Start

### Remote — Async (Python)

```python
import asyncio
import httpx

ALICE_URL = "https://rohanjain1648-alice-rl-environment.hf.space"

async def main():
    async with httpx.AsyncClient(base_url=ALICE_URL, timeout=30) as client:
        # Start an episode
        ep = (await client.post("/reset")).json()
        episode_id = ep["episode_id"]
        task       = ep["task"]
        print(f"Task: {task}")

        # Submit an answer (turn 1 of 3)
        result = (await client.post("/step", json={
            "episode_id": episode_id,
            "action": "result = 42"
        })).json()

        print(f"Reward: {result['reward']:.4f}")
        print(f"Done:   {result['done']}")
        print(f"Score:  {result['info']['verification']['composite_score']:.4f}")

asyncio.run(main())
```

### Remote — Sync

```python
import httpx

ALICE_URL = "https://rohanjain1648-alice-rl-environment.hf.space"

with httpx.Client(base_url=ALICE_URL, timeout=30) as client:
    ep = client.post("/reset").json()
    result = client.post("/step", json={
        "episode_id": ep["episode_id"],
        "action": f"result = 'Canberra'"
    }).json()
    print(result["reward"], result["done"])
```

### Local

```bash
# Clone and install
git clone https://github.com/DEBDEEP-BANERJ2E/ALICE_final.git
cd ALICE_final
pip install -e ".[dev]"

# Run the server + dashboard
python alice_server.py
# Gradio dashboard: http://localhost:7860
# API docs:         http://localhost:7860/docs
```

```python
import httpx

with httpx.Client(base_url="http://localhost:7860", timeout=30) as env:
    ep     = env.post("/reset").json()
    result = env.post("/step", json={
        "episode_id": ep["episode_id"],
        "action": "result = sorted([3,1,4,1,5], reverse=True)"
    }).json()
    print(result)
```

---

## Episode Structure

Every ALICE episode is a **3-turn interaction**:

```
POST /reset
  └─▶ { episode_id, task, timestamp, agent_version }

Turn 1:  POST /step  { episode_id, action }
  └─▶ { reward, done=False, info: { verification, feedback } }

Turn 2:  POST /step  { episode_id, action }   ← uses T1 feedback
  └─▶ { reward, done=False, info: { verification, feedback } }

Turn 3:  POST /step  { episode_id, action }   ← final attempt
  └─▶ { reward, done=True,  info: { verification } }
```

Each `action` must be a single Python assignment: `result = <value>`.  
Rewards are shaped: turn 1 is programmatic pass/fail, turn 2 adds a turn-decay multiplier, turn 3 adds a novelty bonus. The total episode reward is the sum across all turns.

---

## API Reference

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server health, uptime, memory, error rate |
| `GET` | `/state` | Full environment state snapshot |
| `POST` | `/reset` | Start a new episode; returns task |
| `POST` | `/step` | Submit an action; returns reward + feedback |
| `GET` | `/leaderboard` | All leaderboard entries sorted by rl_score |
| `POST` | `/leaderboard/submit` | Register a model for evaluation |
| `POST` | `/leaderboard/update` | Push final scores for a model |
| `POST` | `/training/push` | Push per-episode metrics from a training job |
| `POST` | `/jobs/register` | Register / update a live HF Job in the dashboard |
| `GET` | `/jobs` | List all registered training jobs |
| `GET` | `/failures` | Query the Failure Bank |
| `GET` | `/docs` | Swagger UI |

### Request / Response Shapes

**`POST /reset`**
```json
// Response
{
  "episode_id": "abc123...",
  "task": "Write Python to check if a string is a palindrome.",
  "timestamp": "2026-04-28T12:00:00Z",
  "agent_version": "0.0.0"
}
```

**`POST /step`**
```json
// Request
{ "episode_id": "abc123...", "action": "result = s == s[::-1]" }

// Response
{
  "reward": 0.85,
  "done": false,
  "info": {
    "turn": 1,
    "verification": {
      "composite_score": 0.82,
      "tier1_score": 1.0,
      "tier2_score": 0.75,
      "tier3_score": 0.6,
      "tier1_details": { "success": true, "stdout": "", "error_message": null }
    },
    "feedback": "Score 0.82 (T1=1.00 T2=0.75). Improve correctness."
  }
}
```

**`POST /training/push`**
```json
{
  "model_id":           "Qwen/Qwen2.5-0.5B-Instruct",
  "episode":            12,
  "rewards":            [0.82, 0.91, 0.45, 0.78],
  "advantages":         [0.21, 0.43, -0.55, 0.17],
  "loss":               0.0034,
  "success_rate":       0.74,
  "disc_coverage":      0.68,
  "composites":         [0.80, 0.88, 0.42, 0.76],
  "cumulative_rewards": [0.74, 0.76, 0.77, 0.79]
}
```

---

## Verifier Stack

Every action is graded by a three-tier pipeline. The composite score weights all three tiers:

```
composite = 0.4 × T1 + 0.4 × T2 + 0.2 × T3
```

### T1 — RestrictedPython Sandbox
Executes the agent's Python in an isolated sandbox with:
- `open`, `exec`, `eval`, `__import__`, `compile` — all blocked
- 5-second CPU timeout
- 512 MB memory cap
- stdout / stderr captured

Returns `tier1_score ∈ {0.0, 1.0}`.

### T2 — Dual LLM Judge (CoT Rubric)
Two reference models (`REFERENCE_MODEL_PRIMARY`, `REFERENCE_MODEL_SECONDARY`) independently score the response using a chain-of-thought rubric measuring:
- **Correctness** — is the answer factually right?
- **Reasoning depth** — does the solution reflect understanding?
- **Conciseness** — is it direct, without unnecessary output?

Returns `tier2_score ∈ [0.0, 1.0]` as the average of both judges.

### T3 — Regression Battery
Runs the answer against 20 deterministic variants of the task to measure generalisation. Hard pass/fail on known ground-truth answers.

Returns `tier3_score ∈ [0.0, 1.0]`.

---

## Reward Function

```
R(turn=1) = tier1_score                              # programmatic pass/fail
R(turn=2) = γ × composite_score                     # turn-decay (γ < 1)
R(turn=3) = γ² × composite_score + novelty_bonus    # final attempt + novelty
```

**Novelty bonus** — failed episodes are embedded with `all-MiniLM-L6-v2` and compared to existing Failure Bank entries via k-NN cosine similarity. High-novelty failures (distance > 0.7) earn a bonus and are queued for repair.

**GRPO advantages** — at the end of each group of G rollouts, advantages are group-normalised:

```
advantage_i = (reward_i − mean(rewards)) / std(rewards)
```

If all rollouts in the group achieve the same reward, `std = 0`, all advantages = 0, and no gradient flows. The curriculum intentionally keeps the model in the 20–80% success zone to prevent this.

---

## Curriculum Manager — Discrimination Zone

The curriculum tracks per-task success rates and identifies the **discrimination zone**: tasks the model solves 20–80% of the time. This is the zone of maximum variance in the Bernoulli distribution (`σ² = p(1−p)` is maximised at p=0.5), which gives the strongest gradient signal.

```
p < 20%  → too hard  (model guesses randomly, no gradient)
p ∈ 20–80% → discrimination zone  ← train here
p > 80%  → too easy  (saturated, no gradient)
```

**Escalation rule:** when discrimination zone coverage exceeds 70% for 50+ consecutive episodes, the difficulty tier increments and harder tasks are introduced. The environment permanently co-evolves with the model.

---

## Failure Bank

Every episode where `composite_score < 0.5` produces a **failure entry**. Before storing it, the Failure Bank:

1. Embeds the task + action using `sentence-transformers/all-MiniLM-L6-v2`
2. Queries the k-NN index (k=16) for cosine similarity
3. Computes `novelty = 1 − max_similarity`
4. **Only stores the failure** if `novelty > 0.7` (genuinely new failure)
5. Adds it to the repair queue for targeted SFT

This prevents the curriculum from repeatedly testing known failure modes and keeps the bank semantically diverse.

---

## GRPO Training

ALICE uses **Group Relative Policy Optimisation** (DeepSeek-R1 style) — no value network, no critic.

```
Loss = −mean( advantage_i × log_prob(action_i) ) + KL_coef × log_prob²
```

The KL term (squared log-prob) penalises large policy updates without needing a separate reference model. `KL_coef` defaults to `0.04`.

**LoRA** — training scripts use Low-Rank Adaptation (`rank=16`, `alpha=32`) targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Reduces trainable parameters by ~99%.

---

## Reward Hacking: Risks and Mitigations

Reward hacking is the core failure mode of RL training: the model learns to maximise the reward signal without achieving the intended capability. The simple rule is **the model will give you exactly what you asked for, which may not be what you wanted**.

### Known attack surfaces

**Verifier gaming** — A model could learn that the T1 sandbox passes if it returns the correct type with the right surface form, without solving the underlying problem. For example, returning a hardcoded string that matches frequent task patterns.

**Timing exploitation (cobra effect)** — On timing or performance benchmarks, a model could learn to manipulate timer variables, sleep calls, or resource limits within its generated code rather than actually optimising. It optimises the measurement, not the capability.

**Environment corruption** — In the worst case, a model could attempt to overwrite test files, alter the test harness, or inject code that modifies evaluation logic. This doesn't improve the model's learning but breaks the safety and integrity of the environment.

**Metric gaming** — The model could learn to score well on T1 (exact execution pass) while systematically failing T2 and T3, indicating it has learned surface-level syntactic patterns rather than semantic understanding.

### ALICE's mitigations

| Layer | Mechanism | What it prevents |
|---|---|---|
| **Sandbox (T1)** | RestrictedPython blocks `open`, `exec`, `eval`, `__import__`, `compile`; 5 s CPU cap; 512 MB memory cap | File system corruption, timer manipulation, environment modification |
| **Dual judge (T2)** | Two independent reference models score independently; the training model never judges itself | Self-referential gaming; a policy that fools one judge rarely fools both |
| **Regression battery (T3)** | 20 deterministic task variants with known ground-truth answers | Surface-form matching — hard to game 20 independent variants simultaneously |
| **Novelty filter** | Failure Bank only stores entries with novelty > 0.7 (k-NN cosine distance) | Repetitive identical outputs don't accumulate gradient signal |
| **Entropy monitor** | `monitors/entropy_monitor.py` — DAPO-style collapse detection | Policy entropy collapse (mode seeking to a single exploiting output) |
| **Trajectory sampler** | `monitors/trajectory_sampler.py` — 5% random sample flagged for inspection | Anomalous reward spikes or suspicious output patterns |
| **Curriculum pressure** | Curriculum escalates difficulty when disc_coverage > 70% for 50+ episodes | Easy gaming strategies get pushed out of the discrimination zone automatically |

### Detecting reward hacking in practice

The primary tool is **trajectory inspection** — examining the actual sequence of (task, action, reward, feedback) tuples rather than just aggregate metrics. Signals that warrant investigation:

- A sudden reward spike not accompanied by a rise in disc_coverage
- T1 score consistently high while T2/T3 scores remain low
- The GRPO loss dropping to near-zero while avg_reward stays flat
- Outputs that are syntactically valid but semantically identical across diverse tasks

The trajectory sampler logs 5% of all episodes to a dedicated table visible in the dashboard, enabling a smell test: *does it make sense what the model is doing here? Is the task a real task a capable model would solve this way?*

A high-quality environment is one where a post-training RL colleague would be willing to use it in a real training run — because it represents real-world tasks, has a reward signal that can't be trivially gamed, and produces trajectories whose success meaningfully maps to deployed capability.

---

## Training Scripts

### 1. Free-Tier HF Job (cpu-basic) — `training/hf_cpu_job.py`

Uses the HF Inference API for inference — **no GPU, no credits needed**. Launch directly from the dashboard (Training Metrics → Start Training) or submit manually:

```python
from huggingface_hub import run_uv_job

job = run_uv_job(
    "training/hf_cpu_job.py",
    flavor="cpu-basic",
    namespace="your-hf-username",
    env={
        "HF_SPACE_ID": "rohanjain1648/alice-rl-environment",
        "MODEL_ID":    "Qwen/Qwen2.5-0.5B-Instruct",
        "EPISODES":    "50",
        "MAX_TURNS":   "3",
    },
    secrets={"HF_TOKEN": "hf_..."},
    token="hf_...",
)
print(f"Job URL: {job.url}")
```

Metrics push to `/training/push` in real time — visible in the dashboard immediately.

### 2. GPU Training Job — `training/hf_job_train.py`

Full GRPO with local LoRA fine-tuning. Requires an `a10g-small` (24 GB VRAM) HF Job.

```bash
ALICE_ENV_URL=https://rohanjain1648-alice-rl-environment.hf.space \
python training/hf_job_train.py
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Model to train |
| `EPISODES` | `100` | Number of episodes |
| `GROUP_SIZE` | `8` | GRPO group size (rollouts per update) |
| `MAX_TURNS` | `3` | Turns per episode |
| `LR` | `1e-5` | Learning rate |
| `KL_COEF` | `0.04` | KL penalty coefficient |
| `LORA_R` | `16` | LoRA rank |
| `LOAD_IN_4BIT` | `0` | 4-bit quantisation (for <24 GB VRAM) |
| `PUSH_TO_HUB` | `0` | Push LoRA checkpoint to HF Hub after training |
| `HUB_REPO_ID` | — | Hub repo to push checkpoint to |

### 3. Pure TRL GRPO — `training/train_trl.py`

```bash
python training/train_trl.py \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --episodes 200 \
    --load_in_4bit \
    --update_leaderboard
```

### 4. Unsloth + TRL — `training/train_unsloth.py`

~2× faster, ~60% less VRAM than standard transformers. Falls back automatically if Unsloth is not installed.

```bash
python training/train_unsloth.py \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --episodes 200 \
    --update_leaderboard
```

---

## Experiment Tracking

ALICE uses a **built-in, self-hosted tracking system** — no wandb, MLflow, or external service required.

### How it works

Every training script calls two internal endpoints after each episode:

| Endpoint | When | What is logged |
|---|---|---|
| `POST /training/push` | Every episode | `rewards`, `advantages`, `loss`, `success_rate`, `disc_coverage`, `cumulative_rewards` |
| `POST /leaderboard/update` | Every 10 episodes + run end | `avg_reward`, `success_rate`, `disc_coverage`, `episodes_run` |

Metrics are held in memory and flushed to `data/leaderboard.json` for persistence across restarts.

### Where to see the results

Open the **Training Metrics** tab on the live dashboard:

```
https://rohanjain1648-alice-rl-environment.hf.space
```

It shows four auto-refreshing panels (updated every 3 s):

- **Reward curve** — per-episode mean reward over time
- **Cumulative reward** — running total reward
- **Reward distribution** — histogram of episode rewards
- **GRPO loss** — policy gradient loss per update

The **Leaderboard** tab shows all models ranked by composite RL score with per-model episode counts, avg reward, success rate, and discrimination coverage.

### Tracked metrics glossary

| Metric | Description |
|---|---|
| `avg_reward` | Mean episode reward over the last 80 episodes |
| `success_rate` | Fraction of episodes with reward > 0.3 |
| `disc_coverage` | Fraction of tasks in the 20–80% success zone |
| `loss` | GRPO policy gradient loss (lower = more stable policy) |
| `advantages` | Group-normalised GRPO advantages for the episode batch |
| `cumulative_rewards` | Running mean reward — the primary learning curve |

---

## Colab Notebooks

| Notebook | Hardware | Description |
|---|---|---|
| [alice_full_colab.ipynb](notebooks/alice_full_colab.ipynb) | Free T4 | Full end-to-end: install deps, run server, train, plot results |
| [train_trl_colab.ipynb](notebooks/train_trl_colab.ipynb) | Free T4 | Pure TRL GRPO against the live HF Space |
| [train_unsloth_colab.ipynb](notebooks/train_unsloth_colab.ipynb) | Colab Pro A100 | Unsloth 4-bit QLoRA — fastest training path |

---

## Dashboard Tabs

The Gradio dashboard at port 7860 has 8 tabs, all auto-refreshing every 3 seconds:

| Tab | What it shows |
|---|---|
| **Mission** | Problem statement, ALICE architecture, research papers, presentation slides, how-to guide |
| **Overview** | Live episode count, avg reward, success rate, discrimination coverage, health metrics, recent episodes |
| **Training Metrics** | 4-panel live chart (reward curve + cumulative + distribution + GRPO loss), episode table, "Start Training" button |
| **Curriculum** | Domain × difficulty heatmap, discrimination zone coverage over time, curriculum info |
| **HF Space & Jobs** | Space status badge, training job history table with live status |
| **Failure Bank** | Total failures, avg novelty, repair queue size, filterable failure table |
| **Leaderboard** | Bar chart + ranked table of all models, model submission form |
| **Pro Access** | Pricing tiers, contact form |

---

## Leaderboard

5 benchmark models evaluated on ALICE, ranked by composite RL score:

```
rl_score = 0.5 × avg_reward + 0.3 × success_rate + 0.2 × disc_coverage
```

| Rank | Model | Params | rl_score | avg_reward | success_rate | disc_coverage | Episodes |
|---|---|---|---|---|---|---|---|
| 1 | Qwen2.5-0.5B | 0.5B | 1.1189 | 1.6549 | 0.7375 | 0.7362 | 150 |
| 2 | Qwen2.5-3B | 3.0B | 0.8992 | 1.4278 | 0.5800 | 0.2550 | 50 |
| 3 | Qwen2.5-1.5B | 1.5B | 0.5251 | 0.8515 | 0.1775 | 0.1450 | 50 |
| 4 | SmolLM2-1.7B | 1.7B | 0.3015 | 0.5279 | 0.0825 | 0.0625 | 50 |
| 5 | Gemma-3-1B | 1.0B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 50 |

### Submit Your Model

**Via the Dashboard** — open the Leaderboard tab, enter your model ID, and click Submit & Eval. A free-tier HF Job is submitted automatically; results appear in the table when the job completes.

**Via API:**
```python
import httpx

resp = httpx.post(
    "https://rohanjain1648-alice-rl-environment.hf.space/leaderboard/submit",
    json={
        "model_id":    "your-org/your-model",
        "display_name": "My Model",
        "params_b":    0.5,
        "episodes":    50,
    }
).json()
print(resp)
# { "status": "submitted", "job_id": "...", "job_url": "https://huggingface.co/jobs/..." }
```

---

## Evidence of Reward Improvement

The leaderboard results from 5 benchmark runs provide three concrete signals that ALICE's training loop produces real capability improvement — not noise or reward hacking.

### Signal 1 — Absolute score growth over episodes

Qwen2.5-0.5B is the only model evaluated at 150 episodes (vs 50 for the others). Its final metrics:

| Metric | Qwen2.5-0.5B (150 ep) | Qwen2.5-3B (50 ep) |
|---|---|---|
| rl_score | **1.1189** | 0.8992 |
| avg_reward | **1.6549** | 1.4278 |
| success_rate | **73.75%** | 58.00% |
| disc_coverage | **73.62%** | 25.50% |

A 0.5B model outperforms a 3B model — the capability gap is explained by training time, not parameter count. This is the expected behaviour of a functioning RL loop: more episodes → better policy regardless of base model size.

### Signal 2 — Discrimination coverage as a learning signal

`disc_coverage` measures the fraction of tasks in the 20–80% success zone. Qwen2.5-0.5B reaches 73.6% coverage — meaning the curriculum has successfully escalated difficulty to stay at the model's frontier. A model that was gaming the reward (always winning easily) would show low disc_coverage as the curriculum would push it to harder tasks it can't solve. High coverage at high reward confirms the model is genuinely operating at its capability boundary.

### Signal 3 — Multi-turn correction behaviour

ALICE episodes are 3 turns. A model that improves across turns (turn 1 → turn 2 → turn 3 reward trajectory trending up) demonstrates in-context self-correction, which is a direct capability signal. This is visible per-episode in the Training Metrics dashboard under the episode detail table.

### Live reward curves

All reward curves, loss trajectories, cumulative reward, and advantage distributions are visible in real time on the **Training Metrics** tab at:

```
https://rohanjain1648-alice-rl-environment.hf.space
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HF token. Required for submitting HF Jobs and pushing checkpoints. |
| `HF_SPACE_ID` | `rohanjain1648/alice-rl-environment` | Space ID used to construct job URLs and API base URL. |
| `ALICE_HF_REPO_ID` | — | Training Space ID shown in the HF Jobs tab. |
| `REFERENCE_MODEL_PRIMARY` | `gpt-4o-mini` | Primary LLM for the T2 judge. |
| `REFERENCE_MODEL_SECONDARY` | `Qwen/Qwen2.5-72B` | Secondary LLM for the T2 judge. |
| `OPENAI_API_KEY` | — | Required if using OpenAI models as T2 judges. |
| `LEADERBOARD_PATH` | `data/leaderboard.json` | Path for leaderboard persistence. |
| `PORT` | `7860` | Server port. |
| `ALICE_ENV_URL` | `http://localhost:7860` | Used by training scripts to locate the environment server. |

---

## Deployment

### Hugging Face Spaces (Docker)

The Space runs the `Dockerfile` in the repo root. Push to the `main` branch of your HF Space to trigger a rebuild:

```bash
# Add the HF Space as a remote (one-time)
git remote add hf https://<username>:<HF_TOKEN>@huggingface.co/spaces/<username>/alice-rl-environment

# Push
git push hf your-branch:main
```

Or use the deploy script:

```bash
HF_SPACE_ID=rohanjain1648/alice-rl-environment \
HF_TOKEN=hf_... \
bash scripts/deploy_spaces.sh
```

The Dockerfile uses a two-stage build: a `builder` stage installs all Python deps (including the optional Unsloth GPU stack) into `/app/.venv`, and a lean `runtime` stage copies only the venv — keeping the image small. On GPU-capable hardware, Unsloth installs automatically; on CPU-only builds, it silently falls back to standard transformers.

### Local Docker

```bash
docker build -t alice-env:latest .
docker run --rm -p 7860:7860 \
  -e HF_TOKEN=hf_... \
  -e OPENAI_API_KEY=sk_... \
  alice-env:latest
```

### Without Docker

```bash
pip install -e ".[dev]"
uvicorn alice_server:app --host 0.0.0.0 --port 7860
```

---

## Testing

```bash
# All tests
pytest tests/ -v

# API endpoint tests
pytest tests/test_server.py -v

# End-to-end episode lifecycle
pytest tests/test_integration.py -v

# Reward function + curriculum invariants
pytest tests/test_properties.py -v
```

Health check:
```bash
curl https://rohanjain1648-alice-rl-environment.hf.space/health
# { "uptime": 3600, "error_rate": 0.0, "latency_p95": 0.21, "memory_usage": 142.3 }
```

---

## Project Structure

```
ALICE_final/
├── alice_server.py          # Main entry point — FastAPI + Gradio in one file
├── Dockerfile               # Two-stage production build
├── pyproject.toml           # Package manifest and dependencies
├── openenv.yaml             # OpenEnv spec descriptor
│
├── environment/             # Core RL environment logic
│   ├── episode_handler.py   # 3-turn episode lifecycle
│   ├── verifier_stack.py    # T1 / T2 / T3 verification pipeline
│   ├── reward_function.py   # Bellman-shaped GRPO reward
│   ├── curriculum_manager.py# Discrimination zone + difficulty escalation
│   ├── task_generator.py    # Hunt (adversarial) + Repair (SFT pairs)
│   ├── failure_bank.py      # k-NN novelty scoring + repair queue
│   ├── oracle_interface.py  # Reference model calibration (HF Inference API)
│   ├── leaderboard.py       # In-memory leaderboard + JSON persistence
│   └── state.py             # MDP state vector definition
│
├── training/
│   ├── hf_cpu_job.py        # Free-tier HF Job (cpu-basic, Inference API)
│   ├── hf_job_train.py      # GPU training job (LoRA + GRPO)
│   ├── train_trl.py         # Pure TRL GRPO trainer
│   └── train_unsloth.py     # Unsloth + TRL (fastest on GPU)
│
├── monitors/
│   ├── entropy_monitor.py   # DAPO collapse detection
│   ├── sandbox.py           # RestrictedPython execution sandbox
│   └── trajectory_sampler.py# Anomaly detection (5% sample)
│
├── notebooks/
│   ├── alice_full_colab.ipynb
│   ├── train_trl_colab.ipynb
│   └── train_unsloth_colab.ipynb
│
├── scripts/
│   ├── launch_hf_job.py     # CLI wrapper for HF Job submission
│   ├── deploy_spaces.sh     # Push to HF Space
│   └── verify_spaces.sh     # Health-check the deployed Space
│
├── tests/
│   ├── test_server.py
│   ├── test_integration.py
│   └── test_properties.py
│
├── images/                  # Static assets for dashboard + README
├── data/                    # Runtime data (leaderboard.json — not committed)
└── PROJECT_STRUCTURE.md     # Detailed description of every file and folder
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for a detailed description of every file.

---

## Research Foundation

ALICE synthesises ideas from several lines of prior work:

| Strategy | Original Work | ALICE's version |
|---|---|---|
| RLHF | InstructGPT (Ouyang et al., 2022) | Oracle discrimination reward |
| Constitutional AI | Anthropic (Bai et al., 2022) | Verifier stack as constitution |
| Self-play | SPAG (Cheng et al., 2024) | Hunt agent in discrimination zone |
| Curriculum Learning | Bengio et al., 2009 | 20–80% success rate zone |
| Dynamic Benchmarks | ARC-AGI | Auto-generated from model's own failures |
| GRPO | DeepSeek-R1 (Guo et al., 2025) | Group-normalised advantages, no critic |
| Scalable reasoning | DeepSeek-V3 Technical Report (2025) | Multi-stage post-training: SFT → RL, no human labels |
| Open post-training lifecycle | OLMo 3 Technical Report (AI2, 2025) | Full open pipeline: base → SFT → DPO → RL alignment |
| Minimax self-play | MiniMax-01 (MiniMax, 2025) | Adversarial task generation from model's own failure modes |
| DAPO | DAPO: Direct Alignment from Preference Optimisation (2025) | Entropy regularisation to prevent policy collapse |
| Adversarial red-teaming | MART / Rainbow Teaming (Perez et al., 2022; Samvelyan et al., 2024) | Failure bank + novelty-filtered repair queue |
| Contamination-resistant benchmarks | LiveCodeBench (Jain et al., 2024) | RL-generated tasks replace static test sets |

### Multistage post-training

ALICE's design is informed by the full LLM training lifecycle (base model → SFT → DPO/RLHF → RL):

- **Base model** — learns the distribution of text; no instruction following
- **SFT** — supervised fine-tuning on curated demonstrations; teaches format and basic task completion
- **DPO / RLHF** — aligns model to human preferences; reduces harmful outputs
- **RL (ALICE's zone)** — continuous online improvement against a dynamic environment; no static dataset

ALICE targets the RL stage specifically. The Failure Bank's repair queue produces SFT-quality (task, correction) pairs that can feed back into earlier stages, closing the full loop.

### Novelty: what exists vs what ALICE uniquely does

Prior work has explored adjacent problems:

| System | What it does | What's missing |
|---|---|---|
| MART, APL, Rainbow Teaming | Adversarial hunt + repair loops | Safety-only; benchmark does not co-evolve |
| Rainbow Teaming, MAP-Elites for LLMs | Diverse adversarial prompt generation | No repair loop; no discrimination reward |
| SPELL, Multi-Agent Evolve | Self-play capability improvement | Fixed benchmark sets; no task generation |
| LiveCodeBench, GAIA2 | Contamination-resistant dynamic benchmarks | Human-constructed; not RL-generated; no repair loop |

**What does not exist anywhere** — a system where:

1. A benchmark generator is trained specifically on **inter-model discrimination score** (not just "hard for model X"), and
2. Its outputs feed directly into a **capability repair loop** (not safety), and
3. The repair loop's success forces the benchmark to **co-evolutionarily escalate**

The co-evolutionary framing with a discrimination reward is the gap ALICE fills.

---

## Citation

```bibtex
@software{alice_rl_environment,
  title  = {ALICE: Adversarial Loop for Inter-model Co-evolutionary Environment},
  author = {Banerjee, Debdeep and Jain, Rohan},
  year   = {2026},
  url    = {https://huggingface.co/spaces/rohanjain1648/alice-rl-environment}
}
```

---

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.

---

*Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — the open standard for LLM RL environments.*
