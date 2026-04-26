# ALICE: The Benchmark That Trains Itself

*Adversarial Loop for Inter-model Co-evolutionary Environment*

---

> "Every model you see topping a leaderboard today — it's not as good as it looks."

Not because the engineers are lying. Because the benchmarks are broken in a way that no single engineer can fix.

Here's what actually happens. A benchmark is released. Teams train on adjacent data. Scores climb. Within months, every competitive model clusters near the ceiling. The Hugging Face Open LLM Leaderboard team wrote about this openly — they redesigned the benchmark, and within another few months, the new one saturated too. It's a treadmill with no exit ramp.

![HF Open LLM Leaderboard — why we redesigned it](https://raw.githubusercontent.com/DEBDEEP-BANERJ2E/ALICE_final/main/images/hf_blog_1.jpeg)
*The benchmark saturation problem as documented by Hugging Face themselves.*

![Benchmark saturation: top scores converging](https://raw.githubusercontent.com/DEBDEEP-BANERJ2E/ALICE_final/main/images/hf_blog_2.jpeg)

![Saturation causes: contamination, easy tasks, benchmark errors](https://raw.githubusercontent.com/DEBDEEP-BANERJ2E/ALICE_final/main/images/hf_blog_3.jpeg)

But that's only half the problem. The other half is this: even *during* training, we have no instrument for finding what a model genuinely cannot do. Standard benchmarks measure known capabilities. They cannot find the blind spots — the reasoning combinations, the compositional edge cases, the failure modes that exist in the model right now but that no human thought to test. A model can score 87 on MATH and still fail systematically on negation-plus-arithmetic. Nobody knows. There's no tool to find it.

We built ALICE to be that tool.

---

## What ALICE Is

**ALICE — Adversarial Loop for Inter-model Co-evolutionary Environment** — is a reinforcement learning training environment. It's [OpenEnv-compatible](https://github.com/meta-pytorch/OpenEnv), deployed as a Hugging Face Docker Space, and pluggable into any TRL or Unsloth training run via a simple HTTP API.

The core idea is a co-evolutionary mechanism: the benchmark adapts to the model as the model improves. There is no static task set. There is no ceiling to saturate.

Here is the architecture:

![ALICE co-evolutionary architecture](https://raw.githubusercontent.com/DEBDEEP-BANERJ2E/ALICE_final/main/images/alice_architecture.png)

The loop works like this:

1. An **Oracle Interface** queries stronger reference models to find the *discrimination zone* — tasks the reference model passes but your model fails.
2. A **Hunt agent** adversarially searches that zone for novel failure modes specific to your model right now.
3. A **three-tier Verifier** confirms each failure is real: sandboxed code execution (T1), two independent LLM judges with chain-of-thought rubrics (T2), and a 20-variant regression battery (T3).
4. Confirmed failures go to the **Failure Bank**, deduplicated by semantic novelty using sentence-transformer embeddings and k-NN cosine similarity. If it's not genuinely new, it gets discarded.
5. A **Repair agent** synthesises targeted training pairs to fix confirmed novel failures.
6. **GRPO** (Group Relative Policy Optimisation, DeepSeek-R1 style) provides the gradient signal — no value network, no critic.
7. When the model improves on a task, the Oracle's discrimination reward collapses on it. It's no longer informative. The curriculum is structurally forced to escalate — harder tasks, higher tier, continued improvement.

The key insight: **co-evolution is self-sustaining**. You don't redesign the benchmark by hand. The model's own improvement forces the environment to grow with it.

---

## The Dashboard: A Tour of All 8 Tabs

The live environment runs at [rohanjain1648-alice-rl-environment.hf.space](https://rohanjain1648-alice-rl-environment.hf.space). There are eight tabs. Here's what each one does and why a researcher training their own model would care.

---

### Tab 0 — Mission

The Mission tab is where first-time visitors land. It's structured as four accordions that open progressively.

**"The Problem: Why Benchmarks Are Broken"** — opens by default. This is the argument: static benchmarks saturate, models contaminate them, and the community loses the signal it needs to measure real capability differences. The HF leaderboard saturation screenshots live here.

**"Our Solution: ALICE Architecture"** — the architecture diagram plus a comparison table showing how ALICE synthesises prior work: RLHF's oracle discrimination, Constitutional AI's self-critique, SPAG's self-play adversarial generation, and ARC's dynamic benchmark framing — all in a single loop.

**"Research Foundation — Papers & Articles"** — a curated reading list with links to the papers ALICE is built on: InstructGPT, DeepSeek-R1, DAPO, Rainbow Teaming, LiveCodeBench, OLMo 3. If you want to understand why specific design choices were made, start here.

**"How to Use ALICE — Step by Step"** — the practical onboarding guide. Five steps: clone or point at the live Space, call `/reset` to start an episode, call `/step` with your model's action, watch the reward signal, push training metrics to see live charts. This accordion is the fastest path from "what is this?" to "I'm training against it."

**For users training their own models:** The Mission tab is the most important for orientation. The architecture diagram answers the question every researcher asks: "What is this actually doing to my model?" The research papers table answers: "Is this grounded in real work or just vibes?"

---

### Tab 1 — Overview

The Overview tab is the live heartbeat of the environment.

Eight metric cards refresh every 3 seconds:
- **Episodes Run** — total episodes completed since server start
- **Total Rollouts** — total individual turns (episodes × turns/episode)
- **Avg Reward (last 20)** — the most recent reward signal; should trend up as training progresses
- **Success Rate** — fraction of episodes where the composite verifier score ≥ 0.5
- **Disc Coverage** — the fraction of tasks where the model's success rate is 20–80% (the discrimination zone; more on this below)
- **Difficulty Tier** — which tier of the curriculum the environment is currently running
- **Failure Bank Size** — how many novel confirmed failures have been logged
- **Uptime** — how long the server has been running

Below the cards: the active episode display (episode ID, turn number, agent version, current task text), health metrics (error rate, latency P95, memory), system alerts if anything is abnormal, and a recent episode activity log.

There's also an optional accordion, "RL Concepts in This Tab," which explains what each term means for someone new to reinforcement learning — why discrimination coverage matters, what reward means in this context, what "agent version" is tracking.

**For users training their own models:** This is the tab you keep open on a second monitor during a training run. The avg reward and success rate numbers are the primary signals that training is working. The failure bank size growing means the environment is finding genuinely new things your model can't do — which is exactly what you want.

---

### Tab 2 — Training Metrics

This is the experiment tracking tab. No wandb. No MLflow. No external service. Built in.

Every training script calls `POST /training/push` after each episode, sending rewards, advantages, loss, success rate, and disc_coverage. The Training Metrics tab visualises this in a 4-panel auto-refreshing chart:

- **Reward curve** — mean reward per episode over time. Should trend upward as the model learns.
- **Cumulative reward** — running total. A linear slope means constant learning; an accelerating slope means compound improvement.
- **Reward distribution** — histogram of per-episode rewards. A bimodal distribution (spikes at 0 and 1) is normal early in training; it should smooth toward a single peak as the policy becomes consistent.
- **GRPO loss** — policy gradient loss. Should be non-zero and eventually decrease. If it goes to zero while avg_reward stays flat, advantages have collapsed — all rollouts in the group are getting the same reward, so `std(r) = 0` and the gradient is zero.

There's a **Start Training** button that submits a free-tier HF Job (Qwen/Qwen2.5-0.5B-Instruct, 50 episodes, cpu-basic) with one click. No GPU credits required. Metrics appear in the charts in real time as the job runs.

The optional "GRPO Math" accordion explains the full update rule:

```
advantage_i = (r_i − mean(r)) / std(r)
Loss = −mean( advantage_i × log_prob(action_i) ) + KL_coef × log_prob²
```

And explains why advantages can all be zero, what the KL term is doing, and how to read each panel.

Below the charts: a Recent Episodes table showing episode IDs, task snippets, difficulty scores, rewards, and timestamps.

**For users training their own models:** This is your training log, live. You can watch reward improve in real time without running a separate logging stack. The `POST /training/push` endpoint is the integration point — add five lines to your training loop and all four panels populate automatically.

---

### Tab 3 — Curriculum

The Curriculum tab shows how the difficulty of tasks is managed over time.

Two charts:

**Domain × Difficulty Heatmap** — a 5×10 grid. Rows are task domains (arithmetic, logic, factual, symbolic, code). Columns are difficulty tiers (T1 through T10). Each cell's color is the model's success rate across all attempts in that domain/tier combination: red = 0%, yellow ≈ 50%, green = 100%.

At the start of training, everything is red — the model has no history. As training runs, cells turn yellow and green as the model attempts and solves tasks. The curriculum deliberately keeps the model in the 20–80% zone, so you expect to see a band of yellow-green cells at the model's current capability frontier, with red both below (too hard, not yet attempted) and above (already mastered, curriculum has moved on).

**Discrimination Zone Coverage Over Time** — a time series showing what fraction of tasks are in the 20–80% success zone. The green dashed line is the 70% target. When coverage exceeds 70% for 50+ consecutive episodes, the difficulty tier increments and harder tasks are introduced.

The math behind the discrimination zone is in the optional accordion: a task with success rate p=0.5 maximises variance in the Bernoulli distribution (σ² = p(1−p) = 0.25 at its peak). More variance → more informative gradients → faster learning. This is Vygotsky's "zone of proximal development" expressed as an ML optimisation criterion.

**For users training their own models:** The heatmap tells you which domains your model is weakest at — not just overall, but at each difficulty tier. If code/T3 is still red after 100 episodes, you know the curriculum hasn't reached that combination yet, or your model is systematically blocked there. The discrimination coverage line is the primary diagnostic: if it stays flat near 0%, tasks are either all too easy or all too hard, and you need to adjust the starting difficulty tier.

---

### Tab 4 — HF Space & Jobs

This tab is the operations view for training jobs.

Click **Refresh HF Status** to pull the live status of the ALICE environment Space from the HF API: running (🟢) or stopped (🔴), with the direct URL.

Below that, a **Training History** table shows all registered training jobs with columns: model name, job ID (linked to the HF Jobs dashboard), status (🟢 RUNNING / ✅ COMPLETED), avg reward, success rate, and a direct View link.

The optional "What Are HF Jobs?" accordion explains the compute model: HF Jobs are serverless containers that spin up, run a Python script, and tear down. Pay-per-second. No idle cost. The training script loads the model locally with LoRA, runs GRPO episodes against the ALICE Space API, and streams metrics back in real time. An a10g-small (24 GB VRAM) handles up to 3B models without 4-bit quantisation at roughly 7 seconds/episode for 0.5B models.

Quick links at the bottom: direct links to the HF Space, API Swagger docs, and the HF Jobs dashboard.

**For users training their own models:** This tab answers "is my training job still running?" without leaving the dashboard. The job history table persists across refreshes, so you can track multiple model evaluations in one place. The per-job avg reward column lets you compare model checkpoints before the leaderboard is updated.

---

### Tab 5 — Failure Bank

The Failure Bank tab is where the environment's memory lives.

Three summary metrics:
- **Total Failures** — how many confirmed failures are stored
- **Avg Novelty** — the mean novelty score (0–1) of stored failures; higher means the bank is more semantically diverse
- **Repair Queue** — how many failures are staged for targeted fine-tuning

Below: a filterable table with columns for failure ID, error type, agent version, novelty score, and timestamp. You can filter by error type (`verification_failure`, etc.) or by agent version (the model ID that produced the failure).

The optional accordion explains the full pipeline: a failure is any episode where `composite_score < 0.5`. Before storing it, the system embeds the (task, action) pair using `sentence-transformers/all-MiniLM-L6-v2`, queries the k-NN index (k=16) for cosine similarity, computes `novelty = 1 − max_similarity`, and only stores the failure if `novelty > 0.7`. The rest are discarded as "already known."

Failures in the repair queue are the inputs to the Repair agent, which generates verified (task, correct_solution) pairs for targeted SFT before the next GRPO round.

**For users training their own models:** The Failure Bank is the most actionable tab for understanding *what* your model is getting wrong. Filtering by your model's agent version shows you only your model's confirmed, semantically novel failures. The novelty score tells you whether your model is making the same class of mistakes repeatedly (low novelty, same failure type) or discovering genuinely new failure modes (high novelty, diverse failures). A growing failure bank with high avg novelty is a sign the environment is productively challenging your model.

---

### Tab 6 — Leaderboard

The Leaderboard tab shows all models ranked by composite RL score.

The composite score formula:

```
rl_score = 0.5 × avg_reward + 0.3 × success_rate + 0.2 × disc_coverage
```

A bar chart at the top, then a ranked table with columns: rank, model name, parameter count, rl_score, avg_reward, success_rate, disc_coverage, episodes run, and source (submitted via API or dashboard).

Current rankings from 5 benchmark runs:

| Rank | Model | Params | rl_score | avg_reward | success_rate | disc_coverage |
|---|---|---|---|---|---|---|
| 1 | Qwen2.5-0.5B | 0.5B | **1.1189** | 1.6549 | 73.75% | 73.62% |
| 2 | Qwen2.5-3B | 3.0B | 0.8992 | 1.4278 | 58.00% | 25.50% |
| 3 | Qwen2.5-1.5B | 1.5B | 0.5251 | 0.8515 | 17.75% | 14.50% |
| 4 | SmolLM2-1.7B | 1.7B | 0.3015 | 0.5279 | 8.25% | 6.25% |
| 5 | Gemma-3-1B | 1.0B | 0.0000 | — | — | — |

Note the headline result: a **0.5B model outperforms a 3B model**. The capability gap is explained by training time (150 episodes vs 50), not parameter count. This is what a functioning RL loop produces: more episodes → better policy, regardless of base model size.

The **Submit a Model** form at the bottom takes a HF model ID, display name, parameter count, and episode count, and submits a real HF Job that runs RL episodes against the ALICE environment. Results push to the leaderboard automatically when the job completes. One click. The job URL is returned immediately.

The optional accordion explains the composite score components in detail, including why a perfect-scoring model would have `disc_coverage ≈ 0` — if the model solved every task easily, the curriculum would escalate until it was failing again, collapsing coverage back toward zero.

**For users training their own models:** Submit your model before and after a training run to get a before/after RL score. The disc_coverage delta is the most informative signal — a rising coverage means the environment is successfully finding your model's frontier, not just giving it easy wins. The leaderboard is also a comparison baseline: if your fine-tuned 1.5B model scores below Qwen2.5-0.5B after 150 episodes, the training loop needs attention.

---

## The Verifier Stack: Why You Can Trust the Scores

Every action submitted to ALICE is graded by a three-tier pipeline. The composite score:

```
composite = 0.4 × T1 + 0.4 × T2 + 0.2 × T3
```

**T1 — RestrictedPython Sandbox.** The agent's Python is executed in an isolated sandbox. `open`, `exec`, `eval`, `__import__`, and `compile` are all blocked. 5-second CPU timeout, 512 MB memory cap. Returns 0.0 or 1.0.

**T2 — Dual LLM Judge.** Two reference models (`REFERENCE_MODEL_PRIMARY` and `REFERENCE_MODEL_SECONDARY`) independently score the response using a chain-of-thought rubric: correctness, reasoning depth, conciseness. The training model never judges itself. Returns the average of both scores in [0.0, 1.0].

**T3 — Regression Battery.** The answer is tested against 20 deterministic variants of the task with known ground-truth answers. Hard pass/fail. Returns a score in [0.0, 1.0].

The dual-judge design is specifically to prevent reward hacking: a policy that fools one judge rarely fools two independent judges using different prompts. The regression battery catches surface-form matching — if a model has learned to produce the right surface syntax without understanding the task, it will fail on 20 novel variants.

---

## Episode Structure: Three Turns

Every ALICE episode is a 3-turn interaction:

```
POST /reset
  └─▶ { episode_id, task, timestamp, agent_version }

Turn 1: POST /step { episode_id, action }
  └─▶ { reward, done=false, info: { verification, feedback } }

Turn 2: POST /step — uses T1 feedback to correct
  └─▶ { reward, done=false, info: { verification, feedback } }

Turn 3: POST /step — final attempt
  └─▶ { reward, done=true, info: { verification } }
```

Each `action` must be a single Python assignment: `result = <value>`.

The reward shaping across turns:
- Turn 1: `reward = tier1_score` (programmatic pass/fail)
- Turn 2: `reward = γ × composite_score` (turn-decay multiplier)
- Turn 3: `reward = γ² × composite_score + novelty_bonus` (final attempt + novelty)

A model that improves across turns — turn 1 reward low, turn 3 reward high — demonstrates in-context self-correction, which is a direct capability signal. This is visible per-episode in the episode detail table in Training Metrics.

---

## Training Your Own Model Against ALICE

There are four paths, ordered from simplest to most powerful.

**1. Free-tier HF Job (no GPU, no credits)** — `training/hf_cpu_job.py`. Uses the HF Inference API for inference. Launch from the Training Metrics tab in one click, or submit manually:

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
    },
    secrets={"HF_TOKEN": "hf_..."},
    token="hf_...",
)
```

**2. GPU training job with LoRA** — `training/hf_job_train.py`. Full GRPO with local LoRA fine-tuning on an `a10g-small` (24 GB VRAM). ~7 s/episode for 0.5B.

**3. Pure TRL GRPO** — `training/train_trl.py`. Standard TRL integration:

```bash
python training/train_trl.py \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --episodes 200 \
    --load_in_4bit \
    --update_leaderboard
```

**4. Unsloth + TRL** — `training/train_unsloth.py`. ~2× faster, ~60% less VRAM. Falls back to standard transformers if Unsloth isn't installed.

All four paths push metrics to `POST /training/push` after each episode, and the Training Metrics tab picks them up live.

---

## Evidence That Training Works

Three signals from the benchmark runs show the environment produces real capability improvement, not noise:

**Signal 1 — Absolute score growth over episodes.** Qwen2.5-0.5B at 150 episodes outperforms Qwen2.5-3B at 50 episodes across every metric. A model 6× smaller wins by training time. The RL loop is producing genuine capability gains.

**Signal 2 — Discrimination coverage as a learning signal.** Qwen2.5-0.5B reaches 73.6% disc_coverage — meaning the curriculum has successfully escalated difficulty to stay at the model's frontier. A model gaming the reward (always winning easy tasks) would show low coverage as the curriculum pushed it to harder tasks it can't solve. High coverage at high reward confirms the model is operating at its capability boundary, not gaming.

**Signal 3 — Multi-turn self-correction.** Per-episode turn trajectories (visible in the episode detail table) show rewards improving from turn 1 to turn 3. The model uses T1 feedback to correct its answer in subsequent turns — an in-context self-correction capability signal.

---

## Why This Matters for Your Training Runs

Every researcher who has trained an LLM has hit the same wall: you run the benchmark, scores improve, then plateau, and you don't know if the plateau is real capability saturation or benchmark saturation. You don't know what your model genuinely can't do.

ALICE answers both questions simultaneously. The co-evolutionary loop means there is no ceiling to hit — the benchmark escalates when the model improves. The failure bank means you always have a list of what your model genuinely cannot do, ordered by novelty and queued for repair.

The practical benefit for someone training their own model:

- **Before training:** Submit your base model to the leaderboard. Get a baseline rl_score and a failure bank populated with your model's specific blind spots.
- **During training:** Watch the Training Metrics tab. The reward curve and GRPO loss tell you whether training is working. The disc_coverage line tells you whether the curriculum is staying at your model's frontier.
- **After training:** Submit your fine-tuned checkpoint. Compare rl_score before and after. The delta is a contamination-resistant capability improvement signal — ALICE never uses the same tasks twice.

The environment is live. The API is open. The training scripts are in the repo.

---

## Links

| Resource | URL |
|---|---|
| **Live Dashboard** | [rohanjain1648-alice-rl-environment.hf.space](https://rohanjain1648-alice-rl-environment.hf.space) |
| **API Swagger Docs** | [/docs](https://rohanjain1648-alice-rl-environment.hf.space/docs) |
| **HF Space** | [huggingface.co/spaces/rohanjain1648/alice-rl-environment](https://huggingface.co/spaces/rohanjain1648/alice-rl-environment) |
| **Colab (TRL)** | [train_trl_colab.ipynb](https://colab.research.google.com/github/rohanjain1648/alice-rl-environment/blob/main/notebooks/train_trl_colab.ipynb) |
| **Colab (Unsloth)** | [train_unsloth_colab.ipynb](https://colab.research.google.com/github/rohanjain1648/alice-rl-environment/blob/main/notebooks/train_unsloth_colab.ipynb) |
| **GitHub** | [DEBDEEP-BANERJ2E/ALICE_final](https://github.com/DEBDEEP-BANERJ2E/ALICE_final) |

---

```bibtex
@software{alice_rl_environment,
  title  = {ALICE: Adversarial Loop for Inter-model Co-evolutionary Environment},
  author = {Banerjee, Debdeep and Jain, Rohan},
  year   = {2026},
  url    = {https://huggingface.co/spaces/rohanjain1648/alice-rl-environment}
}
```

*Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — the open standard for LLM RL environments.*
