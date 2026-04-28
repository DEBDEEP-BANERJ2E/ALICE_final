# ALICE RL Environment — Tasks 1, 2, 3 Progress Notes

A plain-English walkthrough of what was built, why, and how.

---

## Task 1 — Scaffold the Project Structure

### What is this?

Before writing any real logic, we needed a clean folder structure so every component has a home. Think of it like setting up the rooms in a house before moving furniture in.

### What was created

```
alice_env/
├── server.py                  # The main API server (FastAPI)
├── server/                    # OpenEnv-compatible server wrapper
│   ├── app.py
│   └── alice_env_environment.py
├── environment/               # All the RL environment logic
│   ├── episode_handler.py
│   ├── task_generator.py
│   ├── curriculum_manager.py
│   ├── oracle_interface.py
│   ├── verifier_stack.py
│   ├── failure_bank.py
│   ├── reward_function.py
│   └── state.py
├── monitors/                  # Anti-hacking safety monitors
│   ├── sandbox.py
│   ├── trajectory_sampler.py
│   └── entropy_monitor.py
├── dashboard/
│   └── gradio_app.py          # Visual monitoring UI (task 14)
├── training/
│   └── train.py               # GRPO trainer (task 12)
├── tests/
│   ├── test_server.py
│   ├── test_properties.py
│   └── test_integration.py
├── scripts/
│   ├── deploy_spaces.sh
│   └── launch_training.sh
├── Dockerfile
├── pyproject.toml
└── README.md
```

### Key concepts

**uv** — a fast Python package manager (like pip but much faster). We use `uv sync` to install all dependencies from `pyproject.toml`.

**pyproject.toml** — the single config file that declares what the project is, what it depends on, and how to run it. Replaces the old `setup.py` + `requirements.txt` combo.

**`__init__.py`** — an empty file that tells Python "this folder is a package you can import from". Every subfolder that contains code needs one.

### Result

A fully wired project skeleton where every file exists and every package is importable. Running `uv sync` installs all dependencies into a local `.venv`.

---

## Task 2 — Implement the Core Environment Contract (OpenEnv Server)

This was the biggest task. It has three parts.

---

### Task 2.1 — The FastAPI Server

#### What is an "environment contract"?

In reinforcement learning, the training loop needs to talk to the environment in a standard way. The standard here is **OpenEnv**, which says every environment must expose four operations:

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/reset` | POST | Start a new episode. Returns the first task for the agent to solve. |
| `/step` | POST | Send the agent's answer. Returns the result (new state, reward, done flag). |
| `/state` | GET | Peek at the current state without changing anything. |
| `/health` | GET | Check if the server is alive and how it's performing. |

#### What is FastAPI?

FastAPI is a Python web framework. You write Python functions and it automatically turns them into HTTP endpoints. It also auto-generates a Swagger UI at `/docs` so you can test the API in a browser.

#### What is an episode?

An episode is one complete interaction between the agent and the environment. In ALICE, every episode is exactly **3 turns**:
- Turn 1: Agent sees the task and gives its first answer
- Turn 2: Agent gets feedback on what went wrong and tries again
- Turn 3: Agent gets a hint and makes its final attempt

After 3 turns, the episode ends and a new one begins with a fresh task.

#### Thread safety

Multiple training jobs might call `/step` at the same time. We use `asyncio.Lock()` — a lock that prevents two requests from modifying the shared state simultaneously, like a "one at a time" sign on a door.

#### Request logging middleware

Every request is timed and logged. The middleware wraps every endpoint call, records how long it took, and increments error counters. This feeds the `/health` endpoint's metrics.

#### What `/health` returns

```json
{
  "uptime": 128.3,          // seconds since server started
  "error_rate": 0.02,       // fraction of requests that errored
  "latency_p95": 0.003,     // 95th percentile response time in seconds
  "memory_usage": 75.3      // RAM used in MB
}
```

#### The root `/` route

Added a simple HTML status page so the HF Spaces iframe shows something useful instead of a blank screen. It lists all available endpoints and links to the auto-generated API docs.

---

### Task 2.2 — Unit Tests for the Server

#### Why test?

Tests prove the server behaves correctly. If someone changes the code later and breaks something, the tests catch it immediately.

#### What was tested (22 tests, all passing)

**Health tests:**
- `/health` returns HTTP 200
- Response contains `uptime`, `error_rate`, `latency_p95`, `memory_usage`
- Uptime is non-negative
- Error rate is between 0 and 1

**Reset tests:**
- `/reset` returns HTTP 200
- Response contains `episode_id`, `task`, `timestamp`, `agent_version`
- Two consecutive resets give different episode IDs (each episode is unique)

**State tests:**
- `/state` returns HTTP 200
- After a reset, `episode_id` is populated
- Calling `/state` twice in a row gives the same result (no side effects)

**Step tests:**
- `/step` with a valid action returns `state`, `reward`, `done`, `info`
- Reward is a number, `done` is a boolean
- Invalid `episode_id` returns HTTP 400 (bad request)
- Empty action returns HTTP 422 (validation error)
- After 3 steps, `done` is `True`
- An invalid step doesn't corrupt the state

---

### Task 2.3 — Deploy to HF Spaces

#### What is Hugging Face Spaces?

HF Spaces is a free hosting platform for ML apps. When you push a Docker Space, it builds your `Dockerfile` and runs your container on their servers. The result is a publicly accessible URL.

#### The three-component model

A single HF Space gives you three things at once:

| Component | How to use it | Purpose |
|-----------|--------------|---------|
| **Server** | `https://rohanjain1648-alice-rl-environment.hf.space` | Live HTTP API — the trainer calls this during training |
| **Repository** | `git clone https://huggingface.co/spaces/rohanjain1648/alice-rl-environment` then `pip install .` | Installable Python package — use the environment code locally |
| **Registry** | `docker pull registry.hf.space/rohanjain1648-alice-rl-environment:latest` | Docker image — run the environment anywhere |

This means you don't need three separate services. One Space = one URL = all three components.

#### What the Dockerfile does

```dockerfile
FROM python:3.11-slim          # Start from a minimal Python image
COPY pyproject.toml .          # Copy dependency list
RUN uv pip install ...         # Install only server deps (not torch/trl — too heavy for free tier)
COPY . .                       # Copy all code
EXPOSE 7860                    # HF Spaces expects port 7860
CMD ["uvicorn", "alice_server:app", ...]  # Start the server
```

We install only the server-side dependencies (FastAPI, uvicorn, sentence-transformers, etc.) and skip `torch`/`trl`/`transformers` — those are only needed for training, not for running the environment server. This keeps the Docker image small enough to build on the free CPU tier.

#### The `alice_server.py` naming fix

The project has both a `server.py` file and a `server/` directory. Python resolves `import server` to the directory (package), not the file. So `uvicorn server:app` would fail because it finds the package, not the FastAPI app. The fix: copy `server.py` to `alice_server.py` and point uvicorn at `alice_server:app` instead.

#### The `openenv.yaml` removal

The scaffolded project included an `openenv.yaml` file that told the OpenEnv framework to run `uvicorn server.app:app`. This conflicted with our Dockerfile CMD. We deleted it from the Space so the Dockerfile is the sole entrypoint.

#### Deployment scripts

**`scripts/deploy_spaces.sh`** — automates the full deployment:
1. Authenticates with HF
2. Creates the Space if it doesn't exist (idempotent — safe to run multiple times)
3. Pushes all code to the Space repository
4. Polls `/health` until the container is running
5. Verifies all three components
6. Writes `ALICE_ENV_URL` to `.env`

**`scripts/verify_spaces.sh`** — smoke-tests a deployed Space:
- Checks `/health` returns 200 with expected fields
- Checks `/state` returns 200
- Verifies the HF Space page is reachable (pip install URL)
- Verifies the HF registry is reachable (docker pull URL)

#### Canonical URL

```
ALICE_ENV_URL=https://rohanjain1648-alice-rl-environment.hf.space
```

This is recorded in `.env` and used by all subsequent components (trainer, dashboard, integration tests).

---

## Task 3 — MDP State Representation

### What is an MDP?

MDP stands for **Markov Decision Process**. It's the mathematical framework that describes how reinforcement learning works:

- **State (S)**: What the agent currently knows about the world
- **Action (A)**: What the agent does
- **Transition (P)**: How the world changes after an action
- **Reward (R)**: How good or bad the action was
- **Discount (γ)**: How much to value future rewards vs immediate ones

The "Markov" part means: **the next state depends only on the current state and action, not on history**. Everything the agent needs to know must be encoded in the current state vector.

### What is the state vector?

ALICE's state is a flat array of 13,065 numbers:

| Component | Size | What it represents |
|-----------|------|--------------------|
| `task_embedding` | 768 | The current task encoded as a vector using a sentence-transformer model |
| `agent_capability_vector` | 5 | The agent's pass rates on 5 skill types: arithmetic, logic, factual, symbolic, code |
| `difficulty_tier` | 1 | Current curriculum difficulty level (1–10) |
| `turn_number` | 1 | Which turn we're on (1, 2, or 3) |
| `failure_bank_snapshot` | 12,288 | Top-16 recent failure embeddings (16 × 768), zero-padded if fewer than 16 |
| `discrimination_coverage` | 1 | Fraction of tasks in the "optimal difficulty" zone |
| `cumulative_reward` | 1 | Total reward accumulated so far in this episode |
| **Total** | **13,065** | |

### What is a sentence-transformer?

A sentence-transformer is a neural network that converts text into a vector of numbers (an "embedding"). Similar sentences produce similar vectors. We use `all-MiniLM-L6-v2` which outputs 384-dimensional vectors, zero-padded to 768 to match the state spec.

### What is the failure bank snapshot?

The failure bank stores tasks the agent got wrong. The snapshot is the top-16 most recent/novel failures, each encoded as a 768-dim embedding. If there are fewer than 16 failures, the remaining slots are filled with zeros. This gives the agent context about what it has struggled with recently.

### The `MDPState` dataclass

```python
@dataclass
class MDPState:
    task_embedding: np.ndarray          # 768 floats
    agent_capability_vector: np.ndarray # 5 floats
    difficulty_tier: int                # 1 integer
    turn_number: int                    # 1 integer
    failure_bank_snapshot: np.ndarray   # 12,288 floats
    discrimination_coverage: float      # 1 float
    cumulative_reward: float            # 1 float
```

Key methods:
- **`to_vector()`** — concatenates all fields into a single `(13065,)` numpy array
- **`from_vector(vec)`** — splits a flat array back into the structured fields (round-trip)
- **`encode_task(task)`** — runs a sentence-transformer to get the 768-dim embedding
- **`encode_failure_bank_snapshot(embeddings)`** — stacks up to 16 embeddings with zero-padding

### Property 1: The Markov Property

The Markov property test verifies that **given the same state and action, the environment always produces the same next state and reward**. No hidden state, no randomness (given fixed seeds).

Test approach: initialize two identical episode handlers with the same episode ID, task, and agent version. Step both with the same action. Assert the outputs are identical.

This is important because if the environment had hidden state (e.g., a random number generator that wasn't seeded), training would be non-reproducible and harder to debug.

### Tests added (8 total, all passing)

| Test | What it checks |
|------|---------------|
| `test_to_vector_shape` | Output is exactly (13065,) |
| `test_from_vector_round_trip` | Serialize → deserialize gives back the same values |
| `test_from_vector_wrong_size_raises` | Wrong-size input raises `ValueError` |
| `test_encode_failure_bank_snapshot_zero_padding` | Fewer than 16 embeddings → rest are zeros |
| `test_encode_failure_bank_snapshot_truncates_to_k` | More than 16 embeddings → truncated to 16 |
| `test_default_state_vector_is_valid` | Default state has no NaN or Inf values |
| `test_same_action_same_episode_gives_same_turn_structure` | Markov property holds for episode handler |
| `test_deterministic_reward_for_same_verification_result` | Markov property holds for reward function |

---

## Summary

| Task | Status | Tests |
|------|--------|-------|
| 1. Scaffold project | ✅ Done | — |
| 2.1 FastAPI server | ✅ Done | 22 passing |
| 2.2 Server unit tests | ✅ Done | 22 passing |
| 2.3 HF Spaces deployment | ✅ Done | 5/5 smoke tests passing |
| 3.1 MDPState dataclass | ✅ Done | 6 passing |
| 3.2 Markov property test | ✅ Done | 2 passing |

Live environment: `https://rohanjain1648-alice-rl-environment.hf.space`


---

## Task 4 — Implement the Episode Handler

### What is the Episode Handler?

The Episode Handler is the component that manages a single training episode from start to finish. Think of it as the "game master" — it keeps track of which turn we're on, what the agent said, what feedback to give next, and when the episode is over.

Every episode in ALICE is exactly **3 turns**. No more, no less. This is a hard constraint enforced by the handler.

### The 3-turn structure

```
Turn 1 — Initial attempt
  Agent sees: the task
  Agent does: gives its first answer
  Handler records: action, reward placeholder, verification result

Turn 2 — CoT reflection + retry
  Agent sees: the task + "your previous attempt was X, please reflect and retry"
  Agent does: reasons through what went wrong and tries again
  Handler records: action, reward, verification result

Turn 3 — Hint + final attempt
  Agent sees: the task + "hint: consider edge cases carefully"
  Agent does: makes its final attempt
  Handler records: action, reward, verification result
  Episode ends: done = True
```

The escalating scaffolding (more help each turn) is intentional — it mirrors how a human tutor would coach a student: first let them try alone, then give feedback, then give a hint.

### Key methods

**`initialize_episode(episode_id, agent_version, task)`**
Sets up a fresh episode. Resets the turn counter to 1, clears the trajectory, and returns the initial state dict with the task and metadata.

**`step(action)`**
The main method. Called once per turn. It:
1. Builds the observation for the current turn (task + feedback/hint depending on turn number)
2. Records the turn data (action, observation, reward placeholder)
3. Sets `done = True` on turn 3
4. Returns `(state, reward, done, info)`

**`record_turn(turn_data)`**
Appends a turn's data to the trajectory. Stores: turn number, observation, action, CoT trace, reward, done flag, verification result.

**`finalize_episode()`**
Called automatically when turn 3 completes. Computes episode statistics (total reward, success rate) and saves the trajectory to the history buffer.

**`serialize_trajectory()`**
Converts the trajectory to JSON for storage and analysis.

### Trajectory history

The handler keeps the last 1,000 episode trajectories in a `deque` (a circular buffer). When the 1,001st episode completes, the oldest one is automatically dropped. This prevents unbounded memory growth during long training runs.

### What is a trajectory?

A trajectory is the complete record of one episode:

```python
{
  "turns": [
    {"turn_number": 1, "observation": "Task: ...", "action": "...", "reward": 0.0, ...},
    {"turn_number": 2, "observation": "Task: ...\nYour previous attempt...", "action": "...", ...},
    {"turn_number": 3, "observation": "Task: ...\nHint: ...", "action": "...", "done": True, ...},
  ],
  "metadata": {
    "episode_id": "abc-123",
    "agent_version": "0.0.0",
    "timestamp": "2026-04-25T...",
    "difficulty_level": 1,
    "task": "What is 15 * 7?",
    "total_reward": 0.85,
    "success_rate": 0.67,
  }
}
```

### Bug fixed during this task

While running the full test suite, we discovered a regression: the HTML root route (`GET /`) added to `server.py` in task 2.3 was accidentally shadowing `POST /reset`. The root cause was using FastAPI's `Response` class as both the `response_class` parameter and the return type annotation, which confused the router.

Fix: switched to `from fastapi.responses import HTMLResponse` and used `HTMLResponse` consistently. Also restored the missing `@app.post("/reset", response_model=ResetResponse)` decorator that had been accidentally stripped.

### Property 11: Episode Termination

The property test verifies that **every episode terminates at exactly turn 3**, regardless of what the agent does. This includes pathological inputs like empty strings, 1000-character strings, `None`, and unicode emoji.

The test runs 3 steps and asserts:
- `done` is `False` after turns 1 and 2
- `done` is `True` after turn 3
- The trajectory has exactly 3 recorded turns

### Tests (31 total, all passing)

| Test class | Count | What it covers |
|-----------|-------|---------------|
| `TestEpisodeTermination` | 7 | Property 11 — always terminates at turn 3 |
| `TestMarkovProperty` | 2 | Property 1 — same input → same output |
| `TestStep` (server) | 8 | Step endpoint happy/error paths |
| `TestReset` (server) | 6 | Reset endpoint |
| `TestState` (server) | 4 | State endpoint idempotency |
| `TestHealth` (server) | 4 | Health metrics |

---

## Task 5 — Implement Task Generator (Hunt and Repair Modes)

### What is the Task Generator?

The Task Generator is the component that creates tasks for the agent to solve. It has two modes:

1. **Hunt Mode** — generates adversarial prompts designed to find the agent's weaknesses
2. **Repair Mode** — synthesizes training pairs from tasks the agent previously failed

Think of Hunt mode as a teacher creating challenging exam questions, and Repair mode as creating study materials from past mistakes.

### Hunt Mode: Adversarial Prompt Generation

#### What does "adversarial" mean here?

Not malicious — it means "strategically challenging". The goal is to generate tasks that are:
- Hard enough to be interesting (not trivial)
- Easy enough to be solvable (not impossible)
- Targeted at the agent's current weaknesses

This is called the **discrimination zone** — tasks where the agent succeeds 20-80% of the time. Too easy (>80% success) and the agent doesn't learn. Too hard (<20% success) and the agent gives up.

#### The four adversarial strategies

| Strategy | What it does | Example |
|----------|-------------|---------|
| `semantic_perturbation` | Changes the meaning while keeping similar words | "What is 5 + 3?" → "What is 5 plus three?" (mixing digits and words) |
| `logical_contradiction` | Introduces conflicting constraints | "Find a number greater than 10 and less than 5" |
| `context_confusion` | Provides misleading examples or context | "Like how 2+2=5, what is 3+3?" |
| `boundary_testing` | Tests edge cases and limits | "What is 0 divided by 0?" or "What is infinity minus infinity?" |

The generator picks a strategy based on the agent's weakness profile. If the agent struggles with arithmetic, it uses `boundary_testing` on math problems. If it struggles with logic, it uses `logical_contradiction`.

#### What Hunt mode returns

```python
{
  "prompt": "What is the result of 15 * 7 when calculated in base 8?",
  "difficulty_score": 65,  # 0-100 scale
  "strategy": "semantic_perturbation",
  "reasoning": "Agent has 45% success rate on base conversion tasks, targeting discrimination zone",
  "cot_trace": "Step 1: Identify agent weakness (base conversion)...\nStep 2: Select strategy..."
}
```

#### Prompt history deduplication

The generator maintains a set of previously generated prompts. Before returning a new prompt, it checks if it's too similar to a recent one (using fuzzy string matching). If so, it regenerates. This prevents the agent from seeing the same task twice in a short window.

### Repair Mode: Training Pair Synthesis

#### What is a training pair?

A training pair is:
```
(prompt, correct_solution, reasoning)
```

For example:
```python
{
  "prompt": "What is 15 * 7?",
  "solution": "105",
  "reasoning": "15 * 7 = (10 * 7) + (5 * 7) = 70 + 35 = 105",
  "priority_score": 0.85
}
```

These pairs are fed directly into the TRL (Transformer Reinforcement Learning) library for supervised fine-tuning.

#### How Repair mode works

```
1. Query the Failure Bank for high-novelty failures
   (failures that are semantically different from past failures)

2. For each failure:
   a. Call reference models (GPT-4o, Qwen-72B) to generate the correct solution
   b. Extract the reasoning from the reference model's response
   c. Validate the solution using Tier 1 verification (run it in the sandbox)

3. Assign repair_priority = novelty_score × failure_frequency
   (novel failures that happen often are highest priority)

4. Return the top-N pairs ordered by priority
```

#### Why use reference models?

The agent being trained might not know the correct answer, but GPT-4o and Qwen-72B (much larger, more capable models) do. We use them as "oracles" to generate ground-truth solutions.

#### Tier 1 validation

Before adding a synthesized pair to the training queue, we run the solution through the RestrictedPython sandbox (Tier 1 verification). This catches cases where the reference model hallucinated or generated invalid code.

### Implementation details

**`TaskGenerator` class** (`environment/task_generator.py`):
- `hunt_mode(agent_performance, discrimination_zone)` — generates one adversarial prompt
- `repair_mode(failure_bank, num_pairs)` — synthesizes N training pairs from failures
- `_select_strategy(agent_weakness_profile)` — picks the best adversarial strategy
- `_generate_prompt_for_strategy(strategy, task_pool)` — creates a prompt using the chosen strategy
- `_prompt_history` — set of recent prompts for deduplication
- `_strategy_effectiveness` — tracks which strategies work best (for future optimization)

**Adversarial strategies** are defined as constants:
```python
ADVERSARIAL_STRATEGIES = [
    "semantic_perturbation",
    "logical_contradiction",
    "context_confusion",
    "boundary_testing"
]
```

### Tests (6 unit tests, all passing)

| Test | What it checks |
|------|---------------|
| `test_hunt_mode_returns_required_keys` | Output has `prompt`, `difficulty_score`, `strategy`, `reasoning`, `cot_trace` |
| `test_hunt_mode_strategy_is_valid` | Strategy is one of the four defined strategies |
| `test_hunt_mode_prompt_is_non_empty` | Prompt is not an empty string |
| `test_hunt_mode_difficulty_score_in_range` | Difficulty score is 0-100 |
| `test_hunt_mode_uses_discrimination_zone` | When given a discrimination zone, it generates a relevant prompt |
| `test_hunt_mode_deduplicates_prompts` | Repeated calls don't return identical prompts |

---

## Task 6 — Implement Curriculum Manager

### What is the Curriculum Manager?

The Curriculum Manager is the component that decides **what difficulty level the agent should train on**. It's like a personal trainer who adjusts the weight on the barbell based on your current strength.

The key insight: training is most effective when tasks are **neither too easy nor too hard**. The Curriculum Manager computes the "discrimination zone" (20-80% success rate) and adjusts difficulty to keep the agent in that zone.

### The discrimination zone

```
For each task, compute success_rate over the last 100 episodes:

  if success_rate < 20%:
    category = "too_easy"  (agent struggles — needs easier tasks)
  
  elif 20% ≤ success_rate ≤ 80%:
    category = "discrimination_zone"  (optimal difficulty — keep training here)
  
  else:  # success_rate > 80%
    category = "too_hard"  (agent has mastered this — needs harder tasks)
```

The manager tracks `discrimination_zone_coverage` — the fraction of tasks in the optimal zone. If coverage drops below 30%, it escalates to harder tasks. If coverage exceeds 70%, it adds easier tasks.

### Co-evolutionary escalation

This is the most important concept in ALICE. **Neither the agent nor the benchmark can advance alone.** Both must improve together.

```
Agent improvement = (current_success_rate - baseline) / baseline
Benchmark improvement = (current_discrimination_coverage - baseline) / baseline

Escalation triggers if and only if:
  Agent improvement > 10% AND Benchmark improvement > 10%
```

Why? If only the agent improves, the benchmark is too easy — escalating would be premature. If only the benchmark improves (more tasks in the discrimination zone), the agent hasn't learned enough yet.

This creates a **coupled dynamical system** where both converge to higher capability together.

### Sliding window metrics

The manager tracks per-task metrics over a **sliding window of 100 episodes**:

```python
task_performance[task_id] = deque([1.0, 0.0, 1.0, ...], maxlen=100)
                                   ↑    ↑    ↑
                                success, fail, success
```

When the 101st episode completes, the oldest entry is automatically dropped. This ensures the manager reacts to recent performance, not ancient history.

Metrics tracked:
- `success_rate` — mean of the sliding window
- `attempt_count` — total times this task was attempted
- `average_reward` — mean reward over the window
- `last_attempted` — ISO timestamp of most recent attempt

### The curriculum heatmap

The heatmap is a 5×10 matrix visualizing task difficulty distribution:

```
Rows = 5 domains (arithmetic, logic, factual, symbolic, code)
Columns = 10 difficulty tiers (1 = easiest, 10 = hardest)
Cell value = success rate for that domain/tier combination
```

This is displayed in the Gradio dashboard (task 14) as a color-coded grid.

### Plateau detection

A plateau is when the agent stops improving. The manager detects this by checking:
```
if no escalation has occurred in the last 100 episodes:
  recommend_curriculum_adjustment()
```

This triggers a warning in the dashboard so the operator can manually intervene (e.g., add more diverse tasks).

### Manual override

For experimentation, the operator can manually set the difficulty tier:
```python
curriculum_manager.set_manual_override(difficulty_tier=5)
```

This bypasses the automatic escalation logic and forces the curriculum to a specific level. Useful for debugging or testing specific difficulty ranges.

### Implementation details

**`CurriculumManager` class** (`environment/curriculum_manager.py`):
- `compute_discrimination_zone(task_performance)` — categorizes tasks into too_easy/discrimination_zone/too_hard
- `should_escalate()` — returns `True` if both agent and benchmark improvement > 10%
- `escalate()` — increases `difficulty_tier` by 1 and resets improvement scores
- `get_curriculum_heatmap()` — returns a `(5, 10)` numpy array
- `update_task_performance(task_id, success)` — records a task outcome
- `get_task_success_rate(task_id)` — returns the sliding-window success rate
- `set_improvement_scores(agent_score, benchmark_score)` — updates the scores used for escalation
- `set_manual_override(tier)` — manually sets the difficulty tier
- `detect_plateau()` — returns `True` if no improvement in 100 episodes
- `_log_change(type, justification)` — logs all curriculum changes with timestamp

**Constants**:
```python
WINDOW_SIZE = 100                      # sliding window size
MIN_EPISODES_BETWEEN_ESCALATIONS = 50  # cooldown period
DISCRIMINATION_LOW = 0.2               # 20% success rate
DISCRIMINATION_HIGH = 0.8              # 80% success rate
ESCALATION_THRESHOLD = 0.1             # 10% improvement required
```

### Property 3: Discrimination Zone Non-Degeneracy

This property test verifies that **the curriculum always maintains a non-empty discrimination zone**. If it becomes empty (all tasks too easy or too hard), the manager must escalate or de-escalate within 10 episodes.

Tests:
- Empty performance → empty zone (edge case)
- Mixed performance → non-empty zone
- All tasks too easy → escalation fires
- Escalation resets improvement scores
- Minimum 50 episodes enforced between escalations

### Property 7: Co-evolutionary Coupling

This property test verifies that **escalation fires if and only if both agent AND benchmark improvement exceed 10%**. Neither condition alone is sufficient.

Test cases:
```python
(agent=0.15, benchmark=0.15) → escalate = True   # both exceed threshold
(agent=0.05, benchmark=0.15) → escalate = False  # agent below threshold
(agent=0.15, benchmark=0.05) → escalate = False  # benchmark below threshold
(agent=0.0,  benchmark=0.0)  → escalate = False  # both below
(agent=0.11, benchmark=0.11) → escalate = True   # both just above
(agent=0.10, benchmark=0.10) → escalate = False  # both at boundary (not strictly greater)
```

The test uses `@pytest.mark.parametrize` to run all 6 cases in a single test function.

### Tests (21 total, all passing)

| Test class | Count | What it covers |
|-----------|-------|---------------|
| `TestDiscriminationZoneNonDegeneracy` | 5 | Property 3 — non-empty discrimination zone |
| `TestCoEvolutionaryCoupling` | 7 | Property 7 — escalation requires both improvements |
| `TestCurriculumMetrics` | 9 | Sliding window, heatmap, plateau detection |

---

---

## Task 7 — Implement the Oracle Interface

### What is the Oracle Interface?

The Oracle Interface is the component that **calibrates task difficulty** using reference models (GPT-4o and Qwen-72B). It answers the question: "How hard is this task for a capable model?"

This is different from asking "how hard is this for the agent being trained." The oracle uses large, frontier models as a ground-truth difficulty signal, then classifies tasks as easy / medium / hard based on that signal. The curriculum manager uses these classifications to decide which difficulty tier the agent should train on.

### Why use two reference models?

A single model might have idiosyncratic biases — tasks it's unusually good or bad at due to training data distribution. By using two independent models (primary and secondary) and averaging their scores, the oracle gets a more robust difficulty estimate.

If the two models **disagree by more than 0.3** (e.g., primary scores 0.1, secondary scores 0.8), the task is **flagged for review**. A human operator can then inspect it — it might be ambiguous, or one model might have been confused by the phrasing.

### The calibration pipeline

```
For a given task string:

1. Compute task_hash = SHA-256(task)

2. Check cache for (task_hash, "primary")
   - Cache hit (< 30 days old) → use cached score
   - Cache miss → call primary reference model

3. Check cache for (task_hash, "secondary")
   - Same logic

4. reference_performance = (primary_score + secondary_score) / 2.0

5. Assign difficulty:
   - reference_performance < 0.4  → "easy"
   - reference_performance ≤ 0.7  → "medium"
   - reference_performance > 0.7  → "hard"

6. Flag for review if |primary - secondary| > 0.3

7. Return calibration result dict
```

### The caching system

API calls to frontier models are slow and expensive. The oracle uses an in-memory cache with a **30-day TTL** (time-to-live):

```python
_cache: Dict[Tuple[task_hash, model_key], {"score": float, "timestamp": datetime}]
```

When `calibrate_task()` is called:
- Cache hit → return stored score immediately (no API call)
- Cache miss → call model, store result with current timestamp
- Stale entry (> 30 days old) → treated as a miss (entry is deleted)

The cache hit rate is tracked and exposed via `get_cache_hit_rate()`. A rate below 50% triggers a warning — it means the oracle is calling the API more than expected, which could indicate the task pool is changing too fast or the cache is being cleared unnecessarily.

### How the reference model call works

The oracle formats a difficulty-rating prompt:

```
Rate how difficult this task is for an AI assistant on a scale of 0.0 to 1.0.
0.0 = trivially easy, 1.0 = extremely hard.

Task: {task[:400]}

Reply with ONLY a single decimal number between 0.0 and 1.0.
```

The model replies with a single number (e.g., `0.65`). The oracle parses the first token, clamps to [0.0, 1.0], and returns it. If the API call fails for any reason (network error, model error, malformed response), the oracle falls back to `0.5` (medium difficulty) and logs a warning.

### Implementation details

**`OracleInterface` class** (`environment/oracle_interface.py`):
- `calibrate_task(task)` — main entry point; returns full calibration result dict
- `get_cached_score(task_hash, model_name)` — returns cached score or `None` if missing/expired
- `invalidate_cache(task_hash)` — removes all cached entries for a task (e.g., after task is modified)
- `get_cache_hit_rate()` — returns `cache_hits / (cache_hits + misses)`, warns if below 50%
- `_get_or_fetch_score(task_hash, model_key, task)` — checks cache, falls back to model call
- `_call_reference_model(model_key, task)` — makes the actual API call via OpenAI-compatible client
- `_assign_difficulty(reference_performance)` — maps float → "easy" / "medium" / "hard"
- `_hash_task(task)` — SHA-256 hash for deterministic cache keys
- `_log_calibration(...)` — logs calibration events for observability

**Constants**:
```python
CACHE_TTL_DAYS = 30               # Cache entries valid for 30 days
DIVERGENCE_THRESHOLD = 0.3        # Flag if |primary - secondary| > 0.3
DIFFICULTY_EASY_MAX = 0.4         # reference_performance < 0.4 → easy
DIFFICULTY_HARD_MIN = 0.7         # reference_performance > 0.7 → hard
CACHE_HIT_RATE_WARNING = 0.5      # Warn if hit rate falls below 50%
```

**Environment variables** (for the API client):
```
API_BASE_URL                 # HF Inference API base (default: HF endpoint)
HF_TOKEN / OPENAI_API_KEY    # Authentication token
REFERENCE_MODEL_PRIMARY      # Primary model ID (default: TinyLlama in tests)
REFERENCE_MODEL_SECONDARY    # Secondary model ID
```

### What `calibrate_task()` returns

```python
{
    "gpt4o_score": 0.65,           # Primary model's difficulty score
    "qwen72b_score": 0.72,         # Secondary model's difficulty score
    "reference_performance": 0.685, # Average of the two
    "difficulty": "medium",         # "easy" / "medium" / "hard"
    "flagged_for_review": False,    # True if models disagree by > 0.3
}
```

### Tests (6 unit tests, Task 7.2, all passing)

| Test | What it checks |
|------|---------------|
| `test_cache_hit_avoids_api_call` | Second call to same task uses cache (0 new API calls) |
| `test_different_tasks_call_api` | Two different tasks each call both models (4 total) |
| `test_cache_hit_rate_computed_correctly` | Hit rate > 0 after a cached call |
| `test_difficulty_assigned_correctly` | score=0.2 → "easy"; score=0.8 → "hard" |
| `test_divergence_flagged` | Primary=0.1, secondary=0.8 → `flagged_for_review=True` |
| `test_cache_expiry_after_30_days` | Entry older than 30 days → `get_cached_score` returns `None` |

---

## Task 8 — Implement the Verifier Stack (Three-Tier Verification)

### What is the Verifier Stack?

The Verifier Stack is the component that **scores agent outputs**. It runs three verification tiers in sequence, producing a single composite score in [0.0, 1.0].

The three-tier design solves a fundamental tradeoff: programmatic checks are fast and tamper-proof but only work for code/math tasks; LLM judges are flexible but slow and expensive; regression batteries catch regressions that individual checks miss. Running all three gives the best coverage.

### The cascade logic

```
verify(agent_output, task):

  1. Tier 1 — Programmatic (RestrictedPython sandbox)
     if Tier 1 FAILS:
       composite = 0.0        ← absolute veto, skip everything else
       add to Failure Bank
       return immediately

  2. Tier 2 — Dual-model LLM Judge (primary + secondary models)
     tier2_score = average of both models' criterion scores
     if |primary_composite - secondary_composite| > 0.2: flagged = True

     if tier2_score < 0.5 OR episode_count % 100 == 0:
       run Tier 3 (regression battery — auto-triggered every 100 episodes)
     else:
       skip Tier 3 (tier3_score = 1.0)

  3. composite = 0.3×T1 + 0.4×T2 + 0.3×T3   (clamped to [0, 1])

  if composite < 0.5:
    add to Failure Bank
```

**Tier 1 is an absolute veto** — no amount of high Tier 2/3 scores can rescue an output that fails the sandbox. This prevents reward hacking via outputs that fool the LLM judge but are programmatically invalid.

### Tier 1 — RestrictedPython Sandbox

Tier 1 executes the agent's code inside RestrictedPython with hard constraints:

| Constraint | Implementation |
|-----------|---------------|
| Timeout | 5s — daemon thread, `t.join(timeout=5)` |
| Memory cap | 512 MB — `resource.setrlimit(RLIMIT_AS, ...)` in sandbox thread (best-effort, UNIX) |
| Blocked builtins | `open`, `exec`, `eval`, `__import__`, `compile`, `breakpoint` stripped from `safe_builtins` |
| stdout/stderr | Captured via `io.StringIO` redirect, returned in result dict |
| Hooks | `_getiter_=iter`, `_getattr_=getattr`, `_write_=identity`, `_inplacevar_` handles `+= -= *= /=` |

The sandbox runs code in a daemon thread. If the thread is still alive after 5 seconds, `TimeoutError` is returned. stdout and stderr are redirected before exec and restored in a `finally` block.

Return value: `{success, output, stdout, stderr, execution_time, error_message, error_type}`

### Tier 2 — Dual-Model LLM Judge with Output Cache

Tier 2 calls **two independent reference models** (primary and secondary) against the same four-criterion rubric, then averages their scores:

| Criterion | What it measures |
|-----------|-----------------|
| Correctness | Is the answer factually right? |
| Completeness | Does it fully address the task? |
| Clarity | Is the reasoning clear? |
| Efficiency | Is the solution concise? |

```
primary_scores  = {correctness, completeness, clarity, efficiency}  ← model 1
secondary_scores = {correctness, completeness, clarity, efficiency}  ← model 2
avg_scores = (primary + secondary) / 2
composite_score = mean(avg_scores)
flagged = |mean(primary) - mean(secondary)| > 0.2
```

**Output cache**: results are stored in `_t2_cache` keyed by `SHA-256(output + "|||" + task)`. A cache hit returns the stored result with no API call. This is critical because Tier 2 is the most expensive tier.

Return value: `{criterion_scores, primary_scores, secondary_scores, composite_score, reasoning, flagged_for_review}`

### Tier 3 — Regression Battery (762 tasks)

Tier 3 runs a battery of **762 held-out tasks** generated at import time by `_build_regression_battery()`. The battery covers:

| Category | Count | Examples |
|----------|-------|---------|
| Arithmetic (+, -, *, //, %) | 525 | `result = 7 * 6` → 42 |
| Boolean / comparison | 75 | `result = 3 < 5` → True |
| abs / pow / round | 28 | `result = abs(-7)` → 7 |
| String ops | 53 | `result = 'xx'.upper()` → 'XX' |
| List / range / comprehension | 70 | `result = sum(range(6))` → 15 |
| sorted / reversed / max / min | 8 | `result = sorted([5,2,8])` → [2,5,8] |
| set / dict | 7 | `result = len({1,2,2})` → 2 |

Each task re-uses the Tier 1 sandbox to execute and compares the output to the known expected value. If `pass_rate` drops > 5pp below baseline, a regression warning is logged.

**Auto-trigger**: Tier 3 runs automatically every 100 episodes (tracked via `_episode_count`) in addition to whenever Tier 2 scores < 0.5. This catches slow regressions that individual episode checks might miss.

**Manual trigger**: `set_regression_baseline()` runs the full battery and sets the current pass rate as the new baseline.

Return value: `{pass_rate, failed_tasks, performance_trend, tasks_run}`

### Composite scoring

```
composite = 0.3 × tier1_score + 0.4 × tier2_score + 0.3 × tier3_score
```

Tier 2 (LLM judge) carries the highest weight — it's the most holistic signal. Tier 1 and Tier 3 are equal.

- Tier 1 fails → composite = 0.0 (absolute veto, weights ignored)
- Tier 2 ≥ 0.5 and not at 100-episode mark → Tier 3 skipped, contribution defaults to 1.0

### Failure Bank integration

Any output with `composite < 0.5` (or Tier 1 failure) is routed to `_handle_failure()` → Failure Bank. The Failure Bank queues it for Repair mode synthesis.

### Implementation details

**`VerifierStack` class** (`environment/verifier_stack.py`):
- `verify(agent_output, task)` — runs the full cascade, increments `_episode_count`, returns all tier scores + composite
- `tier1_verify(code)` — sandbox with memory limit, stdout/stderr capture, timeout
- `tier2_verify(output, task)` — dual-model scoring, output cache, divergence flagging
- `tier3_verify(agent_output)` — 762-task regression battery, trend tracking
- `set_regression_baseline()` — manual trigger; runs battery and sets baseline
- `_call_judge_model(output, task, model_key)` — single model call, returns per-criterion dict
- `_parse_rubric_scores(raw)` — parses 4 space-separated floats from LLM response
- `_handle_failure(agent_output, result, task)` — routes to Failure Bank
- `_build_reasoning(t1, t2, t3, composite)` — human-readable score summary

**Constants**:
```python
TIER1_WEIGHT = 0.3
TIER2_WEIGHT = 0.4
TIER3_WEIGHT = 0.3
TIER2_THRESHOLD = 0.5              # below this, Tier 3 is triggered
COMPOSITE_PASS_THRESHOLD = 0.5
TIER2_DIVERGENCE_THRESHOLD = 0.2   # flag if models disagree by more than this
SANDBOX_TIMEOUT = 5                # seconds
SANDBOX_MEMORY_MB = 512
REGRESSION_BATTERY_SIZE = 500      # cap how many tasks tier3 runs (of 762 generated)
T3_AUTO_TRIGGER_INTERVAL = 100     # auto-run Tier 3 every N episodes
```

### Property 4: Tier 1 Override Lower Bound

**If Tier 1 passes, composite ≥ 0.3** (the T1 weight floor). Even if Tier 2 and Tier 3 both score 0.0, the minimum composite is `0.3 × 1.0 = 0.3`.

Test approach: monkey-patch `tier1_verify` to return success, patch `tier2_verify` and `tier3_verify` to return 0.0, assert `composite_score >= 0.3`.

### Property 5: Tier 1 Anti-Manipulation

**If Tier 1 fails, composite = 0.0 exactly**, regardless of Tier 2/3 scores. Tier 2 is never even called when Tier 1 fails.

Test approach: patch `tier1_verify` to fail, track whether `tier2_verify` was called (it should not be), assert `composite_score == 0.0` and `tier2_score is None`.

### Property 10: Sandbox Isolation

The sandbox blocks all dangerous operations and leaves process-level state unchanged after an attempted escape.

| Attempt | Result |
|---------|--------|
| `open('/etc/passwd', 'r')` | `success=False` |
| `eval('1+1')` | `success=False` |
| `exec('import os')` | `success=False` |
| `import os; os.getcwd()` | `success=False` (import blocked) |
| `while True: ...` (infinite loop) | `success=False`, `error_type=TimeoutError` |
| `os.chdir('/')` attempt | `success=False`, `os.getcwd()` unchanged after |
| `result = 2 + 2` | `success=True`, `output=4` |
| `result = [x*2 for x in range(5)]` | `success=True`, `output=[0,2,4,6,8]` |

### Tests (14 total, all passing)

| Test class | Count | Task |
|-----------|-------|------|
| `TestVerifierTier1Override` | 4 | 8.2 & 8.3 — P4 (lower bound) and P5 (anti-manipulation) |
| `TestSandboxIsolation` | 8 | 8.4 — P10 (sandbox isolation) |
| `TestRepairSurgicalMinimality` | 2 | P12 — regression drop ≤ 15pp after repair |

---

## Task 9 — Checkpoint: All Tests Pass

### Purpose

Task 9 is a verification gate. Before moving on to implementing the Failure Bank (task 10), all tests accumulated so far must pass cleanly. This catches any regressions introduced by tasks 7 or 8 before they compound.

### What was verified

The full test suite was run across all three test files:

```
uv run pytest tests/ -v
```

### Result

```
130 passed in 67s
```

| Test file | Tests | Notes |
|-----------|-------|-------|
| `tests/test_server.py` | 22 | Health, reset, state, step endpoints |
| `tests/test_properties.py` | 88 | P1–P12 properties + unit tests for tasks 3–8 |
| `tests/test_integration.py` | 20 | End-to-end episode cycle, hunt → verify → repair pipeline |

No failures. No skips. No warnings about test logic.

### Why this checkpoint mattered

Task 8 was initially marked complete prematurely. An audit revealed 5 gaps against the spec:

| Gap | Before task 9 audit | After fix |
|-----|---------------------|-----------|
| Regression battery size | 20 tasks | **762 tasks** (500+ spec requirement) |
| Tier 2 divergence flagging | `flagged = False` hardcoded | Dual model calls, real divergence check |
| Tier 2 output caching | Missing | SHA-256 keyed `_t2_cache` |
| Tier 1 stdout/stderr capture | Missing | `io.StringIO` redirect, returned in result |
| Tier 1 memory enforcement | Constant unused | `resource.setrlimit(RLIMIT_AS)` in sandbox thread |

All 5 gaps were fixed before the checkpoint was signed off.

---

## Summary (updated)

| Task | Status | Tests |
|------|--------|-------|
| 1. Scaffold project | ✅ Done | — |
| 2.1 FastAPI server | ✅ Done | 22 passing |
| 2.2 Server unit tests | ✅ Done | 22 passing |
| 2.3 HF Spaces deployment | ✅ Done | 5/5 smoke tests |
| 3.1 MDPState dataclass | ✅ Done | 6 passing |
| 3.2 Markov property test | ✅ Done | 2 passing |
| 4.1 EpisodeHandler | ✅ Done | 7 passing |
| 4.2 Episode termination test | ✅ Done | 7 passing |
| 5.1 TaskGenerator (Hunt/Repair) | ✅ Done | 6 passing |
| 5.2 Repair mode | ✅ Done | (included in 5.1) |
| 5.3 Hunt mode unit tests | ✅ Done | 6 passing |
| 6.1 CurriculumManager | ✅ Done | 21 passing |
| 6.2 Discrimination zone test | ✅ Done | 5 passing |
| 6.3 Co-evolutionary coupling test | ✅ Done | 7 passing |
| 7.1 OracleInterface | ✅ Done | — |
| 7.2 Oracle caching unit tests | ✅ Done | 6 passing |
| 8.1 VerifierStack | ✅ Done | — |
| 8.2 Tier 1 anti-manipulation (P5) | ✅ Done | 4 passing |
| 8.3 Tier 1 override lower bound (P4) | ✅ Done | (included in 8.2) |
| 8.4 Sandbox isolation (P10) | ✅ Done | 8 passing |
| 9. Checkpoint | ✅ Done | 130/130 passing |
| 10.1 FailureBank | ✅ Done | — |
| 10.2 Novelty monotonicity (P6) | ✅ Done | 3 passing |
| 11.1 RewardFunction | ✅ Done | — |
| 11.2 Reward boundedness (P2) | ✅ Done | 4 passing |
| 11.3 Attempt decay monotonicity (P8) | ✅ Done | 2 passing |
| 12.1 GRPOTrainer | ✅ Done | — |
| 12.2 Advantage zero-mean (P9) | ✅ Done | 6 passing |

Live environment: `https://rohanjain1648-alice-rl-environment.hf.space`

---

## Task 12 — Implement the GRPO Trainer

### What is GRPO?

GRPO (Group Relative Policy Optimisation) is the training algorithm introduced in DeepSeek-R1. It is a simplification of PPO that **eliminates the separate value model** — instead of learning V(s) to compute advantages, it uses group-level reward statistics.

For each update step, G=8 rollouts are collected from the current policy. The advantage of each rollout is its reward normalised against the group:

```
A_i = (R_i − μ_G) / σ_G
```

Because the advantage is always zero-mean by construction, no value baseline is needed. This halves the memory footprint vs PPO and is stable for LLM fine-tuning.

### The GRPO objective

```
L_GRPO  = E[ min( r_t × A_t,   clip(r_t, 1−ε, 1+ε) × A_t ) ]

where r_t = π_θ(a_t|s_t) / π_ref(a_t|s_t)   (policy ratio)
      ε   = 0.2                                (clip coefficient)

L_total = L_GRPO  −  β × KL(π_θ ‖ π_ref)

where β = 0.01   (KL penalty coefficient)
```

The clipped surrogate prevents excessively large policy updates (same mechanism as PPO). The KL penalty further anchors the policy to the reference, preventing the mode-collapse failure mode where the model degenerates to short repetitive outputs.

### Zero-variance noise injection

When all G rollouts have the same reward (e.g. all succeed or all fail), `σ_G ≈ 0` and all advantages collapse to zero — killing the gradient signal. To prevent this:

1. In `_compute_advantages`: if `σ < 1e-6`, add `N(0, 1e-4)` noise to rewards before normalising
2. In `_grpo_update`: if the advantage array still has `std < 1e-6` (can happen after noise), add another noise pass before computing loss terms

This ensures the trainer always has a non-degenerate gradient.

### Learning rate schedule

```
if KL > 0.1  and not already reduced:
    LR = LR × 0.5          ← halve to prevent collapse
    _lr_reduced = True

elif KL ≤ 0.1 and _lr_reduced:
    LR = _base_lr           ← restore to original once KL recovers
    _lr_reduced = False
```

The `_base_lr` is stored at init so restoration is exact.

### Metrics tracking

Every episode appends to `self._metrics`:

```python
{
    "episode":           int,
    "loss":              float,   # L_total
    "policy_divergence": float,   # KL(π_θ ‖ π_ref)
    "reward_mean":       float,   # μ_G for this update step
    "reward_std":        float,   # σ_G for this update step
}
```

Retrieved via `trainer.get_metrics()`.

### Checkpoint saving

Every `checkpoint_interval=100` episodes, `save_checkpoint()` calls `huggingface_hub.HfApi.upload_folder()` to push the model to `HF_REPO_ID`. Falls back gracefully (logs warning) when `HF_REPO_ID` is unset or the push fails.

### Implementation details

**`GRPOTrainer` class** (`training/train.py`):
- `train(num_episodes)` — main loop: collect → compute advantages → update → log metrics → checkpoint
- `load_model()` — loads `model`, `tokenizer`, and `ref_model` from HF Hub
- `_collect_rollouts()` — fires `G` episodes against the OpenEnv server, returns rewards
- `_compute_advantages(rollouts)` — group normalisation with zero-variance noise injection
- `_grpo_update(rollouts, advantages)` — clipped PPO objective + β×KL penalty, returns scalar loss
- `_compute_kl_divergence()` — estimates KL from log-prob difference; returns 0.0 when model not loaded
- `_sample_action(state)` — generates action via the loaded model (falls back to placeholder)
- `get_metrics()` — returns full per-episode metrics history
- `save_checkpoint()` — pushes model to HF Hub via `huggingface_hub`

**Constants**:
```python
LEARNING_RATE       = 1e-5    # default, configurable via ALICE_LR
GAMMA               = 0.99    # discount factor, configurable via ALICE_GAMMA (range [0.9, 1.0])
GRPO_GROUP_SIZE     = 8       # rollouts per update step
KL_THRESHOLD        = 0.1     # triggers LR halving
KL_BETA             = 0.01    # KL penalty coefficient in objective
CLIP_EPSILON        = 0.2     # PPO clip ratio
CHECKPOINT_INTERVAL = 100     # episodes between checkpoint pushes
```

### Property 9: GRPO Advantage Zero-Mean

**Group-normalised advantages always have mean ≈ 0 and std ≈ 1.**

Tested via both the standalone formula (P9 class) and through the actual `GRPOTrainer._compute_advantages` method:
- 8 distinct rewards → mean < 1e-5, std within 1e-5 of 1.0
- 100 random groups → each has mean < 1e-5
- Identical rewards → degenerate case detected (std ≈ 0 or noise ensures non-collapse)

### Tests (6 total, all passing)

| Test class | Count | Task |
|-----------|-------|------|
| `TestGRPOAdvantageZeroMean` | 4 | P9 standalone (zero-mean, unit-std, 100 random groups, zero-variance handling) |
| `TestGRPOTrainer` | 2 | 12.2 — P9 via trainer (`_compute_advantages` zero-mean and unit-std) |

---

## Task 11 — Implement the Reward Function (Bellman-shaped)

### What is the Reward Function?

The Reward Function translates verification results into scalar rewards for GRPO training. It implements **potential-based reward shaping** — a technique from RL theory that guarantees policy-invariance while providing richer gradient signal.

Without shaping, the agent only learns from the final episode outcome. With shaping, it gets a reward signal at every turn that reflects how much the curriculum improved, giving the trainer a much smoother learning surface.

### Per-turn reward formulas

Each turn has a different formula reflecting the escalating scaffolding structure:

| Turn | Raw reward formula | What it captures |
|------|--------------------|-----------------|
| 1 | `R₁ = R_programmatic` | Was the first attempt correct? Binary. |
| 2 | `R₂ = λ_judge × R_judge − attempt_decay` | Did the CoT retry improve with LLM judge feedback? Penalised for being a second attempt. |
| 3 | `R₃ = R_programmatic × R_regression × (1 − 2×attempt_decay) − repetition_penalty × I(a == a_prev)` | Did the final attempt pass both sandbox and regression? Harder penalty for repeated content. |

After each turn: `raw -= novelty_penalty` if the task is in the Failure Bank (already a known weakness).

Non-zero guarantee: if `raw == 0.0` after all penalties, it is set to `MIN_BASE_REWARD = 0.01`. This prevents degenerate all-zero episodes that would give the GRPO trainer zero-variance reward groups and kill gradient signal.

### Potential-based shaping

```
shaped_reward = raw + γ × Φ(s') − Φ(s)

where Φ(s) = discrimination_coverage(s)
      γ = 0.99 (discount factor)
      s = state before action
      s' = state after action
```

`discrimination_coverage` is the fraction of tasks in the optimal difficulty zone (20–80% success rate). If the agent's action moved more tasks into the discrimination zone, it gets a positive shaping bonus. If the zone shrank, it gets a penalty.

This is a valid potential-based shaping function under the Ng et al. (1999) theorem, meaning it does not change the optimal policy — it only speeds up convergence.

### Cumulative reward

```
cumulative = Σ shaped_rewards, clamped to [-1.0, 1.0]
```

Each individual shaped reward is also clamped to `[-1.0, 1.0]` before summation.

### Weight configuration

The three penalty weights can be updated at runtime:

```python
reward_fn.set_weights({
    "attempt_decay": 0.1,       # per-turn penalty for being a retry
    "novelty_penalty": 0.05,    # penalty for tasks already in the failure bank
    "repetition_penalty": 0.02, # penalty for repeating the same action
})
```

`set_weights()` validates that all values are in `[0, 1]`, raises `ValueError` if not, and logs the change with a timestamp for auditing.

### Implementation details

**`RewardFunction` class** (`environment/reward_function.py`):
- `compute_reward(episode_data)` — iterates turns, computes raw + shaped rewards, accumulates
- `set_weights(weights)` — validates, updates, logs weight changes
- `_compute_turn_reward(turn_data)` — dispatches to per-turn formula, applies penalties, enforces non-zero

**Constants**:
```python
GAMMA = 0.99          # discount factor for potential shaping
LAMBDA_JUDGE = 0.8    # weight of LLM judge score in Turn 2
MIN_BASE_REWARD = 0.01 # floor preventing degenerate zero-reward turns
```

### Property 2: Reward Boundedness

**All per-turn shaped rewards satisfy |R| ≤ 1.0.** Tested over 1000 random episodes with randomised tier scores, turn numbers, and in-bank flags — all 1000 pass.

### Property 8: Attempt Decay Monotonicity

**For identical verification outcomes, Turn 1 reward ≥ Turn 2 reward.** The attempt_decay term `0.1 × (turn_number − 1)` grows with each retry, so the same correct answer is rewarded less on later turns. This incentivises the agent to get things right first time.

### Tests (10 total, all passing)

| Test class | Count | Task |
|-----------|-------|------|
| `TestRewardBoundedness` | 4 | 11.2 — P2 (reward boundedness, 1000 random episodes) |
| `TestAttemptDecayMonotonicity` | 2 | 11.3 — P8 (decay monotonicity) |
| `TestRewardFunctionProperties` | 4 | 11.1 — weight validation, shaping correctness, bounds |

---

## Task 10 — Implement the Failure Bank

### What is the Failure Bank?

The Failure Bank is a **novelty-indexed store of failed tasks**. Every time the agent produces an output that fails verification (composite < 0.5 or Tier 1 failure), the VerifierStack routes it here. The Failure Bank then makes the highest-priority failures available for Repair mode synthesis in the Task Generator.

Think of it as the agent's mistake journal — organized not chronologically but by how *new* each mistake is.

### Novelty scoring

Every failure gets a `novelty_score ∈ [0, 1]` computed via sentence-transformer embeddings and k-NN cosine similarity:

```
embedding = SentenceTransformer("all-MiniLM-L6-v2").encode(prompt)   # 384-dim, zero-padded to 768

similarities = cosine_similarity(embedding, all_existing_embeddings)
top_k_sim = mean(top-5 similarities)

if top_k_sim > 0.8:   novelty = 0.1   ← near-duplicate, don't flood queue
elif top_k_sim < 0.5: novelty = 0.9   ← genuinely novel, high priority
else:                 novelty = 1.0 - top_k_sim
```

If the bank is empty (first failure ever), novelty = 1.0.

Fallback: when sentence-transformers is unavailable, a deterministic hash-based random embedding is used (reproducible given the same prompt text).

### Duplicate detection and frequency tracking

When a new failure has `novelty_score ≤ 0.1` (near-duplicate), the bank does not create a new entry. Instead it:
1. Finds the nearest existing entry via cosine similarity
2. Increments that entry's `failure_frequency` counter
3. Recomputes `repair_priority = novelty_score × failure_frequency`

This means **a task the agent keeps failing rises in repair priority** even though it isn't novel. The same bug showing up 10 times should be fixed before a single exotic edge case.

### Repair queue and priority

Only failures with `novelty_score ≥ 0.7` are added to the repair queue (high-novelty threshold). Repair priority is:

```
repair_priority = novelty_score × failure_frequency
```

`get_repair_candidates(num_pairs)` returns the top-N entries from the repair queue sorted by priority descending. These go directly to Task Generator `repair_mode()`.

### Data model — `FailureBankEntry`

```python
@dataclass-style class FailureBankEntry:
    failure_id: str          # UUID
    timestamp: str           # ISO 8601 UTC
    agent_version: str       # semver of the agent that produced this failure
    error_type: str          # "verification_failure", "timeout", etc.
    prompt: str              # the task prompt
    expected_output: str     # ground-truth answer (if known)
    actual_output: str       # what the agent produced
    cot_trace: str           # agent's chain-of-thought reasoning
    trajectory: Any          # full episode trajectory (serializable)
    novelty_score: float     # 0.0–1.0
    semantic_embedding: np.ndarray  # 768-dim embedding
    repair_priority: float   # novelty_score × failure_frequency
    repair_status: str       # "pending" | "in_progress" | "repaired" | "skipped"
    failure_frequency: int   # how many times this failure has been seen
```

### Archiving

When the bank exceeds `MAX_ENTRIES = 10,000`, the oldest entry is evicted to `_archived` (a plain list). The entry is also removed from the repair queue. The archive is kept in memory for dashboard display but excluded from novelty scoring and repair candidate queries.

### Implementation details

**`FailureBank` class** (`environment/failure_bank.py`):
- `add_failure(failure)` — computes embedding + novelty; deduplicates by incrementing frequency on near-duplicates instead of inserting
- `compute_novelty_score(failure)` — k-NN cosine similarity against all existing entries
- `get_repair_candidates(num_pairs)` — returns top-N by `repair_priority` from the queue
- `query_failures(error_type, agent_version, time_range, novelty_threshold)` — multi-criteria filter including ISO timestamp range
- `update_repair_status(failure_id, status)` — marks an entry's repair lifecycle state
- `get_failure_distribution()` — returns `{total, by_error_type, repair_queue_size, archived}`
- `_find_nearest_entry(embedding)` — returns the failure_id of the most cosine-similar existing entry
- `_compute_embedding(text)` — sentence-transformer encode, zero-padded to 768, with hash-based fallback
- `_update_repair_queue(entry)` — adds to queue if `novelty_score ≥ 0.7`
- `_cosine_similarity_batch(vec, matrix)` — vectorised cosine similarity

**Constants**:
```python
MAX_ENTRIES = 10_000
NOVELTY_DUPLICATE_THRESHOLD = 0.8   # similarity > 0.8 → duplicate
NOVELTY_NOVEL_THRESHOLD = 0.5       # similarity < 0.5 → novel
REPAIR_QUEUE_NOVELTY_MIN = 0.7      # minimum novelty to enter repair queue
KNN_K = 5                            # neighbours used in novelty scoring
```

### Property 6: Novelty Monotonicity

**Adding similar failures never increases the novelty score of subsequent similar failures.**

Formally: for a sequence of semantically similar prompts p₁, p₂, p₃, the novelty scores n₁ ≥ n₂ ≥ n₃ (non-increasing), with tolerance of 0.1 per step.

Test approach:
1. Compute the base novelty of a novel embedding against an empty bank → 1.0
2. Inject 3 near-identical entries directly (bypassing sentence-transformer)
3. Compute novelty of the same embedding again → ≤ 0.15 (duplicate threshold)
4. For a sequence of slightly perturbed embeddings added one-by-one, assert each successive novelty score is ≤ previous + 0.1

### Tests (3 unit tests, Task 10.2, all passing)

| Test | What it checks |
|------|---------------|
| `test_first_failure_is_maximally_novel` | Empty bank → novelty = 1.0 |
| `test_duplicate_failure_has_low_novelty` | 3 identical embeddings injected → next novelty ≤ 0.15 |
| `test_novelty_score_non_increasing_for_similar_failures` | Successive similar embeddings produce non-increasing scores (±0.1 tolerance) |

---

## Task 13 — Implement Anti-Hacking Monitors

### Why anti-hacking monitors?

GRPO training can converge to degenerate strategies — the agent discovers ways to maximise reward that don't reflect real task competence. Two classic failure modes:

1. **Mode collapse / exploration collapse**: the agent keeps producing the same output regardless of the task, because that output happened to get a high reward early in training
2. **Policy collapse**: the entropy of the token distribution decays monotonically — the model becomes overconfident and stops exploring

The three monitors intercept these failure modes before they corrupt the training run.

### Task 13.1 — TrajectorySampler (`monitors/trajectory_sampler.py`)

The sampler randomly inspects 5% of trajectories (configurable via `sample_rate`). For each sampled trajectory it computes:

| Signal | How it's computed | Threshold |
|--------|------------------|-----------|
| `trajectory_entropy` | Shannon entropy of action distribution, normalised to [0,1] | Flag `mode_collapse` if < 0.5 |
| `repetition_rate` | Fraction of turns with the most-common action | Flag `output_repetition` if > 50% |
| Reward hacking | All rewards ≥ 0.9 with no apparent failure | Flag `reward_hacking` |

The `anomaly_score ∈ [0, 1]` is the maximum of all per-signal scores. If `anomaly_score ≥ 0.7`, the trajectory is flagged and logged as an incident with `trajectory_id, anomaly_type, anomaly_score, timestamp`.

`get_anomaly_rate()` tracks `anomalies / sampled_trajectories`. If it exceeds 10%, a system-level warning is logged.

### Task 13.2 — EntropyMonitor (`monitors/entropy_monitor.py`)

The entropy monitor tracks the **policy entropy** at every training step over a sliding window of 100 episodes.

**Collapse detection:**
```
baseline = entropy at the start of the window (oldest entry)
current  = entropy at the end of the window (most recent)
decrease = (baseline − current) / baseline

if decrease > 20%:
    flag potential_collapse = True
    halve the learning rate
```

When entropy recovers (collapse flag clears), the learning rate is restored to its original value via `_baseline_lr`.

**Action diversity**: at each step, the fraction of tokens with probability > `1/(10 × vocab_size)` is computed. If it falls below 0.3, a `diversity_alert` is logged.

Metrics are logged every 10 steps to `_metrics_log` for dashboard display.

### Task 13.3 — Sandbox restriction enforcement tests

A dedicated `TestSandboxRestrictionEnforcement` class verifies all blocked builtins and safety limits:

| Test | What it checks |
|------|---------------|
| `test_open_raises_exception` | `open('/etc/passwd')` → `success=False` |
| `test_exec_raises_exception` | `exec('x=1')` → `success=False` |
| `test_eval_raises_exception` | `eval('1+1')` → `success=False` |
| `test_import_raises_exception` | `import os` → `success=False` |
| `test_compile_raises_exception` | `compile('x=1', ...)` → `success=False` |
| `test_breakpoint_is_blocked` | `breakpoint()` → `success=False` |
| `test_timeout_configuration` | `SANDBOX_TIMEOUT == 5` |
| `test_timeout_enforced_for_infinite_loop` | `while True: pass` → timed out |
| `test_memory_limit_configured` | `SANDBOX_MEMORY_MB == 512` |
| `test_safe_code_still_passes` | `result = 7 * 6` → `success=True`, `output=42` |

### Tests (25 total, all passing)

| Test class | Count | What it covers |
|-----------|-------|---------------|
| `TestSandboxRestrictionEnforcement` | 10 | 13.3 — all blocked builtins + timeout + memory config |
| `TestTrajectorySampler` | 8 | 13.1 — repetition detection, entropy, reward hacking, anomaly rate |
| `TestEntropyMonitor` | 7 | 13.2 — entropy computation, collapse detection, LR reduction, diversity |

---

## Task 14 — Implement Gradio Dashboard

### What is the dashboard?

The Gradio dashboard is a **live monitoring UI** that runs in the same process as the FastAPI server. Both are mounted into a single uvicorn app in `alice_server.py` via `gr.mount_gradio_app(api, gradio_app, path="/")`. This means one port (7860) serves both the API and the visual dashboard.

### Six display panels

| Panel | What it shows | Update mechanism |
|-------|--------------|-----------------|
| **Curriculum Heatmap** | 5-domain × 5-tier success rate matrix | Refreshes every 3s via `gr.Timer` |
| **Discrimination Zone Coverage** | Time series of zone coverage fraction | Appended on each refresh |
| **Reward Distribution** | Histogram of last 200 episode rewards | Appended on each refresh |
| **Recent Episodes** | Table of last 10 episodes (id, task, difficulty, time) | Populated from `_episode_history` |
| **Current Episode State** | Active episode ID, turn number, task text | Polled from `/state` |
| **Alerts** | Active warnings (high error rate, high latency) | Polled from `/health` |

### Dashboard architecture (`alice_server.py`)

The production entry point combines the FastAPI API and Gradio UI:

```python
gradio_app = build_gradio()
app = gr.mount_gradio_app(api, gradio_app, path="/")
```

The `refresh_dashboard()` function is called every 3 seconds by a `gr.Timer`. It:
1. Polls `/health` and `/state` (loopback HTTP calls)
2. Updates the discrimination history and reward history buffers
3. Rebuilds all matplotlib figures
4. Returns updated values to all Gradio components

### Export and pause

- **Export State JSON**: dumps current `/state`, `/health`, episode count, and timestamp to JSON, displayed in the state text box
- **Pause** is not a button in the final `alice_server.py` implementation (the standalone `dashboard/gradio_app.py` has it) — the server dashboard always refreshes since it's production-facing

### Implementation details

Two dashboard implementations exist:
- **`dashboard/gradio_app.py`** — standalone dashboard that connects to a remote `ALICE_ENV_URL` via HTTP; has pause/resume, full failure table browser, CSV export
- **`alice_server.py`** (Gradio embedded section) — same panels but runs in-process for HF Spaces; uses shared `_episode_history` buffer instead of remote polling

The standalone `gradio_app.py` is used for local development; the embedded version is what runs in production on HF Spaces.

---

## Task 15 — Checkpoint: All Tests Pass (post-tasks 13/14)

All 155 tests pass:

| Test file | Tests |
|-----------|-------|
| `tests/test_server.py` | 22 |
| `tests/test_properties.py` | 113 |
| `tests/test_integration.py` | 20 |

---

## Task 16 — HF Spaces Production Hardening

### Task 16.1 — Multi-stage Dockerfile

The Dockerfile was upgraded to a two-stage build to minimise the final image size:

```
Stage 1 (builder):
  - python:3.11-slim
  - install uv
  - copy pyproject.toml
  - uv pip install all server deps into /app/.venv

Stage 2 (runtime):
  - python:3.11-slim  (fresh base — no build tools)
  - copy uv binary
  - copy /app/.venv from builder  (pre-compiled wheels, no gcc)
  - copy application code
  - mkdir /app/data/failure_bank /app/data/trajectories /app/logs
  - ENV PATH="/app/.venv/bin:$PATH"
  - CMD ["python", "alice_server.py"]
```

The key saving: the builder stage installs packages from scratch (downloading, compiling if needed), but only the pre-compiled `.venv` directory is copied to the runtime stage. Compiler toolchains and intermediate build files never reach the final image.

### Task 16.2 — README.md YAML front-matter

```yaml
---
title: ALICE RL Environment
sdk: docker
app_port: 7860
hardware: t4-small
secrets:
  - OPENAI_API_KEY
  - HF_TOKEN
  - REFERENCE_MODEL_PRIMARY
  - REFERENCE_MODEL_SECONDARY
---
```

- `hardware: t4-small` — requests a T4 GPU for the Space (handles inference workload from VerifierStack and OracleInterface)
- `secrets` — the four secrets are mounted as environment variables at runtime; `HF_TOKEN` is used by the oracle for Inference API calls; `OPENAI_API_KEY` is the fallback

### Task 16.3 — deploy_spaces.sh with rollback

The deploy script now supports two modes:

**Normal deploy:**
```bash
HF_SPACE_ID=username/alice-rl-environment HF_TOKEN=hf_... bash scripts/deploy_spaces.sh
```
Before pushing, it captures the current HEAD SHA of the Space repo and prints:
```
Previous HEAD SHA: abc123... (use ROLLBACK_SHA=abc123... to roll back)
```

**Rollback:**
```bash
ROLLBACK_SHA=abc123... HF_SPACE_ID=username/... HF_TOKEN=hf_... bash scripts/deploy_spaces.sh --rollback
```
Downloads the specified revision to a temp directory, re-uploads it to the Space, then cleans up the temp dir.

Both modes poll the health endpoint and verify all three components (server, repository, registry) after the push.

---

## Task 17 — HF Jobs Training Pipeline

### Task 17.1 — `scripts/launch_training.sh`

```bash
# Pull latest checkpoint from HF Hub
huggingface-cli download "$HF_REPO_ID" --local-dir ./checkpoints

# Submit training job to HF Jobs targeting T4 GPU
huggingface-cli jobs run \
    --gpu t4-medium \
    --env "ALICE_MODEL_ID=$MODEL_ID" \
    --env "ALICE_ENV_URL=$ENV_URL" \
    --env "ALICE_HF_REPO_ID=$HF_REPO_ID" \
    --env "HF_TOKEN=$HF_TOKEN" \
    "uv run python training/train.py"
```

The trainer (`training/train.py`) reads `ALICE_ENV_URL` to connect to the environment server, runs GRPO, and calls `save_checkpoint()` every 100 episodes to push back to `ALICE_HF_REPO_ID`.

### Task 17.2 — `pyproject.toml` with all dependencies

All required packages are declared:

```toml
dependencies = [
    "openenv-core[core]>=0.2.2",
    "numpy>=1.26.0",
    "openai>=1.0.0",
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "trl>=0.8.0",
    "sentence-transformers>=2.0.0",
    "RestrictedPython>=7.0",
    "gradio>=4.0.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.20.0",
    ...
]
```

uv scripts configured:
```toml
[tool.uv.scripts]
serve     = "uvicorn alice_server:app --host 0.0.0.0 --port 7860"
train     = "python training/train.py"
dashboard = "python dashboard/gradio_app.py"
```

---

## Task 18 — Integration Tests

### What was tested

`tests/test_integration.py` wires all components together without mocks (except where external API calls are replaced with stubs) and runs end-to-end pipelines.

| Test class | What it covers |
|-----------|---------------|
| `TestFullEpisodeCycle` | HTTP reset → 3 steps → done=True; each step returns valid state/reward/done |
| `TestHuntVerifyFailurePipeline` | Hunt mode → VerifierStack → FailureBank; failing output lands in bank |
| `TestRepairModeIntegration` | Failure bank → repair_mode → training pairs with required keys |
| `TestCurriculumEscalationIntegration` | Set both improvement scores > 0.1 → escalate() fires |
| `TestDashboardDataRefresh` | `/state` and `/health` return parseable JSON; dashboard can build figures |

**20 integration tests, all passing.**

### Property 12: Repair Surgical Minimality

After applying repair pairs from the bank, the regression battery pass rate must not drop by more than 15 percentage points. This ensures repair-mode training pairs don't introduce regressions on held-out tasks.

---

## Task 19 — Final Checkpoint

All 155 tests pass:

```
.......................................................................
.......................................................................
...........
155 passed in ~90s
```

| Component | Status |
|-----------|--------|
| FastAPI server (22 tests) | ✅ |
| Property + unit tests (113 tests) | ✅ |
| Integration tests (20 tests) | ✅ |
| Dockerfile (multi-stage) | ✅ |
| HF Spaces config (secrets + hardware) | ✅ |
| deploy_spaces.sh (rollback support) | ✅ |
| launch_training.sh (HF Jobs) | ✅ |
| Gradio dashboard | ✅ |

---

## Summary (final)

| Task | Status | Tests |
|------|--------|-------|
| 1. Scaffold | ✅ | — |
| 2.1–2.3 Server + HF Spaces | ✅ | 22 passing |
| 3.1–3.2 MDPState | ✅ | 8 passing |
| 4.1–4.2 EpisodeHandler | ✅ | 7 passing |
| 5.1–5.3 TaskGenerator | ✅ | 13 passing |
| 6.1–6.3 CurriculumManager | ✅ | 21 passing |
| 7.1–7.2 OracleInterface | ✅ | 6 passing |
| 8.1–8.4 VerifierStack | ✅ | 12 passing |
| 9. Checkpoint | ✅ | 130/130 |
| 10.1–10.2 FailureBank | ✅ | 3 passing |
| 11.1–11.3 RewardFunction | ✅ | 10 passing |
| 12.1–12.2 GRPOTrainer | ✅ | 6 passing |
| 13.1–13.3 Anti-Hacking Monitors | ✅ | 25 passing |
| 14.1 Gradio Dashboard | ✅ | — |
| 15. Checkpoint | ✅ | 155/155 |
| 16.1–16.3 HF Spaces hardening | ✅ | — |
| 17.1–17.2 HF Jobs pipeline | ✅ | — |
| 18.1–18.2 Integration tests | ✅ | 20 passing |
| 19. Final checkpoint | ✅ | **155/155** |

Live environment: `https://rohanjain1648-alice-rl-environment.hf.space`
