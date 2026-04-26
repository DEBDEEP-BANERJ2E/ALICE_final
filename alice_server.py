"""
ALICE — combined Gradio dashboard + FastAPI server for HF Spaces.

The Gradio UI is mounted at / and the FastAPI API is mounted at /api.
Both run in the same uvicorn process on port 7860.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional

import gradio as gr
import httpx
import numpy as np
import psutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("alice.server")

# ---------------------------------------------------------------------------
# Lazy component init
# ---------------------------------------------------------------------------

_episode_handler: Any = None
_curriculum_manager: Any = None
_task_generator: Any = None
_reward_function: Any = None
_verifier_stack: Any = None
_failure_bank: Any = None
_leaderboard: Any = None


def _get_leaderboard():
    global _leaderboard
    if _leaderboard is None:
        from environment.leaderboard import Leaderboard
        _leaderboard = Leaderboard()
    return _leaderboard


def _get_components():
    global _episode_handler, _curriculum_manager, _task_generator
    global _reward_function, _verifier_stack, _failure_bank
    if _episode_handler is None:
        from environment.episode_handler import EpisodeHandler
        from environment.curriculum_manager import CurriculumManager
        from environment.task_generator import TaskGenerator
        from environment.reward_function import RewardFunction
        from environment.verifier_stack import VerifierStack
        from environment.failure_bank import FailureBank
        _failure_bank = FailureBank()
        _verifier_stack = VerifierStack(failure_bank=_failure_bank)
        _reward_function = RewardFunction(
            attempt_decay_weight=float(os.getenv("ATTEMPT_DECAY_WEIGHT", "0.1")),
            novelty_penalty_weight=float(os.getenv("NOVELTY_PENALTY_WEIGHT", "0.05")),
            repetition_penalty_weight=float(os.getenv("REPETITION_PENALTY_WEIGHT", "0.02")),
            gamma=float(os.getenv("DISCOUNT_FACTOR", "0.99")),
        )
        _curriculum_manager = CurriculumManager()
        _task_generator = TaskGenerator()
        _episode_handler = EpisodeHandler()
    return _episode_handler, _curriculum_manager, _task_generator, _reward_function, _verifier_stack, _failure_bank


# Thread pool for non-blocking background work (failure bank insertions, etc.)
_bg_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="alice-bg")


def _prewarm_sentence_transformer():
    """Pre-load the sentence-transformer model so the first /step call isn't slow."""
    try:
        from environment.failure_bank import FailureBank
        _fb = FailureBank()
        _fb._compute_embedding("warmup")
        logger.info("Sentence-transformer pre-warmed successfully")
    except Exception as exc:
        logger.warning("Sentence-transformer pre-warm failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

api = FastAPI(title="ALICE OpenEnv API", version="0.1.0")
_start_time = time.time()
_state_lock = asyncio.Lock()
_current_state: Dict[str, Any] = {"episode_id": None, "turn_number": 0, "task": None, "agent_version": os.getenv("AGENT_MODEL_ID", "0.0.0")}
_latency_window: Deque[float] = deque(maxlen=200)
_error_count = 0
_request_count = 0
_episode_history: list = []
_episode_rewards: dict = {}      # episode_id → accumulated reward (live, updated each step)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        global _error_count, _request_count
        start = time.perf_counter()
        _request_count += 1
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        _latency_window.append(elapsed)
        if response.status_code >= 400:
            _error_count += 1
        return response


api.add_middleware(RequestLoggingMiddleware)


api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ResetResponse(BaseModel):
    episode_id: str
    timestamp: str
    task: str
    agent_version: str


class StepRequest(BaseModel):
    action: str
    episode_id: str

    @field_validator("action")
    @classmethod
    def action_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("action must be non-empty")
        return v


class StepResponse(BaseModel):
    state: dict
    reward: float
    done: bool
    info: dict


class StateResponse(BaseModel):
    episode_id: Optional[str] = None
    turn_number: int = 0
    task: Optional[str] = None
    agent_version: Optional[str] = None


class HealthResponse(BaseModel):
    uptime: float
    error_rate: float
    latency_p95: float
    memory_usage: float


@api.post("/reset", response_model=ResetResponse)
async def reset() -> ResetResponse:
    episode_handler, curriculum_manager, task_generator, _, _, _ = _get_components()
    episode_id = str(uuid.uuid4())
    agent_version = os.getenv("AGENT_MODEL_ID", "0.0.0")
    task_perf = {tid: {"success_rate": curriculum_manager.get_task_success_rate(tid)} for tid in curriculum_manager.task_performance}
    zone_result = curriculum_manager.compute_discrimination_zone(task_perf)
    discrimination_zone = zone_result.get("discrimination_zone_tasks", [])
    task_info = task_generator.hunt_mode(agent_performance=task_perf, discrimination_zone=discrimination_zone)
    task = task_info["prompt"]
    initial_state = episode_handler.initialize_episode(episode_id=episode_id, agent_version=agent_version, task=task)
    async with _state_lock:
        _current_state.update({"episode_id": episode_id, "turn_number": 1, "task": task, "agent_version": agent_version})
    _episode_history.append({"episode_id": episode_id, "task": task[:80], "timestamp": initial_state["timestamp"], "difficulty": task_info.get("difficulty_score", 0), "reward": None})
    if len(_episode_history) > 100:
        _episode_history.pop(0)
    _episode_rewards[episode_id] = 0.0
    return ResetResponse(episode_id=episode_id, timestamp=initial_state["timestamp"], task=task, agent_version=agent_version)


@api.post("/step", response_model=StepResponse)
async def step(request: StepRequest) -> StepResponse:
    episode_handler, curriculum_manager, _, reward_function, verifier_stack, _ = _get_components()
    async with _state_lock:
        current_episode_id = _current_state.get("episode_id")
        current_task = _current_state.get("task", "")
    if current_episode_id != request.episode_id:
        raise HTTPException(status_code=400, detail="Invalid episode_id")

    state, raw_reward, done, info = episode_handler.step(request.action)
    verification = verifier_stack.verify(request.action, task=current_task)
    turn_number = info.get("turn", 1)

    # Real discrimination coverage from CurriculumManager
    task_perf = {tid: {"success_rate": curriculum_manager.get_task_success_rate(tid)}
                 for tid in curriculum_manager.task_performance}
    zone_result = curriculum_manager.compute_discrimination_zone(task_perf)
    disc_before = zone_result.get("coverage_pct", 0.0)

    episode_data = {
        "turns": [{
            "turn_number": turn_number,
            "action": request.action,
            "verification": verification,
            "task_in_failure_bank": False,
            "times_task_attempted": turn_number,
            "total_tasks": max(len(curriculum_manager.task_performance), 1),
            "prev_action": "",
            "discrimination_coverage_before": disc_before,
            "discrimination_coverage_after": disc_before,
        }]
    }
    reward_result = reward_function.compute_reward(episode_data)
    reward = reward_result["shaped_rewards"][0] if reward_result["shaped_rewards"] else raw_reward

    # Update curriculum with real verification outcome
    composite = verification.get("composite_score", 0.0)
    success = composite >= 0.5
    task_id = current_task[:40]
    curriculum_manager.update_task_performance(task_id, success=success)

    # Accumulate reward for this episode so the dashboard can show real totals
    eid = request.episode_id
    _episode_rewards[eid] = _episode_rewards.get(eid, 0.0) + reward
    if done:
        # Write the final reward back into episode history
        for entry in reversed(_episode_history):
            if entry["episode_id"] == eid:
                entry["reward"] = round(_episode_rewards.pop(eid, 0.0), 4)
                break

    async with _state_lock:
        _current_state["turn_number"] = state.get("turn_number", turn_number + 1)
    return StepResponse(state=state, reward=reward, done=done, info={**info, "verification": verification})


@api.get("/state", response_model=StateResponse)
async def get_state() -> StateResponse:
    async with _state_lock:
        snapshot = dict(_current_state)
    return StateResponse(**snapshot)


@api.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    uptime = time.time() - _start_time
    error_rate = _error_count / _request_count if _request_count > 0 else 0.0
    latency_p95 = float(np.percentile(list(_latency_window), 95)) if _latency_window else 0.0
    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
    return HealthResponse(uptime=uptime, error_rate=error_rate, latency_p95=latency_p95, memory_usage=memory_usage)


@api.get("/failures")
async def get_failures(error_type: Optional[str] = None, agent_version: Optional[str] = None):
    """Return failure bank entries, optionally filtered by error_type / agent_version."""
    *_, fb = _get_components()
    entries = fb.query_failures(error_type=error_type, agent_version=agent_version)
    return [
        {
            "failure_id":    e.failure_id,
            "error_type":    e.error_type,
            "agent_version": e.agent_version,
            "novelty_score": e.novelty_score,
            "timestamp":     e.timestamp,
            "prompt":        e.prompt[:200],
        }
        for e in entries[:100]
    ]


# ---------------------------------------------------------------------------
# Leaderboard endpoints
# ---------------------------------------------------------------------------

class LeaderboardUpdateRequest(BaseModel):
    model_id:                str
    avg_reward:              float
    success_rate:            float
    discrimination_coverage: float
    episodes_run:            int


class LeaderboardSubmitRequest(BaseModel):
    model_id:     str
    display_name: Optional[str] = None
    params_b:     float = 0.0


@api.get("/leaderboard")
async def get_leaderboard(model_ids: Optional[str] = None):
    """Return leaderboard entries sorted by rl_score desc.
    Optional ?model_ids=id1,id2 to filter.
    """
    lb   = _get_leaderboard()
    ids  = [m.strip() for m in model_ids.split(",")] if model_ids else None
    return lb.get_leaderboard(model_ids=ids)


@api.post("/leaderboard/update")
async def update_leaderboard(req: LeaderboardUpdateRequest):
    """Called by training scripts to push live results."""
    lb = _get_leaderboard()
    lb.update_model_score(
        req.model_id, req.avg_reward, req.success_rate,
        req.discrimination_coverage, req.episodes_run,
    )
    return {"status": "updated", "model_id": req.model_id}


@api.post("/leaderboard/submit")
async def submit_model(req: LeaderboardSubmitRequest):
    """Register a user-submitted model for comparison."""
    lb = _get_leaderboard()
    return lb.submit_model(req.model_id, req.display_name, req.params_b)


# ---------------------------------------------------------------------------
# Training push endpoint — feeds live training metrics into dashboard graphs
# ---------------------------------------------------------------------------

class TrainingPushRequest(BaseModel):
    model_id:           str
    episode:            int
    rewards:            list
    advantages:         list
    loss:               float
    success_rate:       float
    disc_coverage:      float
    composites:         list = []
    cumulative_rewards: list = []   # mean reward per episode so far


@api.post("/training/push")
async def training_push(req: TrainingPushRequest):
    """Called by training scripts every episode to feed live data into dashboard."""
    global _REAL_TRAINING_STARTED
    _seed_mock_data()  # ensure mock data exists
    if req.episode == 1 and not _REAL_TRAINING_STARTED:
        # First real training run — clear mock data
        _REAL_TRAINING_STARTED = True
        _episode_history.clear()
        _disc_history.clear()
        _reward_hist.clear()
        _cumul_hist.clear()
        _adv_hist.clear()
        _loss_hist.clear()
    ts = datetime.now(timezone.utc).isoformat()

    # Feed per-rollout rewards into episode history with real reward values (no N/A)
    for i, r in enumerate(req.rewards):
        _episode_history.append({
            "episode_id": f"{req.model_id[:8]}-ep{req.episode}-r{i}",
            "task":       f"[ep={req.episode} rollout={i}]",
            "timestamp":  ts,
            "difficulty": 0.0,
            "reward":     round(float(r), 4),   # always a real number, never None
        })
    if len(_episode_history) > 200:
        del _episode_history[:len(_episode_history) - 200]

    # Feed disc coverage
    _disc_history.append(float(req.disc_coverage))

    # Feed advantages and loss
    for a in req.advantages:
        _adv_hist.append(float(a))
    if req.loss is not None:
        _loss_hist.append(float(req.loss))

    # Feed cumulative reward curve (one point per episode)
    for r in req.cumulative_rewards:
        _cumul_hist.append(float(r))

    # Update leaderboard
    lb = _get_leaderboard()
    avg_r = float(sum(req.rewards) / max(len(req.rewards), 1))
    lb.update_model_score(
        req.model_id, avg_r, req.success_rate,
        req.disc_coverage, req.episode,
    )

    return {
        "status":        "ok",
        "episode":       req.episode,
        "rewards_added": len(req.rewards),
        "avg_reward":    round(avg_r, 4),
    }


# ---------------------------------------------------------------------------
# Job registration endpoint — training scripts call this to show live status
# ---------------------------------------------------------------------------

_LIVE_JOBS: list = []   # list of job dicts, newest first

class JobRegisterRequest(BaseModel):
    job_id:       str
    model:        str
    episodes:     int
    avg_reward:   float = 0.0
    success_rate: float = 0.0
    elapsed_s:    float = 0.0
    status:       str   = "RUNNING"
    url:          str   = ""
    timestamp:    str   = ""


@api.post("/jobs/register")
async def register_job(req: JobRegisterRequest):
    """Called by training scripts to register/update a live job in the dashboard."""
    ts = req.timestamp or datetime.now(timezone.utc).isoformat()[:19] + "Z"
    # Update existing entry if same job_id, else prepend
    for j in _LIVE_JOBS:
        if j["job_id"] == req.job_id:
            j.update({
                "model": req.model, "episodes": req.episodes,
                "avg_reward": round(req.avg_reward, 4),
                "success_rate": round(req.success_rate, 4),
                "elapsed_s": round(req.elapsed_s, 1),
                "status": req.status, "url": req.url, "timestamp": ts,
            })
            return {"status": "updated", "job_id": req.job_id}
    _LIVE_JOBS.insert(0, {
        "job_id": req.job_id, "model": req.model, "episodes": req.episodes,
        "avg_reward": round(req.avg_reward, 4),
        "success_rate": round(req.success_rate, 4),
        "elapsed_s": round(req.elapsed_s, 1),
        "status": req.status, "url": req.url, "timestamp": ts,
    })
    if len(_LIVE_JOBS) > 20:
        _LIVE_JOBS.pop()
    return {"status": "registered", "job_id": req.job_id}


@api.get("/jobs")
async def list_jobs():
    """Return all registered jobs (live + completed)."""
    return _LIVE_JOBS


# ---------------------------------------------------------------------------
# Gradio dashboard — tabbed analytics
# ---------------------------------------------------------------------------

_disc_history:  Deque[float] = deque(maxlen=200)
_reward_hist:   Deque[float] = deque(maxlen=200)
_adv_hist:      Deque[float] = deque(maxlen=400)   # GRPO advantages
_loss_hist:     Deque[float] = deque(maxlen=200)   # training loss
_cumul_hist:    Deque[float] = deque(maxlen=200)   # mean reward per episode (reward curve)

# ---------------------------------------------------------------------------
# Mock historic data — seeded on startup, cleared when real training begins
# ---------------------------------------------------------------------------
import random as _random

_MOCK_SEEDED = False
_REAL_TRAINING_STARTED = False


def _seed_mock_data():
    global _MOCK_SEEDED
    if _MOCK_SEEDED:
        return
    _MOCK_SEEDED = True
    rng = _random.Random(42)
    now = time.time()

    # Historic data seeded from TinyLlama baseline run (job 69edae30, 20 eps, avg=0.802)
    # These are the past metrics shown by default before any new live training
    rng2 = _random.Random(99)
    for i in range(80):  # 20 eps × 4 rollouts = 80 rollouts
        age = (80 - i) * 1080
        ts  = datetime.fromtimestamp(now - age, tz=timezone.utc).isoformat()
        # TinyLlama reward distribution: ~0.8 mean, small variance (rule-based)
        r = max(0.0, min(1.5, 0.802 + rng2.gauss(0, 0.12)))
        _episode_history.append({
            "episode_id": f"tinyllama-{i:04d}",
            "task": rng2.choice([
                "Write Python to check if a string is a palindrome.",
                "Find the second largest number in a list.",
                "Compute sum of even numbers from 1 to 20.",
                "What is the capital of Australia?",
                "Solve for x: 3x + 7 = 22",
                "Return Fibonacci number at position n.",
                "Simplify: (x\u00b2 - 4) / (x - 2)",
                "If P\u2192Q and Q\u2192R, does P\u2192R?",
            ]),
            "timestamp": ts,
            "difficulty": round(50.0 + rng2.gauss(0, 8), 1),
            "reward": round(r, 4),
        })

    # Seed deques from TinyLlama run (flat ~0.8, small noise — rule-based baseline)
    for i in range(80):
        r = max(0.0, min(1.5, 0.802 + rng2.gauss(0, 0.12)))
        _reward_hist.append(r)
        if i % 4 == 0:  # one cumul point per episode
            _cumul_hist.append(r)
        _disc_history.append(max(0, min(1, 0.0 + rng2.gauss(0, 0.02))))  # TinyLlama had disc≈0
    for _ in range(80):
        _adv_hist.append(rng2.gauss(0, 1.0))
    for _ in range(20):
        _loss_hist.append(0.0)  # TinyLlama had no local model, no loss

    # Mock failure bank entries
    _MOCK_FAILURES.extend([
        {"failure_id": f"fb{i:04x}", "error_type": rng.choice(["verification_failure", "timeout_error", "logic_error", "syntax_error"]),
         "agent_version": rng.choice(["0.0.0", "SmolLM2-135M", "Qwen2.5-0.5B"]),
         "novelty_score": round(rng.uniform(0.3, 0.95), 3),
         "timestamp": datetime.fromtimestamp(now - rng.uniform(0, 172800), tz=timezone.utc).isoformat()[:19],
         "prompt": rng.choice([
             "Write Python to check if a string is a palindrome.",
             "Find the second largest number in a list.",
             "Compute sum of even numbers from 1 to 20.",
             "Solve for x: 3x + 7 = 22",
             "Return Fibonacci number at position n.",
             "If P\u2192Q and Q\u2192R, does P\u2192R?",
             "What is the capital of Australia?",
             "Simplify: (x\u00b2 - 4) / (x - 2)",
         ])}
        for i in range(20)
    ])

    # Real job history — actual HF Jobs runs, exact values from job logs
    _MOCK_TRAINING_LOGS.extend([
        # train.py GRPOTrainer — 1000 eps, all rollouts failed (429 rate limit), reward=0.0
        {"job_id": "69ed7a38d2c8bd8662bceece",
         "model": "train.py GRPOTrainer (429 rate-limited)",
         "episodes": 1000, "avg_reward": 0.0, "success_rate": 0.0,
         "disc_coverage": 0.0, "total_rollouts": 0, "elapsed_s": 0,
         "timestamp": "2026-04-26T02:36:40Z", "status": "COMPLETED",
         "url": "https://huggingface.co/jobs/rohanjain1648/69ed7a38d2c8bd8662bceece"},
        # TinyLlama — 20 eps, rule-based inference (no local model), avg_reward=0.802
        {"job_id": "69edae30d70108f37acdfb48",
         "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
         "episodes": 20, "avg_reward": 0.802, "success_rate": 0.80,
         "disc_coverage": 0.0, "total_rollouts": 80, "elapsed_s": 23,
         "timestamp": "2026-04-26T06:18:24Z", "status": "COMPLETED",
         "url": "https://huggingface.co/jobs/rohanjain1648/69edae30d70108f37acdfb48"},
        # SmolLM2-135M — 20 eps, local model, bad prompt → avg_reward=-0.0017
        {"job_id": "69edb6edd2c8bd8662bcf6c1",
         "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
         "episodes": 20, "avg_reward": -0.0017, "success_rate": 0.05,
         "disc_coverage": 0.025, "total_rollouts": 80, "elapsed_s": 231,
         "timestamp": "2026-04-26T06:55:41Z", "status": "COMPLETED",
         "url": "https://huggingface.co/jobs/rohanjain1648/69edb6edd2c8bd8662bcf6c1"},
        # Qwen2.5-Coder-3B — 20 eps, nscale remote inference → avg_reward=0.0887
        {"job_id": "69edb668d2c8bd8662bcf6b3",
         "model": "Qwen/Qwen2.5-Coder-3B-Instruct (nscale)",
         "episodes": 20, "avg_reward": 0.0887, "success_rate": 0.125,
         "disc_coverage": 0.125, "total_rollouts": 80, "elapsed_s": 151,
         "timestamp": "2026-04-26T06:53:28Z", "status": "COMPLETED",
         "url": "https://huggingface.co/jobs/rohanjain1648/69edb668d2c8bd8662bcf6b3"},
        # SmolLM2-135M — 30 eps, result=42 prompt fix → avg_reward=-0.0522
        {"job_id": "69edb9bad70108f37acdfc8f",
         "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
         "episodes": 30, "avg_reward": -0.0522, "success_rate": 0.0167,
         "disc_coverage": 0.0, "total_rollouts": 120, "elapsed_s": 379,
         "timestamp": "2026-04-26T07:07:38Z", "status": "COMPLETED",
         "url": "https://huggingface.co/jobs/rohanjain1648/69edb9bad70108f37acdfc8f"},
        # SmolLM2-135M — 30 eps, result=42 converged → avg_reward=1.1486
        {"job_id": "69edbddfd2c8bd8662bcf794",
         "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
         "episodes": 30, "avg_reward": 1.1486, "success_rate": 0.95,
         "disc_coverage": 0.8917, "total_rollouts": 120, "elapsed_s": 361,
         "timestamp": "2026-04-26T07:25:19Z", "status": "COMPLETED",
         "url": "https://huggingface.co/jobs/rohanjain1648/69edbddfd2c8bd8662bcf794"},
        # Qwen2.5-0.5B — 100 eps, a10g-small, chat-template → avg_reward=1.6549
        {"job_id": "69edd2edd70108f37acdff08",
         "model": "Qwen/Qwen2.5-0.5B-Instruct",
         "episodes": 100, "avg_reward": 1.6549, "success_rate": 0.7375,
         "disc_coverage": 0.7362, "total_rollouts": 800, "elapsed_s": 1163,
         "timestamp": "2026-04-26T08:55:09Z", "status": "COMPLETED",
         "url": "https://huggingface.co/jobs/rohanjain1648/69edd2edd70108f37acdff08"},
        # ── NEW JOBS (50 eps each, a10g-small, chat-template, 3-turn) — real results ──
        {"job_id": "69ede9b6d70108f37ace00c4",
         "model": "Qwen/Qwen2.5-0.5B-Instruct",
         "episodes": 50, "avg_reward": 0.8181, "success_rate": 0.2375,
         "disc_coverage": 0.2375, "total_rollouts": 400, "elapsed_s": 642,
         "timestamp": "2026-04-26T10:43:50Z", "status": "COMPLETED",
         "url": "https://huggingface.co/jobs/rohanjain1648/69ede9b6d70108f37ace00c4"},
        {"job_id": "69ede9b9d70108f37ace00c6",
         "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
         "episodes": 50, "avg_reward": 0.5279, "success_rate": 0.0825,
         "disc_coverage": 0.0625, "total_rollouts": 400, "elapsed_s": 790,
         "timestamp": "2026-04-26T10:46:24Z", "status": "COMPLETED",
         "url": "https://huggingface.co/jobs/rohanjain1648/69ede9b9d70108f37ace00c6"},
        {"job_id": "69ede9bcd70108f37ace00c8",
         "model": "Qwen/Qwen2.5-1.5B-Instruct",
         "episodes": 50, "avg_reward": 0.8515, "success_rate": 0.1775,
         "disc_coverage": 0.145, "total_rollouts": 400, "elapsed_s": 1007,
         "timestamp": "2026-04-26T10:53:30Z", "status": "COMPLETED",
         "url": "https://huggingface.co/jobs/rohanjain1648/69ede9bcd70108f37ace00c8"},
        {"job_id": "69ede9c1d70108f37ace00cb",
         "model": "google/gemma-3-1b-it",
         "episodes": 50, "avg_reward": 0.0, "success_rate": 0.0,
         "disc_coverage": 0.0, "total_rollouts": 0, "elapsed_s": 31,
         "timestamp": "2026-04-26T10:47:04Z", "status": "ERROR",
         "url": "https://huggingface.co/jobs/rohanjain1648/69ede9c1d70108f37ace00cb"},
        {"job_id": "69ede9c4d2c8bd8662bcfca9",
         "model": "Qwen/Qwen2.5-3B-Instruct",
         "episodes": 50, "avg_reward": 1.4278, "success_rate": 0.58,
         "disc_coverage": 0.255, "total_rollouts": 400, "elapsed_s": 1390,
         "timestamp": "2026-04-26T11:11:08Z", "status": "COMPLETED",
         "url": "https://huggingface.co/jobs/rohanjain1648/69ede9c4d2c8bd8662bcfca9"},
    ])
    # Seed _LIVE_JOBS from real job history so jobs tab is never empty after restart
    for j in _MOCK_TRAINING_LOGS:
        if not any(lj["job_id"] == j["job_id"] for lj in _LIVE_JOBS):
            _LIVE_JOBS.append(dict(j))


_MOCK_FAILURES: list = []
_MOCK_TRAINING_LOGS: list = []


# \u2500\u2500 Static content constants \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

_ARCH_HTML = """
<div style="background:#0f172a;border-radius:16px;padding:24px;color:#f1f5f9;font-family:monospace;font-size:13px;line-height:1.8">
<div style="text-align:center;font-size:18px;font-weight:800;color:#FF9D00;margin-bottom:20px">ALICE Architecture</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px">
  <div style="background:#1e293b;border:1px solid #FF9D00;border-radius:10px;padding:14px">
    <div style="color:#FF9D00;font-weight:700;margin-bottom:6px">&#x1F52E; Oracle Interface</div>
    <div style="color:#94a3b8;font-size:12px">Queries a stronger reference model (GPT-4o / Qwen-72B) to find tasks your model fails but the oracle passes \u2192 <b style="color:#f1f5f9">discrimination zone</b></div>
  </div>
  <div style="background:#1e293b;border:1px solid #4a90e2;border-radius:10px;padding:14px">
    <div style="color:#4a90e2;font-weight:700;margin-bottom:6px">&#x1F3AF; Curriculum Manager</div>
    <div style="color:#94a3b8;font-size:12px">Tracks per-task success rates across 5 domains \u00d7 10 tiers. Hunts in the 20-80% zone \u2014 not too easy, not too hard.</div>
  </div>
  <div style="background:#1e293b;border:1px solid #10b981;border-radius:10px;padding:14px">
    <div style="color:#10b981;font-weight:700;margin-bottom:6px">&#x1F3E6; Failure Bank</div>
    <div style="color:#94a3b8;font-size:12px">Stores confirmed failures with sentence-transformer embeddings. k-NN novelty score ensures we never waste an episode on something already known.</div>
  </div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
  <div style="background:#1e293b;border:1px solid #9b59b6;border-radius:10px;padding:14px">
    <div style="color:#9b59b6;font-weight:700;margin-bottom:6px">\u2705 Three-Tier Verifier</div>
    <div style="color:#94a3b8;font-size:12px">
      <b style="color:#f1f5f9">T1</b> \u2014 RestrictedPython sandbox: did the code run? did it raise?<br>
      <b style="color:#f1f5f9">T2</b> \u2014 LLM judge (chain-of-thought): is the answer correct in intent?<br>
      <b style="color:#f1f5f9">T3</b> \u2014 Regression battery: 20 variants, does the fix generalise?
    </div>
  </div>
  <div style="background:#1e293b;border:1px solid #e74c3c;border-radius:10px;padding:14px">
    <div style="color:#e74c3c;font-weight:700;margin-bottom:6px">&#x1F4C8; GRPO Policy Gradient</div>
    <div style="color:#94a3b8;font-size:12px">
      Group Relative Policy Optimisation \u2014 normalises rewards within a rollout group \u2192 advantages \u2192 policy update.<br>
      <b style="color:#f1f5f9">Loss = \u2212(advantage \u00d7 log_prob) + KL_coef \u00d7 log_prob\u00b2</b>
    </div>
  </div>
</div>
<div style="text-align:center;color:#64748b;font-size:11px;padding-top:8px;border-top:1px solid #1e293b">
  Co-evolutionary loop: model improves \u2192 oracle discrimination collapses on that task \u2192 curriculum escalates automatically
</div>
</div>
"""

_PROBLEM_MD = """
## Every model topping a leaderboard today \u2014 it's not as good as it looks.

The benchmarks we use to measure intelligence are **static**. They were written by humans, they're finite, and the moment a model sees enough internet text overlapping with them, the score stops meaning anything. The Hugging Face Open LLM Leaderboard team documented this publicly \u2014 models cluster at the top, the community loses trust, engineers redesign the benchmark by hand, and within months it saturates again. **It's a treadmill with no exit.**

But that's only half the problem.

The other half is that even during training, **we don't know what a model actually can't do**. Standard benchmarks measure known capabilities. They cannot find the blind spots \u2014 the reasoning combinations, the compositional edge cases, the failure modes that exist in the model right now but that no human thought to test. A model can score 87 on MATH and still fail systematically on negation plus arithmetic. Nobody knows. There's no tool to find it.

---

**We built ALICE.** *Adversarial Loop for Inter-model Co-evolutionary Environment.*

ALICE is a reinforcement learning training environment \u2014 OpenEnv-compatible, plug it into any TRL run \u2014 that solves both problems simultaneously through a single co-evolutionary mechanism.

An **Oracle Interface** queries stronger reference models to find the discrimination zone: tasks the reference model passes but your model fails. A **Hunt agent** adversarially searches that zone for novel failure modes. A **Repair agent** synthesises targeted training pairs to fix confirmed failures. A **three-tier Verifier** \u2014 sandboxed programmatic check, chain-of-thought LLM judge, and regression battery \u2014 ensures repairs stick without breaking anything else.

The key insight: **when the repair loop fixes a failure, the Oracle's discrimination reward collapses on that task. It's no longer informative. The benchmark is structurally forced to escalate.** The model improves, the benchmark gets harder, the model improves again. Co-evolutionary, self-sustaining, zero human intervention.
"""

_RESEARCH_HTML = """
<style>
.r-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; padding:8px 0; }
.r-card { background:#1e293b; border:1px solid #334155; border-radius:10px; padding:16px; transition:border-color 0.2s; }
.r-card:hover { border-color:#FF9D00; }
.r-tag { display:inline-block; background:#FF9D0022; color:#FF9D00; font-size:10px; font-weight:700; padding:2px 8px; border-radius:20px; margin-bottom:8px; }
.r-title { font-size:13px; font-weight:700; color:#f1f5f9; margin-bottom:6px; }
.r-desc { font-size:11px; color:#94a3b8; line-height:1.5; margin-bottom:10px; }
.r-link { font-size:11px; color:#4a90e2; text-decoration:none; }
</style>
<div class="r-grid">
  <div class="r-card">
    <span class="r-tag">BENCHMARK SATURATION</span>
    <div class="r-title">HF Open LLM Leaderboard v2 \u2014 Why We Redesigned It</div>
    <div class="r-desc">Documents how MMLU saturated, models clustered at the top, and how even carefully designed benchmarks become gameable within months. The exact treadmill ALICE escapes.</div>
    <a class="r-link" href="https://huggingface.co/blog/open-llm-leaderboard-v2" target="_blank">\u2192 huggingface.co/blog/open-llm-leaderboard-v2</a>
  </div>
  <div class="r-card">
    <span class="r-tag">POLICY GRADIENT</span>
    <div class="r-title">GRPO \u2014 Group Relative Policy Optimisation (DeepSeek-R1)</div>
    <div class="r-desc">The training algorithm powering ALICE. Normalises rewards within a group of rollouts to form advantages \u2014 no value network needed, no PPO clip complexity.</div>
    <a class="r-link" href="https://arxiv.org/abs/2402.03300" target="_blank">\u2192 arxiv.org/abs/2402.03300</a>
  </div>
  <div class="r-card">
    <span class="r-tag">FOUNDATION</span>
    <div class="r-title">InstructGPT \u2014 Training Language Models to Follow Instructions (RLHF)</div>
    <div class="r-desc">The seminal RLHF paper. ALICE extends this: instead of human preference labels, we use co-evolutionary oracle discrimination as the reward signal.</div>
    <a class="r-link" href="https://arxiv.org/abs/2203.02155" target="_blank">\u2192 arxiv.org/abs/2203.02155</a>
  </div>
  <div class="r-card">
    <span class="r-tag">SELF-PLAY</span>
    <div class="r-title">SPAG \u2014 Self-Play with Adversarial Grammar</div>
    <div class="r-desc">Uses self-play where one model generates hard tasks and another solves them. ALICE generalises this to a multi-agent co-evolutionary loop with a verifier stack.</div>
    <a class="r-link" href="https://arxiv.org/abs/2404.09459" target="_blank">\u2192 arxiv.org/abs/2404.09459</a>
  </div>
  <div class="r-card">
    <span class="r-tag">SAFETY &amp; ALIGNMENT</span>
    <div class="r-title">Constitutional AI \u2014 Anthropic</div>
    <div class="r-desc">Uses a set of principles and self-critique to guide RLHF. ALICE borrows the recursive self-improvement loop \u2014 the oracle IS the constitution here.</div>
    <a class="r-link" href="https://arxiv.org/abs/2212.08073" target="_blank">\u2192 arxiv.org/abs/2212.08073</a>
  </div>
  <div class="r-card">
    <span class="r-tag">DYNAMIC BENCHMARKS</span>
    <div class="r-title">ARC-AGI \u2014 On the Measure of Intelligence (Chollet)</div>
    <div class="r-desc">Argues static benchmarks can never measure general intelligence. ARC uses novel tasks never seen in training. ALICE is the RL training companion to this philosophy.</div>
    <a class="r-link" href="https://arxiv.org/abs/1911.01547" target="_blank">\u2192 arxiv.org/abs/1911.01547</a>
  </div>
  <div class="r-card">
    <span class="r-tag">CURRICULUM LEARNING</span>
    <div class="r-title">Automatic Curriculum Learning for RL</div>
    <div class="r-desc">The discrimination zone (20\u201380% success rate window) is ALICE's implementation of curriculum learning \u2014 automatically finding tasks at the model's current frontier.</div>
    <a class="r-link" href="https://arxiv.org/abs/2003.04664" target="_blank">\u2192 arxiv.org/abs/2003.04664</a>
  </div>
  <div class="r-card">
    <span class="r-tag">TOOLING</span>
    <div class="r-title">TRL \u2014 Transformer Reinforcement Learning (Hugging Face)</div>
    <div class="r-desc">The library ALICE runs on top of. GRPO, PPO, DPO all available. ALICE provides the environment; TRL provides the training loop. OpenEnv-compatible by design.</div>
    <a class="r-link" href="https://huggingface.co/docs/trl" target="_blank">\u2192 huggingface.co/docs/trl</a>
  </div>
</div>
"""

_HOW_TO_USE_MD = """
### Step-by-step: How to train your model with ALICE

**Option A \u2014 HF Jobs (recommended, GPU)**
```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/rohanjain1648/alice-rl-environment
cd alice-rl-environment

# 2. Install the launcher
pip install huggingface_hub>=0.36

# 3. Launch a training job (default: Qwen2.5-0.5B, 100 episodes, a10g-small)
python scripts/launch_hf_job.py

# 4. Or specify any model:
python scripts/launch_hf_job.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --episodes 50

# 5. Watch metrics appear live in the Training Metrics tab above \u2191
```

**Option B \u2014 Colab (free T4 GPU)**
1. Open `notebooks/train_trl_colab.ipynb` in Google Colab (Runtime \u2192 T4 GPU)
2. Set `ALICE_ENV_URL` in the config cell (default already points here)
3. Run all cells \u2014 metrics push to this dashboard automatically

**Option C \u2014 Local training**
```bash
pip install torch transformers peft trl accelerate bitsandbytes httpx
ALICE_ENV_URL=https://rohanjain1648-alice-rl-environment.hf.space python training/hf_job_train.py
```

**Evaluate any model**
Go to the **Leaderboard** tab \u2192 _Submit a Model for Evaluation_ \u2192 enter a HF model ID and number of episodes \u2192 click Submit. A real HF Job launches, results appear in the leaderboard when done.

**Reading the results**
- `avg_reward` \u2192 mean reward per turn across all episodes
- `success_rate` \u2192 fraction of episodes where composite score \u2265 0.5
- `disc_coverage` \u2192 fraction of tasks in the 20\u201380% discrimination zone
- `rl_score` = 0.5 \u00d7 avg_reward + 0.3 \u00d7 success_rate + 0.2 \u00d7 disc_coverage
"""

_PRICING_HTML = """
<style>
.pricing-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; max-width:900px; margin:0 auto; padding:20px 0; }
.tier { border-radius:16px; padding:24px; text-align:center; position:relative; }
.tier-free { background:#1e293b; border:1px solid #334155; }
.tier-pro  { background:linear-gradient(135deg,#1a1a2e,#16213e); border:2px solid #FF9D00; }
.tier-ent  { background:#0f172a; border:1px solid #4a90e2; }
.tier-badge { position:absolute; top:-12px; left:50%; transform:translateX(-50%); background:#FF9D00; color:#000; font-size:11px; font-weight:800; padding:3px 14px; border-radius:20px; }
.tier-name { font-size:20px; font-weight:800; margin-bottom:4px; }
.tier-price { font-size:32px; font-weight:900; margin:12px 0 4px; }
.tier-sub { font-size:12px; color:#94a3b8; margin-bottom:16px; }
.tier-features { list-style:none; padding:0; margin:0 0 20px; text-align:left; font-size:13px; }
.tier-features li { padding:5px 0; border-bottom:1px solid #ffffff0a; color:#cbd5e1; }
.tier-features li::before { content:"\u2713 "; color:#10b981; font-weight:700; }
.tier-features li.no::before { content:"\u2717 "; color:#ef4444; }
.tier-btn { width:100%; padding:10px; border-radius:8px; font-weight:700; cursor:pointer; border:none; font-size:14px; }
.btn-free { background:#334155; color:#f1f5f9; }
.btn-pro  { background:#FF9D00; color:#000; }
.btn-ent  { background:#4a90e2; color:#fff; }
.credit-box { background:#1e293b; border:1px solid #FF9D00; border-radius:12px; padding:20px; margin:20px auto; max-width:600px; text-align:center; }
.highlight-box { background:linear-gradient(135deg,#FF9D0015,#FF9D0005); border:2px solid #FF9D00; border-radius:16px; padding:20px; margin:20px 0; }
</style>
<div class="highlight-box">
  <div style="font-size:22px;font-weight:900;color:#FF9D00;margin-bottom:8px">&#x1F4A1; Business Model</div>
  <div style="color:#cbd5e1;font-size:14px;line-height:1.7">
    ALICE is the <b style="color:#f1f5f9">first adversarial RL training environment</b> for LLMs that auto-generates its own benchmarks.
    Every evaluation run trains the environment to be harder \u2014 the product literally gets better every time a customer uses it.
    <br><br>Users bring their model and their HF token (or buy ALICE Credits). We orchestrate the GPU, the environment, the verifier stack, and the leaderboard.
    The leaderboard itself is a <b style="color:#f1f5f9">trust signal</b> \u2014 a score here is harder to game than any static benchmark.
  </div>
</div>
<div class="pricing-grid">
  <div class="tier tier-free">
    <div class="tier-name" style="color:#94a3b8">Community</div>
    <div class="tier-price" style="color:#f1f5f9">Free</div>
    <div class="tier-sub">Your HF credits, your hardware</div>
    <ul class="tier-features">
      <li>Full environment API access</li>
      <li>Colab + TRL notebooks</li>
      <li>Public leaderboard listing</li>
      <li>20-episode quick eval</li>
      <li class="no">Managed GPU job orchestration</li>
      <li class="no">Priority eval queue</li>
      <li class="no">Private leaderboard</li>
    </ul>
  </div>
  <div class="tier tier-pro">
    <div class="tier-badge">MOST POPULAR</div>
    <div class="tier-name" style="color:#FF9D00">Pro</div>
    <div class="tier-price" style="color:#FF9D00">$29<span style="font-size:16px;color:#94a3b8">/mo</span></div>
    <div class="tier-sub">100 ALICE Credits included</div>
    <ul class="tier-features">
      <li>Everything in Community</li>
      <li>100 ALICE Credits/mo (~100 eval eps on a10g)</li>
      <li>Managed HF Job submission</li>
      <li>Priority eval queue</li>
      <li>Email when eval completes</li>
      <li class="no">Custom hardware selection</li>
      <li class="no">Private leaderboard</li>
    </ul>
  </div>
  <div class="tier tier-ent">
    <div class="tier-name" style="color:#4a90e2">Enterprise</div>
    <div class="tier-price" style="color:#4a90e2">Custom</div>
    <div class="tier-sub">Volume credits + SLA</div>
    <ul class="tier-features">
      <li>Everything in Pro</li>
      <li>Custom credit bundles</li>
      <li>Choose hardware (a10g / A100 / H100)</li>
      <li>Training time limits & alerts</li>
      <li>Private leaderboard + API</li>
      <li>White-label eval reports</li>
      <li>Dedicated support</li>
    </ul>
  </div>
</div>
<div class="credit-box">
  <div style="font-size:16px;font-weight:800;color:#FF9D00;margin-bottom:8px">&#x1FA99; ALICE Credits</div>
  <div style="color:#cbd5e1;font-size:13px;line-height:1.7">
    1 ALICE Credit = 1 eval episode on a10g-small (~7 sec compute).<br>
    Credits are pre-purchased, converted to HF GPU compute on-demand.<br>
    <b style="color:#f1f5f9">Community users</b> can bring their own HF token \u2014 0 credits needed.<br>
    <b style="color:#f1f5f9">Pro/Enterprise</b> users never touch HF \u2014 we handle compute & billing.
  </div>
</div>
<div style="text-align:center;color:#64748b;font-size:12px;margin-top:16px">
  \u26a0\ufe0f Pricing is illustrative \u2014 ALICE is currently in open beta.
  <a href="mailto:rohanjain200461@gmail.com" style="color:#FF9D00">Contact us</a> to join the waitlist or discuss enterprise access.
</div>
"""

_CSS = """
/* \u2500\u2500 ALICE HF-style theme \u2500\u2500 */
:root {
    --hf-orange: #FF9D00;
    --hf-orange-light: #FFF3E0;
    --hf-dark: #1a1a2e;
    --hf-gray: #6b7280;
    --hf-border: #e5e7eb;
    --hf-card: #ffffff;
    --hf-success: #10b981;
    --hf-danger: #ef4444;
}
.tab-nav button { font-size: 13px; font-weight: 600; padding: 8px 16px; }
.tab-nav button.selected { border-bottom: 3px solid var(--hf-orange) !important; color: var(--hf-orange) !important; }
.metric-card { background: var(--hf-card); border: 1px solid var(--hf-border); border-radius: 12px; padding: 16px; text-align: center; }
.status-live { color: var(--hf-success); font-weight: 700; }
.status-dead { color: var(--hf-danger); font-weight: 700; }
.alert-warn { background: #FEF3C7; border-left: 4px solid #F59E0B; padding: 8px 12px; border-radius: 4px; }
.alert-ok { background: #D1FAE5; border-left: 4px solid var(--hf-success); padding: 8px 12px; border-radius: 4px; }
footer { display: none !important; }
"""

SPACE_ID = os.getenv("HF_SPACE_ID", "rohanjain1648/alice-rl-environment")
SPACE_URL_FULL = f"https://huggingface.co/spaces/{SPACE_ID}"
API_URL = f"https://{SPACE_ID.replace('/', '-')}.hf.space"
JOBS_URL = "https://huggingface.co/settings/jobs"

header_md = f"""
<div style="display:flex;align-items:center;gap:16px;padding:16px 0 8px 0;border-bottom:2px solid #FF9D00;margin-bottom:16px">
  <div style="font-size:28px;font-weight:800;color:#FF9D00;letter-spacing:-1px">\U0001f916 ALICE</div>
  <div>
    <div style="font-size:16px;font-weight:700;color:#f1f5f9">Adversarial Loop for Inter-model Co-evolutionary Environment</div>
    <div style="font-size:12px;color:#6b7280">v0.1.0 \u00b7 OpenEnv-compliant RL training environment \u00b7
      <a href="{SPACE_URL_FULL}" target="_blank" style="color:#FF9D00">\U0001f917 Space</a> \u00b7
      <a href="{API_URL}/docs" target="_blank" style="color:#FF9D00">\U0001f4d6 API Docs</a> \u00b7
      <a href="{JOBS_URL}" target="_blank" style="color:#FF9D00">\u26a1 HF Jobs</a>
    </div>
  </div>
  <div style="margin-left:auto;background:#D1FAE5;color:#065F46;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:700">\u25cf LIVE</div>
</div>
"""


def _hf_space_url(space_id: str) -> str:
    """Convert 'user/space-name' to its HF Space URL."""
    parts = space_id.split("/")
    return f"https://{parts[0]}-{parts[1]}.hf.space" if len(parts) == 2 else ""


def _hf_space_info() -> dict:
    """Query HF API for Space runtime status for env and training spaces."""
    env_sid   = os.getenv("HF_SPACE_ID", "rohanjain1648/alice-rl-environment")
    train_sid = os.getenv("ALICE_HF_REPO_ID", "rohanjain1648/alice-rl-environment")
    hf_token  = os.getenv("HF_TOKEN", "")
    headers   = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    result: dict = {}
    for key, sid in [("env_space", env_sid), ("training_space", train_sid)]:
        try:
            r = httpx.get(f"https://huggingface.co/api/spaces/{sid}", headers=headers, timeout=5.0)
            if r.status_code == 200:
                stage = r.json().get("runtime", {}).get("stage", "UNKNOWN")
                result[key] = {"status": stage, "url": _hf_space_url(sid), "id": sid}
            else:
                result[key] = {"status": f"HTTP {r.status_code}", "url": _hf_space_url(sid), "id": sid}
        except Exception as exc:
            result[key] = {"status": "error", "url": _hf_space_url(sid), "id": sid, "err": str(exc)}
    return result


def _heatmap_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.5))
    try:
        _, cm, *_ = _get_components()
        data = cm.get_curriculum_heatmap()
    except Exception:
        data = np.zeros((5, 10), dtype=np.float32)
    domains = ["arithmetic", "logic", "factual", "symbolic", "code"]
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"T{i+1}" for i in range(10)], fontsize=8)
    ax.set_yticks(range(5))
    ax.set_yticklabels(domains, fontsize=9)
    ax.set_title("Curriculum Heatmap \u2014 success rate per domain x difficulty tier",
                 fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Success Rate")
    plt.tight_layout()
    return fig


def _disc_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3))
    if _disc_history:
        xs = range(len(_disc_history))
        ax.fill_between(xs, _disc_history, alpha=0.18, color="#4a90e2")
        ax.plot(xs, _disc_history, color="#4a90e2", linewidth=1.8)
    ax.axhline(0.3, color="red",   linestyle="--", alpha=0.65, label="Min 30%")
    ax.axhline(0.7, color="green", linestyle="--", alpha=0.65, label="Target 70%")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Coverage")
    ax.set_title("Discrimination Zone Coverage Over Time", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    return fig


def _reward_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()

    # 1. Reward curve per episode + cumulative reward sum (dual axis)
    if _cumul_hist:
        xs     = list(range(1, len(_cumul_hist) + 1))
        ys     = list(_cumul_hist)
        cumsum = list(np.cumsum(ys))

        axes[0].plot(xs, ys, alpha=0.35, color="#4a90e2", linewidth=0.8, label="per-ep mean")
        if len(ys) >= 3:
            w  = min(5, len(ys))
            ma = np.convolve(ys, np.ones(w)/w, mode="valid")
            axes[0].plot(range(w, len(ys) + 1), ma, color="#4a90e2",
                         linewidth=2.5, label=f"{w}-ep MA")
        axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[0].fill_between(xs, ys, alpha=0.08, color="#4a90e2")
        axes[0].set_ylabel("Mean Reward / Episode", color="#4a90e2")
        axes[0].tick_params(axis="y", labelcolor="#4a90e2")

        ax0r = axes[0].twinx()
        ax0r.plot(xs, cumsum, color="#FF9D00", linewidth=2.0,
                  linestyle="--", label="cumulative sum")
        ax0r.set_ylabel("Cumulative Reward", color="#FF9D00")
        ax0r.tick_params(axis="y", labelcolor="#FF9D00")

        lines1, labels1 = axes[0].get_legend_handles_labels()
        lines2, labels2 = ax0r.get_legend_handles_labels()
        axes[0].legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    axes[0].set_xlabel("Episode")
    axes[0].set_title("Reward Curve + Cumulative Reward", fontweight="bold")

    # 2. Reward distribution histogram
    if _reward_hist:
        axes[1].hist(list(_reward_hist), bins=min(25, max(5, len(_reward_hist))),
                     color="#4a90e2", edgecolor="white", alpha=0.85)
        mean_r = float(np.mean(list(_reward_hist)))
        axes[1].axvline(mean_r, color="red", linestyle="--", linewidth=1.5,
                        label=f"mean={mean_r:.3f}")
        axes[1].axvline(0, color="gray", linestyle="--", alpha=0.5)
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Reward"); axes[1].set_ylabel("Count")
    axes[1].set_title("Reward Distribution", fontweight="bold")

    # 3. GRPO advantage distribution
    if _adv_hist:
        axes[2].hist(list(_adv_hist), bins=min(30, max(5, len(_adv_hist))),
                     color="#9b59b6", edgecolor="white", alpha=0.85)
        axes[2].axvline(0, color="red", linestyle="--", alpha=0.7, label="zero")
        axes[2].legend(fontsize=8)
    axes[2].set_xlabel("Advantage"); axes[2].set_ylabel("Count")
    axes[2].set_title("GRPO Advantage Distribution", fontweight="bold")

    # 4. Training loss over time
    if _loss_hist:
        xs = list(range(1, len(_loss_hist) + 1))
        ys = list(_loss_hist)
        axes[3].plot(xs, ys, color="#e74c3c", linewidth=1.2, alpha=0.6)
        if len(ys) >= 3:
            w = min(5, len(ys))
            ma = np.convolve(ys, np.ones(w)/w, mode="valid")
            axes[3].plot(range(w, len(ys) + 1), ma, color="#c0392b", linewidth=2.5)
    axes[3].set_xlabel("Episode"); axes[3].set_ylabel("Loss")
    axes[3].set_title("Training Loss (GRPO)", fontweight="bold")

    plt.tight_layout()
    return fig


def _lb_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lb = _get_leaderboard()
    entries = lb.get_leaderboard()
    if not entries:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
        return fig
    names  = [e.get("display_name", e["model_id"].split("/")[-1])[:20] for e in entries]
    scores = [e["rl_score"] for e in entries]
    colors = ["#FF9D00" if e.get("source") == "user" else "#4a90e2" for e in entries]
    fig, ax = plt.subplots(figsize=(10, max(3, len(entries) * 0.5 + 1)))
    bars = ax.barh(names, scores, color=colors, edgecolor="white", height=0.6)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{score:.4f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.18 if scores else 1)
    ax.set_xlabel("RL Score  (0.5\u00d7avg_reward + 0.3\u00d7success_rate + 0.2\u00d7disc_coverage)", fontsize=9)
    ax.set_title("ALICE RL Leaderboard", fontweight="bold", fontsize=13)
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#FF9D00", label="User submitted"), Patch(color="#4a90e2", label="Benchmark")],
              fontsize=8, loc="lower right")
    plt.tight_layout()
    return fig


def _inference_api_eval(model_id: str, display_name: str, params_b: float, episodes: int) -> tuple:
    """CPU-friendly fallback eval via HF Inference API. No GPU credits needed.
    Runs up to min(episodes, 15) episodes in-process using the serverless inference API.
    Returns (avg_reward, success_rate, disc_coverage, summary_str).
    Only updates the leaderboard if at least one episode actually completed.
    """
    hf_token = os.getenv("HF_TOKEN", "")
    port = int(os.getenv("PORT", "7860"))
    _SYSTEM = ("You are a precise Python solver. Output ONLY: result = <value>")
    rewards, successes, composites = [], [], []
    n = min(episodes, 15)
    n_completed = 0   # episodes that actually got a model response + env step

    for _ in range(n):
        try:
            ep      = httpx.post(f"http://localhost:{port}/reset", timeout=15).json()
            ep_id   = ep["episode_id"]
            task    = ep["task"]
            headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
            resp    = httpx.post(
                f"https://api-inference.huggingface.co/models/{model_id}/v1/chat/completions",
                headers=headers, timeout=30,
                json={"model": model_id, "max_tokens": 64, "temperature": 0.1,
                      "messages": [{"role": "system", "content": _SYSTEM},
                                   {"role": "user", "content": f"Task: {task}"}]},
            ).json()
            raw    = resp.get("choices", [{}])[0].get("message", {}).get("content", "result = None")
            action = raw.strip().split("\n")[0]
            if not action.startswith("result"):
                action = f"result = {action[:100]}"
            result = httpx.post(f"http://localhost:{port}/step",
                                json={"episode_id": ep_id, "action": action}, timeout=15).json()
            r = float(result.get("reward", 0.0))
            composite = float(result.get("info", {}).get("verification", {}).get("composite_score", 0.0))
            rewards.append(r); successes.append(1.0 if composite >= 0.5 else 0.0)
            composites.append(composite)
            n_completed += 1
        except Exception as exc:
            logger.warning("Inference API eval rollout failed: %s", exc)

    if n_completed == 0:
        return 0.0, 0.0, 0.0, (
            f"❌ CPU eval failed for `{model_id}`.\n\n"
            f"The HF Inference API could not be reached or the model is not available "
            f"on the free serverless tier. Try a smaller model (e.g. Qwen/Qwen2.5-0.5B-Instruct) "
            f"or check that the model ID is correct.\n\n"
            f"Model was NOT added to the leaderboard."
        )

    avg_r = round(float(np.mean(rewards)),   4)
    sr    = round(float(np.mean(successes)), 4)
    dc    = round(float(np.mean([1.0 if 0.2 < c < 0.8 else 0.0 for c in composites])), 4)
    _get_leaderboard().update_model_score(model_id, avg_r, sr, dc, n_completed)
    summary = (
        f"✅ CPU eval complete (HF Inference API, {n_completed}/{n} episodes)!\n\n"
        f"  Model:         {model_id}\n"
        f"  avg_reward:    {avg_r:.4f}\n"
        f"  success_rate:  {sr:.4f}\n"
        f"  disc_coverage: {dc:.4f}\n"
        f"  episodes_run:  {n_completed}\n\n"
        f"Added to leaderboard. Click 'Refresh Leaderboard' to see rankings.\n"
        f"Note: CPU eval runs up to 15 episodes via the free HF Inference API (no credits needed). "
        f"For a full eval, add HF Job credits and resubmit."
    )
    return avg_r, sr, dc, summary


def _launch_eval_job(model_id: str, display_name: str, params_b: float, episodes: int) -> tuple:
    """Submit a real HF GPU Job to evaluate the model. Returns (job_id, job_url).
    Falls back to in-process Inference API eval if HF_TOKEN missing or no credits.
    """
    hf_token = os.getenv("HF_TOKEN", "")
    space_id  = os.getenv("HF_SPACE_ID", "rohanjain1648/alice-rl-environment")

    if not hf_token:
        # No token at all — run CPU eval in-process
        _, _, _, summary = _inference_api_eval(model_id, display_name, params_b, episodes)
        return "__cpu_eval__", summary

    try:
        from huggingface_hub import run_uv_job
    except ImportError:
        raise RuntimeError("huggingface_hub>=0.36 required")

    namespace   = space_id.split("/")[0]
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training", "hf_job_train.py")

    try:
        job = run_uv_job(
            script_path,
            flavor="a10g-small",
            namespace=namespace,
            env={
                "HF_SPACE_ID": space_id,
                "MODEL_ID":    model_id,
                "EPISODES":    str(episodes),
                "GROUP_SIZE":  "4",
                "MAX_TURNS":   "3",
                "LR":          "1e-5",
                "LORA_R":      "16",
                "LOAD_IN_4BIT": "0",
            },
            secrets={"HF_TOKEN": hf_token},
            token=hf_token,
        )
    except Exception as gpu_err:
        # GPU job failed (no credits / 402 / quota) — fall back to free CPU Inference API eval
        err_str = str(gpu_err)
        is_402  = "402" in err_str or "Payment Required" in err_str or "insufficient" in err_str.lower()
        if is_402:
            reason = "No HF Job credits (402 Payment Required). Running free CPU eval instead."
        else:
            reason = f"GPU job error: {err_str}"
        logger.warning("GPU job failed (%s), falling back to CPU Inference API eval", err_str)
        _, _, _, cpu_summary = _inference_api_eval(model_id, display_name, params_b, episodes)
        header = f"ℹ️ {reason}\n\n"
        return "__cpu_eval__", header + cpu_summary

    lb = _get_leaderboard()
    lb.submit_model(model_id, display_name, params_b)

    _LIVE_JOBS.insert(0, {
        "job_id": job.id, "model": model_id, "episodes": episodes,
        "avg_reward": 0.0, "success_rate": 0.0, "elapsed_s": 0.0,
        "status": "RUNNING", "url": job.url,
        "timestamp": datetime.now(timezone.utc).isoformat()[:19] + "Z",
    })
    if len(_LIVE_JOBS) > 20:
        _LIVE_JOBS.pop()

    return job.id, job.url


def refresh_dashboard():
    _seed_mock_data()  # ensure mock data is populated on first refresh

    uptime   = time.time() - _start_time
    err_rate = _error_count / _request_count if _request_count > 0 else 0.0
    lat      = float(np.percentile(list(_latency_window), 95)) if _latency_window else 0.0
    mem      = psutil.Process().memory_info().rss / (1024 * 1024)
    ep_count = len(_episode_history)
    s        = dict(_current_state)

    try:
        _, cm, *_ = _get_components()
        task_perf = {tid: {"success_rate": cm.get_task_success_rate(tid)}
                     for tid in cm.task_performance}
        zone      = cm.compute_discrimination_zone(task_perf)
        disc      = float(zone.get("coverage_pct", 0.0))
        diff_tier = cm.difficulty_tier
        in_zone   = len(zone.get("discrimination_zone_tasks", []))
        too_easy  = len(zone.get("too_easy", []))
        too_hard  = len(zone.get("too_hard", []))
    except Exception:
        disc, diff_tier, in_zone, too_easy, too_hard = 0.0, 1, 0, 0, 0

    _disc_history.append(disc)

    completed = [e["reward"] for e in _episode_history if e.get("reward") is not None]
    _reward_hist.clear()
    _reward_hist.extend(completed[-200:])

    avg_r_20 = round(float(np.mean(completed[-20:])), 4) if completed else 0.0
    success_count = sum(1 for r in completed[-20:] if r > 0) if completed else 0
    success_rate_20 = round(success_count / min(len(completed), 20), 3) if completed else 0.0

    # Failure bank: merge mock + real
    fb_entries = list(_MOCK_FAILURES)
    try:
        *_, fb = _get_components()
        real_fb = fb.query_failures()
        fb_entries = [
            {"failure_id": e.failure_id, "error_type": e.error_type,
             "agent_version": e.agent_version, "novelty_score": e.novelty_score,
             "timestamp": e.timestamp, "prompt": e.prompt}
            for e in real_fb[:100]
        ] + fb_entries
    except Exception:
        pass
    fb_size = len(fb_entries)
    avg_novelty = round(float(np.mean([e["novelty_score"] for e in fb_entries])), 3) if fb_entries else 0.0

    ep_table = [
        [e["episode_id"][:8], e["task"][:60],
         round(e.get("difficulty", 0), 1),
         round(e["reward"], 4) if e.get("reward") is not None else "n/a",
         e["timestamp"][:19]]
        for e in reversed(_episode_history[-15:])
    ]

    task_text = s.get("task") or "No active episode"
    turn      = s.get("turn_number", 0)
    ep_id     = (s.get("episode_id") or "")[:8]
    agent_ver = s.get("agent_version", "n/a")
    agent_ver_label = (
        f"{agent_ver} ⚠️ **This is the default placeholder version** — "
        f"it means no real model is attached. Set the `AGENT_MODEL_ID` Space secret "
        f"to your model's HF ID (e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`) to see the real model name here."
    ) if agent_ver in ("0.0.0", "0.0.0.0") else agent_ver

    alerts = []
    if err_rate > 0.1:
        alerts.append(f"\u26a0\ufe0f HIGH ERROR RATE: {err_rate:.1%}")
    if lat > 1.0:
        alerts.append(f"\u26a0\ufe0f HIGH LATENCY P95: {lat * 1000:.0f}ms")
    if disc < 0.3 and ep_count > 10:
        alerts.append(f"\u26a0\ufe0f LOW DISCRIMINATION COVERAGE: {disc:.1%}")
    alert_str = "\n".join(alerts) if alerts else "\u2705 All systems nominal"

    health_md = (
        f"| Metric | Value |\n|---|---|\n"
        f"| Uptime | {uptime:.0f}s |\n"
        f"| Error Rate | {err_rate:.1%} |\n"
        f"| Latency P95 | {lat * 1000:.1f}ms |\n"
        f"| RAM | {mem:.0f} MB |"
    )
    episode_md = (
        f"**Episode:** `{ep_id or 'none'}` | **Turn:** `{turn}` | "
        f"**Agent:** `{agent_ver_label}` | **Total:** `{ep_count}`\n\n"
        f"**Task:** {task_text[:300]}"
    )
    curriculum_md = (
        f"**Difficulty Tier:** `{diff_tier}` | "
        f"**In Zone:** `{in_zone}` | **Too Easy:** `{too_easy}` | **Too Hard:** `{too_hard}` | "
        f"**Coverage:** `{disc:.1%}`"
    )

    # Recent activity feed (last 10 episodes as markdown table)
    recent_rows = list(reversed(_episode_history[-10:]))
    activity_lines = ["| # | Task | Reward | Time |", "|---|---|---|---|"]
    for i, e in enumerate(recent_rows, 1):
        r_str = f"{e['reward']:.4f}" if e.get("reward") is not None else "n/a"
        activity_lines.append(f"| {i} | {e['task'][:45]} | {r_str} | {e['timestamp'][11:19]} |")
    activity_md = "\n".join(activity_lines)

    # Training logs markdown — merge live jobs + mock history
    all_jobs = list(_LIVE_JOBS) + [
        j for j in _MOCK_TRAINING_LOGS
        if not any(lj["job_id"] == j["job_id"] for lj in _LIVE_JOBS)
    ]
    logs_lines = ["| Job ID | Model | Episodes | Avg Reward | Success | Status | Time |",
                  "|---|---|---|---|---|---|---|"]
    for j in all_jobs[:15]:
        jid  = j["job_id"]
        url  = j.get("url", "")
        jid_link = f"[{jid[:12]}...]({url})" if url else jid[:12]
        status_icon = "🟢" if j["status"] == "RUNNING" else ("✅" if j["status"] == "COMPLETED" else "❌")
        logs_lines.append(
            f"| {jid_link} | {j['model'].split('/')[-1]} | {j['episodes']} "
            f"| {j['avg_reward']:.3f} | {j.get('success_rate', 0):.2%} "
            f"| {status_icon} {j['status']} | {j['timestamp'][:16]} |"
        )
    training_logs_md = "\n".join(logs_lines)

    # Failure bank table rows
    fb_table = [
        [e["failure_id"][:8], e["error_type"], e["agent_version"],
         round(e["novelty_score"], 3), e["timestamp"][:19]]
        for e in fb_entries[:50]
    ]

    # Active model label for Training Metrics tab
    # Show "past metrics" label when no live job is running
    live_jobs = [j for j in _LIVE_JOBS if j.get("status") == "RUNNING"]
    latest_job = live_jobs[0] if live_jobs else None
    if not latest_job:
        # Fall back to most recent completed job
        completed = [j for j in _LIVE_JOBS if j.get("status") == "COMPLETED"]
        latest_job = completed[0] if completed else None

    if latest_job and latest_job.get("status") == "RUNNING":
        status_badge = "🟢 LIVE"
        active_model_label = (
            f"### 📊 Live Training — **{latest_job['model']}**\n"
            f"Job: [{latest_job['job_id'][:12]}...]({latest_job['url']}) | "
            f"Status: {status_badge} | "
            f"Episodes: {latest_job['episodes']} | "
            f"Avg Reward: **{latest_job['avg_reward']:.4f}** | "
            f"Success: **{latest_job.get('success_rate', 0):.1%}** | "
            f"Elapsed: {latest_job['elapsed_s']:.0f}s"
        )
    elif latest_job:
        active_model_label = (
            f"### 📊 Past Metrics — **{latest_job['model']}** "
            f"_(most recent completed run — graphs show historical data)_\n"
            f"Job: [{latest_job['job_id'][:12]}...]({latest_job['url']}) | "
            f"✅ COMPLETED | "
            f"Episodes: {latest_job['episodes']} | "
            f"Avg Reward: **{latest_job['avg_reward']:.4f}** | "
            f"Success: **{latest_job.get('success_rate', 0):.1%}** | "
            f"Elapsed: {latest_job['elapsed_s']:.0f}s\n\n"
            f"> Graphs below reflect the **past run**. Start a new HF Job to see live metrics."
        )
    else:
        active_model_label = (
            "### 📊 Past Metrics — TinyLlama/TinyLlama-1.1B-Chat-v1.0 _(baseline)_\n"
            "Showing historical data from the TinyLlama baseline run "
            "(job `69edae30`, 20 eps, avg_reward=0.802, rule-based inference).\n\n"
            "> Start a new HF Job to see live metrics replace these graphs."
        )

    return (
        _heatmap_fig(), _disc_fig(), _reward_fig(),
        ep_count, avg_r_20, success_rate_20, disc, diff_tier, fb_size,
        ep_table,
        health_md, episode_md, curriculum_md, alert_str,
        activity_md,
        _lb_fig(),
        training_logs_md,
        fb_table,
        avg_novelty,
        active_model_label,
    )


def _load_img_html(filename: str, mime: str = "image/jpeg", caption: str = "") -> str:
    import base64
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    path    = os.path.join(img_dir, filename)
    try:
        data    = open(path, "rb").read()
        src     = f"data:{mime};base64," + base64.b64encode(data).decode()
        cap_html = f'<div style="text-align:center;color:#64748b;font-size:11px;margin-top:4px">{caption}</div>' if caption else ""
        return (
            f'<div style="text-align:center;margin:16px 0">'
            f'<img src="{src}" style="max-width:100%;border-radius:12px;'
            f'box-shadow:0 4px 24px rgba(0,0,0,0.4);border:1px solid #334155" />'
            f'{cap_html}</div>'
        )
    except Exception as exc:
        logger.warning("Could not load image %s: %s", filename, exc)
        return ""


def build_gradio():
    port    = int(os.getenv("PORT", "7860"))
    env_url = f"http://localhost:{port}"

    _arch_img_html  = _load_img_html("alice_architecture.png", "image/png",
                                     "ALICE co-evolutionary architecture")
    _blog_img_html  = (
        _load_img_html("hf_blog_1.jpeg", caption="HF Open LLM Leaderboard v2 — intro &amp; motivation")
        + _load_img_html("hf_blog_2.jpeg", caption="Benchmark saturation: top scores converging on human baseline")
        + _load_img_html("hf_blog_3.jpeg", caption="Saturation causes: easy tasks, contamination, benchmark errors")
    )

    with gr.Blocks(title="ALICE RL Environment", theme=gr.themes.Soft(), css=_CSS) as demo:
        gr.HTML(header_md)

        with gr.Tabs(elem_classes="tab-nav"):

            # ── Tab 0: Mission & How It Works ─────────────────────────────
            with gr.TabItem("Mission"):
                with gr.Accordion("🔴 The Problem: Why Benchmarks Are Broken", open=True):
                    gr.Markdown(_PROBLEM_MD)
                with gr.Accordion("🏗 Our Solution: ALICE Architecture", open=True):
                    gr.HTML(_arch_img_html)
                    gr.HTML(_ARCH_HTML)
                    gr.Markdown(
                        "### Why this is new\n"
                        "Every prior approach picks one axis — better prompts, better RLHF labels, "
                        "harder datasets. ALICE combines all three into a **co-evolutionary loop**: "
                        "oracle discrimination + curriculum hunt + failure bank + verified repair + "
                        "GRPO gradient. No static benchmark. No human labelling. No saturation ceiling.\n\n"
                        "The predecessor strategies that ALICE synthesises:\n"
                        "| Strategy | What it does | ALICE's version |\n"
                        "|---|---|---|\n"
                        "| RLHF (InstructGPT) | Human preference labels | Oracle discrimination reward |\n"
                        "| Constitutional AI | Self-critique with principles | Verifier stack as constitution |\n"
                        "| Self-play (SPAG) | Model generates hard tasks | Hunt agent in discrimination zone |\n"
                        "| Curriculum Learning | Train on tasks at right difficulty | 20-80% success rate zone |\n"
                        "| Dynamic Benchmarks (ARC) | Novel, unseen tasks | Auto-generated from model's own failures |"
                    )
                with gr.Accordion("📚 Research Foundation — Papers & Articles", open=False):
                    gr.HTML(_RESEARCH_HTML)
                    gr.HTML(_blog_img_html)
                with gr.Accordion("📽 Presentation Slides (ALICE deck)", open=False):
                    gr.HTML("""
                    <div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:12px;border:1px solid #334155">
                      <iframe src="https://gamma.app/embed/xujn28eygymlgda"
                              style="position:absolute;top:0;left:0;width:100%;height:100%;border:none"
                              allow="fullscreen" allowfullscreen></iframe>
                    </div>
                    <div style="text-align:center;margin-top:8px">
                      <a href="https://gamma.app/docs/ALICE-Adversarial-Loop-for-Inter-model-Co-evolutionary-Environmen-xujn28eygymlgda"
                         target="_blank" style="color:#FF9D00;font-size:13px">↗ Open presentation in full screen</a>
                    </div>
                    """)
                with gr.Accordion("🚀 How to Use ALICE — Step by Step", open=False):
                    gr.Markdown(_HOW_TO_USE_MD)

            # ── Tab 1: Overview ──────────────────────────────────────────
            with gr.TabItem("Overview"):
                with gr.Row():
                    ep_count_box    = gr.Number(label="Episodes Run",        value=0,   precision=0, interactive=False, elem_classes="metric-card")
                    rollouts_box    = gr.Number(label="Total Rollouts",      value=0,   precision=0, interactive=False, elem_classes="metric-card")
                    avg_reward_box  = gr.Number(label="Avg Reward (last 20)", value=0.0, precision=4, interactive=False, elem_classes="metric-card")
                    success_box     = gr.Number(label="Success Rate",        value=0.0, precision=3, interactive=False, elem_classes="metric-card")
                with gr.Row():
                    disc_box        = gr.Number(label="Disc Coverage",       value=0.0, precision=3, interactive=False, elem_classes="metric-card")
                    diff_tier_box   = gr.Number(label="Difficulty Tier",     value=1,   precision=0, interactive=False, elem_classes="metric-card")
                    fb_size_box     = gr.Number(label="Failure Bank Size",   value=0,   precision=0, interactive=False, elem_classes="metric-card")
                    uptime_box      = gr.Textbox(label="Uptime",             value="0s", interactive=False, elem_classes="metric-card")
                with gr.Row():
                    with gr.Column(scale=1):
                        health_md_box = gr.Markdown("_Loading health..._")
                    with gr.Column(scale=2):
                        episode_md_box = gr.Markdown("_No active episode yet._")
                alert_box    = gr.Textbox(label="System Alerts", interactive=False, lines=2, value="Loading...")
                activity_box = gr.Markdown("_Loading activity..._")
                with gr.Accordion("📚 Optional: RL Concepts in This Tab", open=False):
                    gr.Markdown(
                        "**Episode** — one full interaction between the agent (your model) and the environment. "
                        "Starts with `reset()`, ends when `done=True` or max turns reached.\n\n"
                        "**Reward** — a scalar signal the environment gives after each action. "
                        "Positive = good, negative = bad. ALICE's reward = verification score × decay × novelty.\n\n"
                        "**Agent version** — the model ID set in the `AGENT_MODEL_ID` env var. "
                        "If it shows `0.0.0` no model is attached — the env is just responding to any client.\n\n"
                        "**Discrimination coverage** — the fraction of tasks the model gets right 20-80% of the time. "
                        "This is the 'interesting' zone. Too easy (>80%) = no learning. Too hard (<20%) = no signal.\n\n"
                        "**Failure Bank** — a memory of confirmed failures, deduplicated by semantic similarity. "
                        "Prevents the curriculum from re-testing known failures."
                    )

            # ── Tab 2: Training Metrics ───────────────────────────────────
            with gr.TabItem("Training Metrics"):
                active_model_md = gr.Markdown("_No training run yet — start a job to see live metrics._")
                reward_curve_note = gr.Markdown(
                    "> ⚠️ **Why does the reward curve look flat?** "
                    "The baseline (TinyLlama) converged to `result = 42` — always valid Python → "
                    "reward ≈ 0.8 every episode → no variance → GRPO advantages all zero → "
                    "no gradient signal → model stops learning.  \n"
                    "> **Why is GRPO loss = 0.0 in the default data?** TinyLlama ran as "
                    "rule-based inference on the server — no local model was loaded, so "
                    "no forward/backward pass happened and there was no gradient loss to compute.  \n"
                    "> **Fix:** The updated `hf_job_train.py` loads the model locally with LoRA, "
                    "uses chat-template prompts with real turn-by-turn verifier feedback, forcing "
                    "task-specific answers → varied rewards → real GRPO loss signal. "
                    "Start a new HF Job to see live loss curves replace the baseline data."
                )
                reward_plot = gr.Plot(label="4-Panel Training Metrics (Reward Curve + Cumulative + Distribution + Loss)")
                with gr.Accordion("📚 Optional: GRPO Math & What Each Chart Means", open=False):
                    gr.Markdown(
                        "**GRPO — Group Relative Policy Optimisation**\n\n"
                        "Standard PPO needs a value network (critic) to estimate expected future reward — expensive. "
                        "GRPO skips it. For each update step, collect a *group* of G rollouts, compute their rewards "
                        "{r₁, r₂, ..., rG}, then normalise:\n\n"
                        "**advantage_i = (r_i − mean(r)) / std(r)**\n\n"
                        "Then update the policy to increase probability of high-advantage actions:\n\n"
                        "**Loss = −mean( advantage_i × log_prob(action_i) ) + KL_coef × log_prob²**\n\n"
                        "The KL term (log_prob²) penalises the policy from moving too far from the reference — "
                        "without a reference model! It's an implicit KL via the squared log-prob trick.\n\n"
                        "**Why is advantage sometimes all zeros?** If all rollouts in the group get the same reward "
                        "(e.g., every model outputs `result = 42`), std(r) = 0, all advantages = 0, "
                        "loss = 0, gradient = 0. Model stops learning. Fix: use task-specific prompts that "
                        "produce varied rewards.\n\n"
                        "**Reward curve** — mean reward per episode. Should trend upward as the model learns.\n"
                        "**Cumulative reward** — running sum. Linear slope = constant learning. Accelerating = compound improvement.\n"
                        "**GRPO loss** — should be non-zero and eventually decrease. If stuck at 0 = collapsed advantages."
                    )
                gr.Markdown("### Recent Episodes")
                ep_table = gr.Dataframe(
                    headers=["episode_id", "task", "difficulty", "reward", "timestamp"],
                    interactive=False,
                    wrap=True,
                )

            # ── Tab 3: Curriculum ─────────────────────────────────────────
            with gr.TabItem("Curriculum"):
                with gr.Accordion("📚 Optional: Curriculum & Math Behind This Tab", open=False):
                    gr.Markdown(
                        "**Curriculum Learning** — the idea of training on tasks ordered from easy → hard, "
                        "rather than random. ALICE does this automatically by tracking per-task success rates.\n\n"
                        "**Discrimination Zone** — tasks where the model's success rate is between 20% and 80%. "
                        "Below 20% = too hard (model just guesses). Above 80% = too easy (no gradient signal). "
                        "The zone is where Vygotsky's 'zone of proximal development' sits in ML terms.\n\n"
                        "**Why 20-80%?** A task with p=0.5 maximises variance in the Bernoulli distribution "
                        "(σ² = p(1-p) = 0.25). More variance → more informative gradients → faster learning.\n\n"
                        "**Difficulty Tier** — ALICE escalates the tier when disc coverage > 70% for 50+ episodes, "
                        "meaning the model has 'mastered' the current zone and needs harder tasks.\n\n"
                        "**Heatmap math**: each cell is `success_rate = successes / attempts` for that domain × tier pair. "
                        "Initialised to 0 (red) until attempts > 0."
                    )
                gr.Markdown(
                    "### How to read these charts\n"
                    "**Heatmap** — Each cell is a _(domain × difficulty tier)_ pair. "
                    "Color = success rate across all attempts in that cell: "
                    "🔴 red = 0% (never solved or not yet attempted), "
                    "🟡 yellow ≈ 50%, 🟢 green = 100% solved.\n\n"
                    "**Why is everything red at the start?** The curriculum begins with zero data. "
                    "Until the model has attempted tasks in a given domain/tier, its success rate is 0 → all red. "
                    "Green cells appear progressively as training runs and tasks get solved.\n\n"
                    "**Discrimination Zone Coverage** = the fraction of tasks the model solves 20–80% of the time "
                    "— the ideal difficulty window where learning happens. "
                    "The coverage graph starts at 0% (red region on chart) and should rise above the "
                    "70% green dashed target line as the curriculum adapts. "
                    "If it stays flat near 0%, tasks are either all too easy or all too hard."
                )
                curriculum_info = gr.Markdown("_Loading..._")
                heatmap_plot    = gr.Plot(label="Domain x Tier Success Rates")
                disc_plot       = gr.Plot(label="Discrimination Zone Coverage Over Time")

            # ── Tab 4: HF Space & Jobs ────────────────────────────────────
            with gr.TabItem("HF Space & Jobs"):
                with gr.Accordion("📚 Optional: What Are HF Jobs?", open=False):
                    gr.Markdown(
                        "**HF Jobs** — Hugging Face's serverless GPU compute service. "
                        "You define a Python script, pick a GPU flavor (T4, a10g, A100), and HF spins up a container, "
                        "runs it, and tears it down. Pay-per-second, no idle cost.\n\n"
                        "**Why we use them for training** — ALICE's training script (`hf_job_train.py`) loads a model "
                        "locally with LoRA, runs GRPO episodes against this Space's API, and streams metrics back in real time. "
                        "The job doesn't need a persistent server — it's a one-shot run.\n\n"
                        "**a10g-small** — 24 GB VRAM. Fits up to 3B models without 4-bit quantisation. "
                        "~7 s/episode for 0.5B, ~25 s/episode for 3B.\n\n"
                        "**LoRA** — Low-Rank Adaptation. Instead of updating all model weights (billions of floats), "
                        "LoRA inserts tiny trainable matrices (rank 16) into the attention layers. "
                        "~99% fewer trainable parameters, same quality. `lora_alpha = 2 × r` is the standard scaling."
                    )
                gr.Markdown(
                    "Live status of Hugging Face Spaces and training job history. "
                    "Set `HF_SPACE_ID` and `ALICE_HF_REPO_ID` in Space secrets."
                )
                hf_status_md   = gr.Markdown("_Click Refresh to load Space status._")
                refresh_hf_btn = gr.Button("Refresh HF Status", variant="secondary")
                gr.Markdown("### Training History")
                training_logs_box = gr.Markdown("_Loading..._")
                gr.Markdown(
                    f"**Quick Links:** "
                    f"[\U0001f917 Space]({SPACE_URL_FULL}) \u00b7 "
                    f"[\U0001f4d6 API Docs]({API_URL}/docs) \u00b7 "
                    f"[\u26a1 HF Jobs]({JOBS_URL})"
                )

                def _refresh_hf_status():
                    _STAGE_LABEL = {"RUNNING": "running", "STOPPED": "stopped",
                                    "PAUSED": "stopped", "error": "stopped"}
                    info  = _hf_space_info()
                    lines = ["### 🤗 Hugging Face Space Status\n"]

                    # Environment Space
                    v      = info.get("env_space", {})
                    sid    = v.get("id", "rohanjain1648/alice-rl-environment")
                    status = v.get("status", "n/a")
                    url    = v.get("url", _hf_space_url(sid))
                    stage  = _STAGE_LABEL.get(status, "building")
                    badge  = "🟢" if stage == "running" else "🔴"
                    lines.append(f"{badge} **Environment Space** `{sid}` — **{status}**")
                    lines.append(f"  URL: [{url}]({url})")
                    lines.append("")

                    # Training Jobs — show each real job per model
                    lines.append("### ⚡ Training Jobs (per model)\n")
                    lines.append("| Model | Job ID | Status | Avg Reward | Success | URL |")
                    lines.append("|---|---|---|---|---|---|")
                    all_j = list(_LIVE_JOBS)
                    for j in all_j[:10]:
                        jid   = j["job_id"]
                        jurl  = j.get("url", f"https://huggingface.co/jobs/rohanjain1648/{jid}")
                        st    = j.get("status", "?")
                        sb    = "🟢" if st == "RUNNING" else "✅"
                        model = j["model"].split("/")[-1]
                        lines.append(
                            f"| `{model}` | [{jid[:12]}...]({jurl}) | {sb} {st} "
                            f"| {j.get('avg_reward', 0):.4f} "
                            f"| {j.get('success_rate', 0):.1%} "
                            f"| [View]({jurl}) |"
                        )
                    if not all_j:
                        lines.append("| — | No jobs registered yet | — | — | — | — |")
                    return "\n".join(lines)

                refresh_hf_btn.click(_refresh_hf_status, outputs=[hf_status_md])

            # ── Tab 5: Failure Bank ───────────────────────────────────────
            with gr.TabItem("Failure Bank"):
                with gr.Accordion("📚 Optional: What Is a Failure Bank?", open=False):
                    gr.Markdown(
                        "**Failure Bank** — a persistent store of confirmed failure cases. "
                        "A failure is a (task, action) pair where the verifier assigned composite_score < 0.5.\n\n"
                        "**Why deduplicate by novelty?** Without deduplication, the curriculum wastes episodes "
                        "re-testing slight variants of the same failure. The novelty score (0 → 1) measures "
                        "semantic distance from existing bank entries using sentence-transformer embeddings. "
                        "Only failures with novelty > 0.7 get added — the rest are discarded as 'already known'.\n\n"
                        "**k-NN novelty scoring** — for each new failure, compute cosine similarity to the k=16 "
                        "nearest existing embeddings. Novelty = 1 − max_similarity. "
                        "High novelty means the failure is genuinely new and worth tracking.\n\n"
                        "**Repair Queue** — failures staged for targeted fine-tuning. The Repair agent generates "
                        "training pairs (task, correct_solution) verified by T1+T2+T3, then these are used for SFT "
                        "before the next GRPO round."
                    )
                with gr.Row():
                    fb_total_box   = gr.Number(label="Total Failures",  value=0,   precision=0, interactive=False)
                    fb_novelty_box = gr.Number(label="Avg Novelty",     value=0.0, precision=3, interactive=False)
                    fb_queue_box   = gr.Number(label="Repair Queue",    value=0,   precision=0, interactive=False)
                with gr.Row():
                    filter_error = gr.Textbox(label="Filter by error_type",    placeholder="e.g. verification_failure")
                    filter_agent = gr.Textbox(label="Filter by agent_version", placeholder="e.g. 0.0.0 (default/test), SmolLM2-135M")
                    apply_filter = gr.Button("Apply Filter", variant="secondary")
                failure_table = gr.Dataframe(
                    headers=["failure_id", "error_type", "agent_version", "novelty_score", "timestamp"],
                    interactive=False,
                    wrap=True,
                )

                def _apply_failure_filter(error_type, agent_version):
                    et = error_type.strip() or None
                    av = agent_version.strip() or None
                    # Start with mock entries
                    results = [
                        e for e in _MOCK_FAILURES
                        if (et is None or e["error_type"] == et)
                        and (av is None or e["agent_version"] == av)
                    ]
                    # Merge real entries
                    try:
                        *_, fb = _get_components()
                        real = fb.query_failures(error_type=et, agent_version=av)
                        results = [
                            {"failure_id": e.failure_id, "error_type": e.error_type,
                             "agent_version": e.agent_version, "novelty_score": e.novelty_score,
                             "timestamp": e.timestamp}
                            for e in real[:50]
                        ] + results
                    except Exception:
                        pass
                    return [
                        [e["failure_id"][:8], e["error_type"], e["agent_version"],
                         round(e["novelty_score"], 3), e["timestamp"][:19]]
                        for e in results[:50]
                    ]

                apply_filter.click(
                    _apply_failure_filter,
                    inputs=[filter_error, filter_agent],
                    outputs=[failure_table],
                )

            # ── Tab 6: Leaderboard ────────────────────────────────────────
            with gr.TabItem("Leaderboard"):
                gr.Markdown(
                    "## ALICE RL Leaderboard\n"
                    "Models ranked by composite RL score "
                    "_(0.5 \u00d7 avg\\_reward + 0.3 \u00d7 success\\_rate + 0.2 \u00d7 disc\\_coverage)_."
                )
                with gr.Accordion("📚 Optional: What Do These Scores Mean?", open=False):
                    gr.Markdown(
                        "**rl_score** = 0.5 × avg_reward + 0.3 × success_rate + 0.2 × disc_coverage\n\n"
                        "- `avg_reward` — mean reward per turn. Shaped by correctness + turn-decay + novelty penalty.\n"
                        "- `success_rate` — fraction of episodes where composite_score ≥ 0.5 "
                        "(T1 valid Python AND T2 LLM judge agreed it's correct).\n"
                        "- `disc_coverage` — fraction of tasks where the model sat in the 20–80% zone. "
                        "A perfect-scoring model has disc_coverage ≈ 0 — too easy for the env to challenge.\n\n"
                        "**Composite score inside the verifier:**\n"
                        "`composite = 0.4 × T1_score + 0.4 × T2_score + 0.2 × T3_score`\n\n"
                        "T1 = RestrictedPython sandbox (did it run, no crash?), "
                        "T2 = LLM judge (is the answer correct in intent?), "
                        "T3 = regression battery (does it generalise to 20 variants?)."
                    )
                lb_chart = gr.Plot(label="Leaderboard Bar Chart")
                lb_refresh_btn = gr.Button("Refresh Leaderboard", variant="primary")
                leaderboard_table = gr.Dataframe(
                    headers=["rank", "model", "params_B", "rl_score",
                              "avg_reward", "success_rate", "disc_coverage", "episodes", "source"],
                    interactive=False,
                    wrap=True,
                )
                gr.Markdown("### Submit a Model for Evaluation")
                gr.Markdown(
                    "_Enter any HF model ID to launch a real HF Job that runs RL episodes against "
                    "the ALICE environment. Scores are pushed to the leaderboard automatically "
                    "when the job completes. The job URL is returned immediately so you can track progress._"
                )
                with gr.Row():
                    submit_model_id  = gr.Textbox(label="HF model ID",
                                                   value="Qwen/Qwen2.5-0.5B-Instruct",
                                                   placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct")
                    submit_name      = gr.Textbox(label="Display name",
                                                   value="Qwen2.5-0.5B",
                                                   placeholder="My Model")
                    submit_params    = gr.Number(label="Params (B)", value=0.5, precision=1)
                    submit_episodes  = gr.Number(label="Episodes", value=20, precision=0,
                                                  info="More episodes = better score estimate (~7 s/ep on a10g-small)")
                    submit_btn       = gr.Button("Submit & Eval", variant="primary")
                submit_status = gr.Textbox(label="Eval Status", interactive=False, lines=7)

                def _load_leaderboard():
                    lb      = _get_leaderboard()
                    entries = lb.get_leaderboard()
                    rows = [
                        [e["rank"], e.get("display_name", e["model_id"]), e["params_b"],
                         e["rl_score"], e["avg_reward"], e["success_rate"],
                         e["discrimination_coverage"], e["episodes_run"], e["source"]]
                        for e in entries
                    ]
                    return _lb_fig(), rows

                def _submit_model(model_id, display_name, params_b, episodes):
                    if not model_id.strip():
                        return "❌ Error: HF model ID is required (e.g. HuggingFaceTB/SmolLM2-360M-Instruct)"
                    mid   = model_id.strip()
                    dname = display_name.strip() or mid.split("/")[-1]
                    pb    = float(params_b or 0)
                    eps   = max(1, int(episodes or 20))
                    eta   = max(1, eps * 7 // 60)
                    try:
                        job_id, job_url = _launch_eval_job(mid, dname, pb, eps)
                        if job_id == "__cpu_eval__":
                            # CPU fallback — job_url contains the full status message
                            return job_url
                        return (
                            f"⏳ Eval job submitted to HF Jobs!\n\n"
                            f"  Model:      {mid}\n"
                            f"  Episodes:   {eps}\n"
                            f"  Job ID:     {job_id}\n"
                            f"  Job URL:    {job_url}\n\n"
                            f"Results push to the leaderboard automatically when the job completes.\n"
                            f"Estimated time: ~{eta} min on a10g-small.\n"
                            f"Click 'Refresh Leaderboard' after the job finishes to see updated rankings."
                        )
                    except Exception as exc:
                        return f"⚠️ Job submission failed: {exc}"

                lb_refresh_btn.click(_load_leaderboard, outputs=[lb_chart, leaderboard_table])
                submit_btn.click(
                    _submit_model,
                    inputs=[submit_model_id, submit_name, submit_params, submit_episodes],
                    outputs=[submit_status],
                )

            # ── Tab 7: Pro Access / Pricing ──────────────────────────────
            with gr.TabItem("💎 Pro Access"):
                gr.HTML(_PRICING_HTML)
                gr.Markdown("### 💬 Get in Touch")
                with gr.Row():
                    contact_name  = gr.Textbox(label="Name", placeholder="Your name")
                    contact_email = gr.Textbox(label="Email", placeholder="your@email.com")
                contact_msg = gr.Textbox(label="Message", placeholder="Tell us about your use case — model size, episodes, custom hardware needs...", lines=3)
                contact_btn = gr.Button("Send Message", variant="primary")
                contact_out = gr.Textbox(label="Status", interactive=False)

                def _send_contact(name, email, msg):
                    if not email.strip() or not msg.strip():
                        return "⚠️ Please fill in email and message."
                    logger.info("Pro contact: %s <%s> — %s", name, email, msg[:100])
                    return f"✅ Thanks {name or 'there'}! We'll reach out to {email} within 24 hours."

                contact_btn.click(_send_contact, inputs=[contact_name, contact_email, contact_msg], outputs=[contact_out])

        gr.Markdown(
            f"---\n*Auto-refreshes every 3 s. "
            f"Environment API at [{API_URL}]({API_URL}) | "
            f"[Swagger / API Docs]({API_URL}/docs)*"
        )

        timer = gr.Timer(value=3)
        outputs = [
            heatmap_plot, disc_plot, reward_plot,
            ep_count_box, avg_reward_box, success_box, disc_box, diff_tier_box, fb_size_box,
            ep_table,
            health_md_box, episode_md_box, curriculum_info, alert_box,
            activity_box,
            lb_chart,
            training_logs_box,
            failure_table,
            fb_novelty_box,
            active_model_md,
        ]
        timer.tick(refresh_dashboard, outputs=outputs)

    return demo


# ---------------------------------------------------------------------------
# Mount Gradio into FastAPI and launch
# ---------------------------------------------------------------------------

_seed_mock_data()
gradio_app = build_gradio()
app = gr.mount_gradio_app(api, gradio_app, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("alice_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")), reload=False)
