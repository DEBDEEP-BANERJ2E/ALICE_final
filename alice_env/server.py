"""
ALICE OpenEnv Server — FastAPI server with OpenEnv-compliant endpoints.

Endpoints:
  POST /reset  — initialize episode, return {episode_id, timestamp, task, agent_version}
  POST /step   — process action, return (state, reward, done, info)
  GET  /state  — return current state without side effects
  GET  /health — return {uptime, error_rate, latency_p95, memory_usage}
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional

import psutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, field_validator

# Load .env before anything else
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("alice.server")

# ---------------------------------------------------------------------------
# Lazy component imports (heavy deps loaded only when first request arrives)
# ---------------------------------------------------------------------------

_episode_handler: Any = None
_curriculum_manager: Any = None
_task_generator: Any = None
_reward_function: Any = None
_verifier_stack: Any = None
_failure_bank: Any = None


def _get_components() -> tuple:
    """Return (episode_handler, curriculum_manager, task_generator,
    reward_function, verifier_stack, failure_bank), initializing lazily."""
    global _episode_handler, _curriculum_manager, _task_generator
    global _reward_function, _verifier_stack, _failure_bank

    if _episode_handler is None:
        # These imports are deferred so uvicorn starts instantly
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

    return (
        _episode_handler,
        _curriculum_manager,
        _task_generator,
        _reward_function,
        _verifier_stack,
        _failure_bank,
    )


# ---------------------------------------------------------------------------
# App & startup time
# ---------------------------------------------------------------------------

app = FastAPI(title="ALICE OpenEnv Server", version="0.1.0")
_start_time = time.time()

# ---------------------------------------------------------------------------
# Thread-safe state
# ---------------------------------------------------------------------------

_state_lock = asyncio.Lock()

_current_state: Dict[str, Any] = {
    "episode_id": None,
    "turn_number": 0,
    "task": None,
    "agent_version": os.getenv("AGENT_MODEL_ID", "0.0.0"),
}

# ---------------------------------------------------------------------------
# Latency tracking for p95
# ---------------------------------------------------------------------------

_latency_window: Deque[float] = deque(maxlen=200)
_error_count: int = 0
_request_count: int = 0


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

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
        logger.info(
            "%s %s %d %.3fs",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response


app.add_middleware(RequestLoggingMiddleware)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


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
    episode_id: Optional[str]
    turn_number: int
    task: Optional[str]
    agent_version: Optional[str]


class HealthResponse(BaseModel):
    uptime: float
    error_rate: float
    latency_p95: float
    memory_usage: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/reset", response_model=ResetResponse)
async def reset() -> ResetResponse:
    """Initialize a new episode and return the initial state."""
    episode_handler, curriculum_manager, task_generator, _, _, _ = _get_components()

    episode_id = str(uuid.uuid4())
    agent_version = os.getenv("AGENT_MODEL_ID", "0.0.0")

    # Build discrimination zone from current curriculum state
    task_perf = {
        tid: {"success_rate": curriculum_manager.get_task_success_rate(tid)}
        for tid in curriculum_manager.task_performance
    }
    zone_result = curriculum_manager.compute_discrimination_zone(task_perf)
    discrimination_zone = zone_result.get("discrimination_zone_tasks", [])

    # Generate task via hunt mode
    task_info = task_generator.hunt_mode(
        agent_performance=task_perf,
        discrimination_zone=discrimination_zone,
    )
    task = task_info["prompt"]

    # Initialize episode
    initial_state = episode_handler.initialize_episode(
        episode_id=episode_id,
        agent_version=agent_version,
        task=task,
    )

    async with _state_lock:
        _current_state.update({
            "episode_id": episode_id,
            "turn_number": 1,
            "task": task,
            "agent_version": agent_version,
        })

    logger.info("Episode %s initialized (task difficulty=%.1f)", episode_id[:8], task_info["difficulty_score"])

    return ResetResponse(
        episode_id=episode_id,
        timestamp=initial_state["timestamp"],
        task=task,
        agent_version=agent_version,
    )


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest) -> StepResponse:
    """Process an agent action and return (state, reward, done, info)."""
    episode_handler, _, _, reward_function, verifier_stack, _ = _get_components()

    async with _state_lock:
        current_episode_id = _current_state.get("episode_id")

    if current_episode_id != request.episode_id:
        raise HTTPException(status_code=400, detail="Invalid episode_id")

    # Delegate to episode handler
    state, raw_reward, done, info = episode_handler.step(request.action)

    # Compute shaped reward using verifier output if available
    verification = verifier_stack.verify(request.action, task=_current_state.get("task", ""))
    turn_number = info.get("turn", 1)
    episode_data = {
        "turns": [{
            "turn_number": turn_number,
            "action": request.action,
            "verification": verification,
            "task_in_failure_bank": False,
            "times_task_attempted": turn_number,
            "total_tasks": 3,
            "prev_action": "",
            "discrimination_coverage_before": 0.0,
            "discrimination_coverage_after": 0.0,
        }]
    }
    reward_result = reward_function.compute_reward(episode_data)
    reward = reward_result["shaped_rewards"][0] if reward_result["shaped_rewards"] else raw_reward

    async with _state_lock:
        _current_state["turn_number"] = state.get("turn_number", turn_number + 1)

    return StepResponse(
        state=state,
        reward=reward,
        done=done,
        info={**info, "verification": verification},
    )


@app.get("/state", response_model=StateResponse)
async def get_state() -> StateResponse:
    """Return current environment state without side effects."""
    async with _state_lock:
        snapshot = dict(_current_state)

    return StateResponse(
        episode_id=snapshot.get("episode_id"),
        turn_number=snapshot.get("turn_number", 0),
        task=snapshot.get("task"),
        agent_version=snapshot.get("agent_version"),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return system health metrics."""
    uptime = time.time() - _start_time
    error_rate = _error_count / _request_count if _request_count > 0 else 0.0

    if _latency_window:
        import numpy as np
        latency_p95 = float(np.percentile(list(_latency_window), 95))
    else:
        latency_p95 = 0.0

    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 * 1024)  # MB

    return HealthResponse(
        uptime=uptime,
        error_rate=error_rate,
        latency_p95=latency_p95,
        memory_usage=memory_usage,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
