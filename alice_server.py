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


@api.on_event("startup")
async def _startup():
    """Pre-warm is disabled on Space — 2GB RAM is too tight for sentence-transformer at startup.
    The model loads lazily on first /step call instead."""
    logger.info("Server startup complete (lazy component init enabled)")
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


# ---------------------------------------------------------------------------
# Gradio dashboard
# ---------------------------------------------------------------------------

_disc_history: list = []
_reward_hist: list = []


def _heatmap_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    try:
        _, cm, *_ = _get_components()
        full = cm.get_curriculum_heatmap()   # (5, 10) real success rates
        data = full[:, :5]                   # show first 5 tiers to fit the display
    except Exception:
        data = np.zeros((5, 5), dtype=np.float32)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Tier {i+1}" for i in range(5)], fontsize=8)
    ax.set_yticks(range(5))
    ax.set_yticklabels(["arithmetic", "logic", "factual", "symbolic", "code"], fontsize=8)
    ax.set_title("Curriculum Heatmap — real success rates per domain/tier", fontsize=9)
    fig.colorbar(im, ax=ax, label="Success Rate")
    plt.tight_layout()
    return fig


def _disc_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 2.5))
    if _disc_history:
        ax.plot(range(len(_disc_history)), _disc_history, "b-o", markersize=3, linewidth=1.5)
    ax.axhline(0.3, color="r", linestyle="--", alpha=0.5, label="Min 30%")
    ax.axhline(0.7, color="g", linestyle="--", alpha=0.5, label="Max 70%")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Coverage")
    ax.set_title("Discrimination Zone Coverage")
    ax.legend(fontsize=7)
    plt.tight_layout()
    return fig


def _reward_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 2.5))
    if _reward_hist:
        ax.hist(_reward_hist, bins=min(20, len(_reward_hist)), color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution")
    plt.tight_layout()
    return fig


def refresh_dashboard():
    import httpx as _httpx
    try:
        h = _httpx.get("http://localhost:7860/health", timeout=2).json()
    except Exception:
        h = {}
    try:
        s = _httpx.get("http://localhost:7860/state", timeout=2).json()
    except Exception:
        s = {}

    uptime = h.get("uptime", 0.0)
    err_rate = h.get("error_rate", 0.0)
    lat = h.get("latency_p95", 0.0)
    mem = h.get("memory_usage", 0.0)

    ep_count = len(_episode_history)

    # Real discrimination zone coverage from CurriculumManager
    try:
        _, cm, *_ = _get_components()
        task_perf = {tid: {"success_rate": cm.get_task_success_rate(tid)}
                     for tid in cm.task_performance}
        zone = cm.compute_discrimination_zone(task_perf)
        disc = float(zone.get("coverage_pct", 0.0))
    except Exception:
        disc = 0.0
    _disc_history.append(disc)
    if len(_disc_history) > 200:
        _disc_history.pop(0)

    # Real rewards from completed episodes (episodes where done=True)
    completed = [e["reward"] for e in _episode_history if e.get("reward") is not None]
    _reward_hist.clear()
    _reward_hist.extend(completed[-200:])

    ep_table = [[e["episode_id"][:8], e["task"][:50], e.get("difficulty", 0), e["timestamp"][:19]] for e in reversed(_episode_history[-10:])]

    task_text = s.get("task") or "No active episode — call POST /reset to start"
    turn = s.get("turn_number", 0)
    ep_id = (s.get("episode_id") or "")[:8]

    alerts = []
    if err_rate > 0.1:
        alerts.append(f"⚠️ High error rate: {err_rate:.1%}")
    if lat > 1.0:
        alerts.append(f"⚠️ High latency P95: {lat:.3f}s")
    alert_str = "\n".join(alerts) if alerts else "✅ All systems nominal"

    health_str = f"⏱ Uptime: {uptime:.0f}s  |  ❌ Error rate: {err_rate:.1%}  |  ⚡ P95: {lat*1000:.1f}ms  |  🧠 RAM: {mem:.0f}MB"

    return (
        _heatmap_fig(), _disc_fig(), _reward_fig(),
        ep_count, health_str,
        ep_table,
        f"Episode: {ep_id or 'none'}  |  Turn: {turn}\n\nTask: {task_text}",
        alert_str,
    )


def build_gradio():
    with gr.Blocks(title="ALICE RL Environment") as demo:
        gr.Markdown("""
# 🔬 ALICE — Adversarial Loop for Inter-model Co-evolutionary Environment
**Live environment monitor** · API at `/reset`, `/step`, `/state`, `/health` · [API Docs](/docs)
""")

        with gr.Row():
            health_box = gr.Textbox(label="System Health", interactive=False, value="Loading...")
            ep_count_box = gr.Number(label="Episodes Run", value=0, precision=0)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Curriculum Heatmap")
                heatmap_plot = gr.Plot()
            with gr.Column():
                gr.Markdown("### Discrimination Zone Coverage")
                disc_plot = gr.Plot()

        with gr.Row():
            gr.Markdown("### Reward Distribution")
            reward_plot = gr.Plot()

        with gr.Row():
            gr.Markdown("### Recent Episodes")
            ep_table = gr.Dataframe(
                headers=["episode_id", "task (truncated)", "difficulty", "timestamp"],
                interactive=False,
            )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Current Episode State")
                state_box = gr.Textbox(label="", lines=4, interactive=False, value="No active episode")
            with gr.Column():
                gr.Markdown("### Alerts")
                alerts_box = gr.Textbox(label="", lines=4, interactive=False, value="Loading...")

        gr.Markdown("---\n*Auto-refreshes every 3 seconds. Start an episode by calling `POST /reset`.*")

        outputs = [heatmap_plot, disc_plot, reward_plot, ep_count_box, health_box, ep_table, state_box, alerts_box]
        timer = gr.Timer(value=3)
        timer.tick(refresh_dashboard, outputs=outputs)

    return demo


# ---------------------------------------------------------------------------
# Mount Gradio into FastAPI and launch
# ---------------------------------------------------------------------------

gradio_app = build_gradio()
app = gr.mount_gradio_app(api, gradio_app, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("alice_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")), reload=False)
