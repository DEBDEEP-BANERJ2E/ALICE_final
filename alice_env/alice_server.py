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
# Gradio dashboard — tabbed analytics
# ---------------------------------------------------------------------------

_disc_history: Deque[float] = deque(maxlen=200)
_reward_hist:  Deque[float] = deque(maxlen=200)

_CSS = """
.tab-nav button { font-size: 14px; font-weight: 600; }
"""


def _hf_space_url(space_id: str) -> str:
    """Convert 'user/space-name' to its HF Space URL."""
    parts = space_id.split("/")
    return f"https://{parts[0]}-{parts[1]}.hf.space" if len(parts) == 2 else ""


def _hf_space_info() -> dict:
    """Query HF API for Space runtime status for env and training spaces."""
    env_sid   = os.getenv("HF_SPACE_ID", "")
    train_sid = os.getenv("ALICE_HF_REPO_ID", "")
    hf_token  = os.getenv("HF_TOKEN", "")
    headers   = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    result: dict = {}
    for key, sid in [("env_space", env_sid), ("training_space", train_sid)]:
        if not sid:
            result[key] = {"status": "not configured", "url": "", "id": ""}
            continue
        try:
            r = httpx.get(f"https://huggingface.co/api/spaces/{sid}", headers=headers, timeout=5.0)
            if r.status_code == 200:
                stage = r.json().get("runtime", {}).get("stage", "UNKNOWN")
                result[key] = {"status": stage, "url": _hf_space_url(sid), "id": sid}
            else:
                result[key] = {"status": f"HTTP {r.status_code}", "url": "", "id": sid}
        except Exception as exc:
            result[key] = {"status": "error", "url": "", "id": sid, "err": str(exc)}
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
    ax.set_title("Curriculum Heatmap — success rate per domain x difficulty tier",
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    if _reward_hist:
        axes[0].hist(_reward_hist, bins=min(20, max(5, len(_reward_hist))),
                     color="#4a90e2", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Reward")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Reward Distribution", fontweight="bold")
    if _reward_hist:
        axes[1].plot(range(len(_reward_hist)), _reward_hist,
                     color="#4a90e2", linewidth=1.3, alpha=0.9)
        axes[1].fill_between(range(len(_reward_hist)), _reward_hist,
                              alpha=0.15, color="#4a90e2")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Reward Over Time", fontweight="bold")
    plt.tight_layout()
    return fig


def refresh_dashboard():
    # Read health metrics directly — same process, no HTTP round-trip needed
    uptime   = time.time() - _start_time
    err_rate = _error_count / _request_count if _request_count > 0 else 0.0
    lat      = float(np.percentile(list(_latency_window), 95)) if _latency_window else 0.0
    mem      = psutil.Process().memory_info().rss / (1024 * 1024)
    ep_count = len(_episode_history)
    s        = dict(_current_state)   # shallow copy; GIL makes this safe for a read-only dashboard

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

    alerts = []
    if err_rate > 0.1:
        alerts.append(f"HIGH ERROR RATE: {err_rate:.1%}")
    if lat > 1.0:
        alerts.append(f"HIGH LATENCY P95: {lat * 1000:.0f}ms")
    if disc < 0.3 and ep_count > 10:
        alerts.append(f"LOW DISCRIMINATION COVERAGE: {disc:.1%}")
    alert_str = "\n".join(alerts) if alerts else "All systems nominal"

    health_md = (
        f"| Metric | Value |\n|---|---|\n"
        f"| Uptime | {uptime:.0f}s |\n"
        f"| Error Rate | {err_rate:.1%} |\n"
        f"| Latency P95 | {lat * 1000:.1f}ms |\n"
        f"| RAM | {mem:.0f} MB |"
    )
    episode_md = (
        f"**Episode:** `{ep_id or 'none'}` | **Turn:** `{turn}` | "
        f"**Agent:** `{agent_ver}` | **Total:** `{ep_count}`\n\n"
        f"**Task:** {task_text[:300]}"
    )
    curriculum_md = (
        f"**Difficulty Tier:** `{diff_tier}` | "
        f"**In Zone:** `{in_zone}` | **Too Easy:** `{too_easy}` | **Too Hard:** `{too_hard}` | "
        f"**Coverage:** `{disc:.1%}`"
    )

    return (
        _heatmap_fig(), _disc_fig(), _reward_fig(),
        ep_count, disc, diff_tier,
        ep_table,
        health_md, episode_md, curriculum_md, alert_str,
    )


def build_gradio():
    port      = int(os.getenv("PORT", "7860"))
    env_url   = f"http://localhost:{port}"
    space_id  = os.getenv("HF_SPACE_ID", "")
    space_url = _hf_space_url(space_id) if space_id else ""
    train_sid = os.getenv("ALICE_HF_REPO_ID", "")
    train_url = _hf_space_url(train_sid) if train_sid else ""

    with gr.Blocks(title="ALICE RL Environment", theme=gr.themes.Soft(), css=_CSS) as demo:
        gr.Markdown(
            "# ALICE — Adversarial Loop for Inter-model Co-evolutionary Environment\n"
            f"Environment API: [Swagger Docs]({env_url}/docs) | [Health]({env_url}/health) | "
            + (f"[HF Space]({space_url})" if space_url else "_Set `HF_SPACE_ID` for Space link_")
        )

        with gr.Tabs(elem_classes="tab-nav"):

            # ── Tab 1: Overview ──────────────────────────────────────────
            with gr.TabItem("Overview"):
                with gr.Row():
                    ep_count_box  = gr.Number(label="Episodes Run",            value=0,   precision=0, interactive=False)
                    disc_box      = gr.Number(label="Discrimination Coverage", value=0.0, precision=3, interactive=False)
                    diff_tier_box = gr.Number(label="Difficulty Tier",         value=1,   precision=0, interactive=False)
                with gr.Row():
                    with gr.Column(scale=1):
                        health_md  = gr.Markdown("_Loading health..._")
                    with gr.Column(scale=2):
                        episode_md = gr.Markdown("_No active episode yet._")
                alert_box = gr.Textbox(label="System Alerts", interactive=False, lines=3, value="Loading...")

            # ── Tab 2: Curriculum ─────────────────────────────────────────
            with gr.TabItem("Curriculum"):
                curriculum_info = gr.Markdown("_Loading..._")
                heatmap_plot    = gr.Plot(label="Domain x Tier Success Rates")
                disc_plot       = gr.Plot(label="Discrimination Zone Coverage Over Time")

            # ── Tab 3: Training Metrics ───────────────────────────────────
            with gr.TabItem("Training Metrics"):
                reward_plot = gr.Plot(label="Reward Distribution & Time Series")
                gr.Markdown("### Recent Episodes")
                ep_table = gr.Dataframe(
                    headers=["episode_id", "task", "difficulty", "reward", "timestamp"],
                    interactive=False,
                    wrap=True,
                )

            # ── Tab 4: HF Space & Jobs ────────────────────────────────────
            with gr.TabItem("HF Space & Jobs"):
                gr.Markdown(
                    "Live status of Hugging Face Spaces. "
                    "Set `HF_SPACE_ID` (env space) and `ALICE_HF_REPO_ID` (training space) "
                    "in your Space secrets. `HF_TOKEN` required for private spaces."
                )
                hf_status_md   = gr.Markdown("_Click Refresh to load Space status._")
                refresh_hf_btn = gr.Button("Refresh HF Status", variant="secondary")
                gr.Markdown(
                    "**Quick Links:**\n"
                    + (f"- [Environment Space]({space_url})\n" if space_url else "- Environment Space: _not configured_\n")
                    + (f"- [Training Space]({train_url})\n" if train_url else "- Training Space: _not configured_\n")
                    + f"- [API Swagger Docs]({env_url}/docs)\n"
                    + f"- [Health Endpoint]({env_url}/health)\n"
                    + "- [HF Space Jobs Dashboard](https://huggingface.co/spaces)"
                )

                def _refresh_hf_status():
                    _STAGE_LABEL = {"RUNNING": "running", "STOPPED": "stopped",
                                    "PAUSED": "stopped", "error": "stopped"}
                    info  = _hf_space_info()
                    lines = ["### Hugging Face Space Status\n"]
                    for key, label in [("env_space", "Environment Space"),
                                       ("training_space", "Training Space")]:
                        v      = info.get(key, {})
                        sid    = v.get("id") or os.getenv(
                            "HF_SPACE_ID" if key == "env_space" else "ALICE_HF_REPO_ID", "_not set_")
                        status = v.get("status", "n/a")
                        url    = v.get("url", "")
                        stage  = _STAGE_LABEL.get(status, "building")
                        lines.append(f"**{label}** `{sid}` — status: **{status}** ({stage})")
                        if url:
                            lines.append(f"  URL: [{url}]({url})")
                        lines.append("")
                    return "\n".join(lines)

                refresh_hf_btn.click(_refresh_hf_status, outputs=[hf_status_md])

            # ── Tab 5: Failure Bank ───────────────────────────────────────
            with gr.TabItem("Failure Bank"):
                with gr.Row():
                    filter_error = gr.Textbox(label="Filter by error_type",    placeholder="e.g. verification_failure")
                    filter_agent = gr.Textbox(label="Filter by agent_version", placeholder="e.g. 0.0.0")
                    apply_filter = gr.Button("Apply Filter", variant="secondary")
                failure_table = gr.Dataframe(
                    headers=["failure_id", "error_type", "agent_version", "novelty_score", "timestamp"],
                    interactive=False,
                )

                def _apply_failure_filter(error_type, agent_version):
                    try:
                        *_, fb = _get_components()
                        entries = fb.query_failures(
                            error_type=error_type.strip() or None,
                            agent_version=agent_version.strip() or None,
                        )
                        return [
                            [e.failure_id[:8], e.error_type, e.agent_version,
                             round(e.novelty_score, 3), e.timestamp[:19]]
                            for e in entries[:50]
                        ]
                    except Exception:
                        return []

                apply_filter.click(
                    _apply_failure_filter,
                    inputs=[filter_error, filter_agent],
                    outputs=[failure_table],
                )

            # ── Tab 6: Leaderboard ────────────────────────────────────────
            with gr.TabItem("Leaderboard"):
                gr.Markdown(
                    "## ALICE RL Leaderboard\n"
                    "Benchmark models ranked by composite RL score "
                    "_(0.5 × avg_reward + 0.3 × success_rate + 0.2 × disc_coverage)_. "
                    "Select models to compare or submit your own."
                )
                with gr.Row():
                    lb_model_filter = gr.CheckboxGroup(
                        label="Show models",
                        choices=[
                            "Qwen/Qwen2.5-0.5B-Instruct",
                            "Qwen/Qwen2.5-1.5B-Instruct",
                            "Qwen/Qwen2.5-3B-Instruct",
                            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                            "google/gemma-3-1b-it",
                        ],
                        value=[
                            "Qwen/Qwen2.5-0.5B-Instruct",
                            "Qwen/Qwen2.5-1.5B-Instruct",
                            "Qwen/Qwen2.5-3B-Instruct",
                            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                            "google/gemma-3-1b-it",
                        ],
                        interactive=True,
                    )
                lb_refresh_btn = gr.Button("Refresh Leaderboard", variant="primary")
                leaderboard_table = gr.Dataframe(
                    headers=["rank", "model", "params_B", "rl_score",
                              "avg_reward", "success_rate", "disc_coverage", "episodes", "source"],
                    interactive=False,
                    wrap=True,
                )
                gr.Markdown("### Submit a Model for Comparison")
                with gr.Row():
                    submit_model_id   = gr.Textbox(label="HF model ID",     placeholder="username/my-model")
                    submit_name       = gr.Textbox(label="Display name",     placeholder="My Model")
                    submit_params     = gr.Number(label="Params (B)",        value=0.0, precision=1)
                    submit_btn        = gr.Button("Submit", variant="secondary")
                submit_status = gr.Textbox(label="Status", interactive=False, lines=1)

                def _load_leaderboard(selected_ids):
                    lb      = _get_leaderboard()
                    ids     = selected_ids if selected_ids else None
                    entries = lb.get_leaderboard(model_ids=ids)
                    return [
                        [e["rank"], e["display_name"], e["params_b"],
                         e["rl_score"], e["avg_reward"], e["success_rate"],
                         e["discrimination_coverage"], e["episodes_run"], e["source"]]
                        for e in entries
                    ]

                def _submit_model(model_id, display_name, params_b):
                    if not model_id.strip():
                        return "Error: model_id is required"
                    lb  = _get_leaderboard()
                    res = lb.submit_model(model_id.strip(),
                                          display_name.strip() or None,
                                          float(params_b or 0))
                    return f"Status: {res['status']} — {res['model_id']}"

                lb_refresh_btn.click(
                    _load_leaderboard,
                    inputs=[lb_model_filter],
                    outputs=[leaderboard_table],
                )
                submit_btn.click(
                    _submit_model,
                    inputs=[submit_model_id, submit_name, submit_params],
                    outputs=[submit_status],
                )

        gr.Markdown(
            f"---\n*Auto-refreshes every 3 s. "
            f"Environment API at `{env_url}` | [Swagger]({env_url}/docs)*"
        )

        timer   = gr.Timer(value=3)
        outputs = [
            heatmap_plot, disc_plot, reward_plot,
            ep_count_box, disc_box, diff_tier_box,
            ep_table,
            health_md, episode_md, curriculum_info, alert_box,
        ]
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
