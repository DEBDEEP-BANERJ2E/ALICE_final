"""
ALICE Analytics Dashboard — standalone Gradio app.

Connects to the ALICE environment server (default http://localhost:7860) via HTTP.
Shows advanced analytics across five tabs:
  1. Overview       — health, current episode, alerts
  2. Curriculum     — heatmap + discrimination zone time series
  3. Training       — reward distribution / time series
  4. HF Space & Jobs — live Space status from HF API
  5. Failure Bank   — filterable failure table

Run:
    python dashboard/gradio_app.py

Or set ALICE_ENV_URL to point at a remote Space, e.g.:
    ALICE_ENV_URL=https://user-space.hf.space python dashboard/gradio_app.py
"""

from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Tuple

import httpx
import numpy as np

ENV_URL      = os.getenv("ALICE_ENV_URL",          "http://localhost:7860")
POLL_INTERVAL = int(os.getenv("DASHBOARD_POLL_INTERVAL", "3"))

# ---------------------------------------------------------------------------
# In-memory ring buffers
# ---------------------------------------------------------------------------
_disc_history: Deque[float]   = deque(maxlen=200)
_reward_history: Deque[float] = deque(maxlen=200)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(path: str, timeout: float = 3.0) -> Dict[str, Any]:
    try:
        return httpx.get(f"{ENV_URL}{path}", timeout=timeout).json()
    except Exception:
        return {}


def _hf_space_url(space_id: str) -> str:
    """Convert 'user/space-name' to its HF Space URL."""
    parts = space_id.split("/")
    return f"https://{parts[0]}-{parts[1]}.hf.space" if len(parts) == 2 else ""


def _hf_space_info() -> Dict[str, Any]:
    env_sid   = os.getenv("HF_SPACE_ID", "")
    train_sid = os.getenv("ALICE_HF_REPO_ID", "")
    hf_token  = os.getenv("HF_TOKEN", "")
    headers   = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    result: Dict[str, Any] = {}
    for key, sid in [("env_space", env_sid), ("training_space", train_sid)]:
        if not sid:
            result[key] = {"status": "not configured", "url": "", "id": ""}
            continue
        try:
            r = httpx.get(f"https://huggingface.co/api/spaces/{sid}",
                          headers=headers, timeout=5.0)
            if r.status_code == 200:
                stage = r.json().get("runtime", {}).get("stage", "UNKNOWN")
                result[key] = {"status": stage, "url": _hf_space_url(sid), "id": sid}
            else:
                result[key] = {"status": f"HTTP {r.status_code}", "url": "", "id": sid}
        except Exception as exc:
            result[key] = {"status": "error", "url": "", "id": sid, "err": str(exc)}
    return result

# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def _heatmap_fig(heatmap_data: Any = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.5))
        if heatmap_data is None:
            heatmap_data = np.zeros((5, 10), dtype=np.float32)
        domains = ["arithmetic", "logic", "factual", "symbolic", "code"]
        im = ax.imshow(heatmap_data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"T{i+1}" for i in range(10)], fontsize=8)
        ax.set_yticks(range(5))
        ax.set_yticklabels(domains, fontsize=9)
        ax.set_title("Curriculum Heatmap — success rate per domain x difficulty tier",
                     fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Success Rate")
        plt.tight_layout()
        return fig
    except Exception:
        return None


def _disc_fig():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3))
        if _disc_history:
            xs = range(len(_disc_history))
            ax.fill_between(xs, list(_disc_history), alpha=0.18, color="#4a90e2")
            ax.plot(xs, list(_disc_history), color="#4a90e2", linewidth=1.8)
        ax.axhline(0.3, color="red",   linestyle="--", alpha=0.65, label="Min 30%")
        ax.axhline(0.7, color="green", linestyle="--", alpha=0.65, label="Target 70%")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Poll")
        ax.set_ylabel("Coverage")
        ax.set_title("Discrimination Zone Coverage Over Time", fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        return fig
    except Exception:
        return None


def _reward_fig():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rh = list(_reward_history)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3))
        if rh:
            axes[0].hist(rh, bins=min(20, max(5, len(rh))),
                         color="#4a90e2", edgecolor="white", alpha=0.85)
        axes[0].set_xlabel("Reward")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Reward Distribution", fontweight="bold")
        if rh:
            axes[1].plot(range(len(rh)), rh, color="#4a90e2", linewidth=1.3, alpha=0.9)
            axes[1].fill_between(range(len(rh)), rh, alpha=0.15, color="#4a90e2")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Reward")
        axes[1].set_title("Reward Over Time", fontweight="bold")
        plt.tight_layout()
        return fig
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Main refresh callback
# ---------------------------------------------------------------------------

def refresh():
    health  = _get("/health")
    state   = _get("/state")

    uptime   = health.get("uptime", 0.0)
    err_rate = health.get("error_rate", 0.0)
    lat      = health.get("latency_p95", 0.0)
    mem      = health.get("memory_usage", 0.0)

    task_text = state.get("task") or "No active episode"
    turn      = state.get("turn_number", 0)
    ep_id     = (state.get("episode_id") or "")[:8]
    agent_ver = state.get("agent_version", "n/a")

    # Discrimination coverage from environment (approximated per poll)
    disc_coverage = 0.0
    heatmap_data  = None

    # Alerts
    alerts = []
    if err_rate > 0.1:
        alerts.append(f"HIGH ERROR RATE: {err_rate:.1%}")
    if lat > 1.0:
        alerts.append(f"HIGH LATENCY P95: {lat * 1000:.0f}ms")
    alert_str = "\n".join(alerts) if alerts else "All systems nominal"

    _disc_history.append(disc_coverage)

    health_md = (
        f"| Metric | Value |\n|---|---|\n"
        f"| Server | `{ENV_URL}` |\n"
        f"| Uptime | {uptime:.0f}s |\n"
        f"| Error Rate | {err_rate:.1%} |\n"
        f"| Latency P95 | {lat * 1000:.1f}ms |\n"
        f"| RAM | {mem:.0f} MB |"
    )
    episode_md = (
        f"**Episode:** `{ep_id or 'none'}` | **Turn:** `{turn}` | **Agent:** `{agent_ver}`\n\n"
        f"**Task:** {task_text[:400]}"
    )

    return (
        _heatmap_fig(heatmap_data),
        _disc_fig(),
        _reward_fig(),
        health_md,
        episode_md,
        alert_str,
    )

# ---------------------------------------------------------------------------
# Dashboard builder
# ---------------------------------------------------------------------------

_CSS = """
.tab-nav button { font-size: 14px; font-weight: 600; }
"""


def build_dashboard():
    import gradio as gr

    space_id  = os.getenv("HF_SPACE_ID", "")
    space_url = _hf_space_url(space_id) if space_id else ""
    train_sid = os.getenv("ALICE_HF_REPO_ID", "")
    train_url = _hf_space_url(train_sid) if train_sid else ""

    with gr.Blocks(title="ALICE Analytics Dashboard", theme=gr.themes.Soft(), css=_CSS) as demo:
        gr.Markdown(
            "# ALICE Analytics Dashboard\n"
            f"Connected to: `{ENV_URL}` | [Swagger Docs]({ENV_URL}/docs) | "
            + (f"[HF Space]({space_url})" if space_url else "_Set `HF_SPACE_ID` env var_")
        )

        with gr.Tabs(elem_classes="tab-nav"):

            # ── Tab 1: Overview ──────────────────────────────────────────
            with gr.TabItem("Overview"):
                with gr.Row():
                    with gr.Column(scale=1):
                        health_md  = gr.Markdown("_Loading health..._")
                    with gr.Column(scale=2):
                        episode_md = gr.Markdown("_No active episode yet._")
                alert_box  = gr.Textbox(label="System Alerts", interactive=False, lines=3)
                export_btn = gr.Button("Export State JSON", variant="secondary")
                export_out = gr.Textbox(label="Exported JSON", lines=6, interactive=False)

                def _export():
                    data = {
                        "state":     _get("/state"),
                        "health":    _get("/health"),
                        "timestamp": datetime.now().isoformat(),
                    }
                    return json.dumps(data, indent=2)

                export_btn.click(_export, outputs=[export_out])

            # ── Tab 2: Curriculum ─────────────────────────────────────────
            with gr.TabItem("Curriculum"):
                heatmap_plot = gr.Plot(label="Domain x Tier Success Rates")
                disc_plot    = gr.Plot(label="Discrimination Zone Coverage Over Time")

            # ── Tab 3: Training Metrics ───────────────────────────────────
            with gr.TabItem("Training Metrics"):
                reward_plot = gr.Plot(label="Reward Distribution & Time Series")
                with gr.Row():
                    pause_btn   = gr.Button("Pause Updates", variant="secondary")
                    status_text = gr.Textbox(label="Status", value="Running", interactive=False, scale=1)
                _paused_state = gr.State(value=False)

                def _toggle_pause(paused):
                    new_paused = not paused
                    label  = "Resume Updates" if new_paused else "Pause Updates"
                    status = "Paused" if new_paused else "Running"
                    return new_paused, gr.update(value=label), status

                pause_btn.click(_toggle_pause,
                                inputs=[_paused_state],
                                outputs=[_paused_state, pause_btn, status_text])

            # ── Tab 4: HF Space & Jobs ────────────────────────────────────
            with gr.TabItem("HF Space & Jobs"):
                gr.Markdown(
                    "Live status of Hugging Face Spaces. "
                    "Set `HF_SPACE_ID` and `ALICE_HF_REPO_ID` env vars, plus `HF_TOKEN` "
                    "for private spaces."
                )
                hf_status_md   = gr.Markdown("_Click Refresh to load Space status._")
                refresh_hf_btn = gr.Button("Refresh HF Status", variant="secondary")
                gr.Markdown(
                    "**Quick Links:**\n"
                    + (f"- [Environment Space]({space_url})\n" if space_url else "- Environment Space: _not configured_\n")
                    + (f"- [Training Space]({train_url})\n" if train_url else "- Training Space: _not configured_\n")
                    + f"- [API Swagger Docs]({ENV_URL}/docs)\n"
                    + f"- [Health Endpoint]({ENV_URL}/health)\n"
                    + "- [HF Space Jobs Dashboard](https://huggingface.co/spaces)"
                )

                def _refresh_hf():
                    info  = _hf_space_info()
                    lines = ["### Hugging Face Space Status\n"]
                    for key, label in [("env_space", "Environment Space"),
                                       ("training_space", "Training Space")]:
                        v      = info.get(key, {})
                        sid    = v.get("id") or os.getenv(
                            "HF_SPACE_ID" if key == "env_space" else "ALICE_HF_REPO_ID", "_not set_")
                        status = v.get("status", "n/a")
                        url    = v.get("url", "")
                        lines.append(f"**{label}** `{sid}` — **{status}**")
                        if url:
                            lines.append(f"  URL: [{url}]({url})")
                        lines.append("")
                    return "\n".join(lines)

                refresh_hf_btn.click(_refresh_hf, outputs=[hf_status_md])

            # ── Tab 5: Failure Bank ───────────────────────────────────────
            with gr.TabItem("Failure Bank"):
                gr.Markdown("Query the environment server's failure bank.")
                with gr.Row():
                    filter_error = gr.Textbox(label="error_type filter",    placeholder="verification_failure")
                    filter_agent = gr.Textbox(label="agent_version filter", placeholder="0.0.0")
                    apply_filter = gr.Button("Query", variant="secondary")
                failure_table = gr.Dataframe(
                    headers=["failure_id", "error_type", "agent_version", "novelty_score", "timestamp"],
                    interactive=False,
                )

                def _query_failures(error_type, agent_version):
                    try:
                        params: Dict[str, Any] = {}
                        if error_type.strip():
                            params["error_type"] = error_type.strip()
                        if agent_version.strip():
                            params["agent_version"] = agent_version.strip()
                        resp = httpx.get(f"{ENV_URL}/failures", params=params, timeout=5.0)
                        if resp.status_code == 200:
                            rows = resp.json()
                            return [[r.get("failure_id", "")[:8], r.get("error_type", ""),
                                     r.get("agent_version", ""), round(r.get("novelty_score", 0), 3),
                                     str(r.get("timestamp", ""))[:19]]
                                    for r in rows[:50]]
                    except Exception:
                        pass
                    return []

                apply_filter.click(_query_failures,
                                   inputs=[filter_error, filter_agent],
                                   outputs=[failure_table])

        gr.Markdown(
            f"---\n*Auto-refreshes every {POLL_INTERVAL}s. "
            f"Connected to `{ENV_URL}`*"
        )

        # Auto-refresh
        timer   = gr.Timer(value=POLL_INTERVAL)
        outputs = [heatmap_plot, disc_plot, reward_plot, health_md, episode_md, alert_box]
        timer.tick(refresh, outputs=outputs)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_dashboard()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7861")),
        share=False,
    )