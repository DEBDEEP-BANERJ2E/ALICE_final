"""
ALICE Gradio Dashboard — live monitoring and observability.

Displays:
- Curriculum heatmap
- Discrimination score time series
- Failure bank browser
- Episode trajectory viewer
- Reward decomposition histogram
- Entropy and escalation alerts
- Training metrics panel
"""

from __future__ import annotations

import os

import httpx

ENV_URL = os.getenv("ALICE_ENV_URL", "http://localhost:8000")
POLL_INTERVAL = 1  # seconds


def fetch_state() -> dict:
    """Fetch current environment state from the OpenEnv server."""
    try:
        resp = httpx.get(f"{ENV_URL}/state", timeout=2.0)
        return resp.json()
    except Exception:
        return {}


def fetch_health() -> dict:
    """Fetch health metrics from the OpenEnv server."""
    try:
        resp = httpx.get(f"{ENV_URL}/health", timeout=2.0)
        return resp.json()
    except Exception:
        return {}


def build_dashboard():
    """Build and return the Gradio Blocks dashboard."""
    import gradio as gr  # type: ignore

    with gr.Blocks(title="ALICE RL Environment Dashboard") as demo:
        gr.Markdown("# 🔬 ALICE RL Environment — Live Dashboard")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Curriculum Heatmap")
                curriculum_plot = gr.Plot(label="Task Difficulty Distribution")

            with gr.Column():
                gr.Markdown("### Discrimination Score")
                discrimination_plot = gr.Plot(label="Discrimination Zone Coverage (%)")

        with gr.Row():
            gr.Markdown("### Training Metrics")
            episode_count = gr.Number(label="Episode Count", value=0)
            cumulative_reward = gr.Number(label="Cumulative Reward", value=0.0)
            policy_loss = gr.Number(label="Policy Loss", value=0.0)

        with gr.Row():
            gr.Markdown("### Failure Bank Browser")
            failure_table = gr.Dataframe(
                headers=["failure_id", "error_type", "agent_version", "novelty_score", "timestamp"],
                label="Failures",
            )
            with gr.Column():
                filter_error_type = gr.Textbox(label="Filter by error_type")
                filter_agent_version = gr.Textbox(label="Filter by agent_version")

        with gr.Row():
            gr.Markdown("### Episode Trajectory Viewer")
            trajectory_json = gr.JSON(label="Selected Trajectory")

        with gr.Row():
            gr.Markdown("### Reward Decomposition")
            reward_plot = gr.Plot(label="Reward Components")

        with gr.Row():
            gr.Markdown("### Entropy & Escalation Alerts")
            alerts_text = gr.Textbox(label="Active Alerts", lines=4)

        with gr.Row():
            export_csv_btn = gr.Button("Export CSV")
            export_json_btn = gr.Button("Export JSON")
            pause_btn = gr.Button("Pause Updates")

        # Placeholder callbacks — wired to real data in Task 14
        def refresh():
            state = fetch_state()
            return (
                None,  # curriculum_plot
                None,  # discrimination_plot
                state.get("episode_id", 0),
                0.0,
                0.0,
                [],   # failure_table
                {},   # trajectory_json
                None,  # reward_plot
                "No active alerts.",
            )

        demo.load(
            refresh,
            outputs=[
                curriculum_plot,
                discrimination_plot,
                episode_count,
                cumulative_reward,
                policy_loss,
                failure_table,
                trajectory_json,
                reward_plot,
                alerts_text,
            ],
            every=POLL_INTERVAL,
        )

    return demo


if __name__ == "__main__":
    app = build_dashboard()
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
