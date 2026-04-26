# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Alice Env Environment.

This module creates an HTTP server that exposes the AliceEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import AliceAction, AliceObservation
    from .alice_env_environment import AliceEnvironment
except (ModuleNotFoundError, ImportError):
    from models import AliceAction, AliceObservation
    from server.alice_env_environment import AliceEnvironment


# Create the app with web interface and README integration
app = create_app(
    AliceEnvironment,
    AliceAction,
    AliceObservation,
    env_name="alice_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


# Add custom health endpoint with detailed metrics
# Note: The OpenEnv framework provides a basic /health endpoint.
# This custom endpoint provides detailed metrics as required by task 2.1.
@app.get("/health/detailed")
async def health_detailed():
    """
    Get detailed health metrics for the ALICE environment.
    
    Returns:
        Dictionary containing:
        - uptime: Seconds since environment initialization
        - error_rate: Ratio of errors to total requests
        - latency_p95: 95th percentile latency in seconds
        - memory_usage: Memory usage (placeholder)
    """
    # Create a temporary environment instance to get health metrics
    # In production, this would query a shared metrics store
    env = AliceEnvironment()
    return env.health()


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m alice_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn alice_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
