# Copyright (c) LifeOS Team 2026. All rights reserved.
# BSD-3-Clause License

"""
FastAPI application for the LifeOS Agent Environment.

Creates an HTTP + WebSocket server using OpenEnv's create_app, exposing
the LifeOSEnvironment for both programmatic RL training and interactive
demos.

Endpoints:
    - POST /reset: Reset the environment with a new crisis scenario
    - POST /step: Execute an action and receive reward
    - GET /state: Get current environment state
    - GET /schema: Get action/observation JSON schemas
    - GET /health: Health check
    - GET /metadata: Environment metadata
    - POST /mcp: MCP JSON-RPC endpoint
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Run directly:
    python -m server.app
"""

from __future__ import annotations

try:
    from openenv.core import create_app
except ImportError as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import LifeOSAction, LifeOSObservation
    from .lifeos_environment import LifeOSEnvironment
except (ImportError, ModuleNotFoundError):
    from models import LifeOSAction, LifeOSObservation
    from server.lifeos_environment import LifeOSEnvironment


# ──────────────────────────────────────────────────────────────────────
# Create the OpenEnv FastAPI application
# ──────────────────────────────────────────────────────────────────────

app = create_app(
    LifeOSEnvironment,
    LifeOSAction,
    LifeOSObservation,
    env_name="lifeos_agent",
    max_concurrent_envs=1,
)


# ──────────────────────────────────────────────────────────────────────
# Entry point for direct execution
# ──────────────────────────────────────────────────────────────────────


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the server directly without Docker.

    Args:
        host: Host address to bind to (default: "0.0.0.0").
        port: Port number to listen on (default: 8000).
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LifeOS Agent Server")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()
    main(host=args.host, port=args.port)