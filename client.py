# Copyright (c) LifeOS Team 2026. All rights reserved.
# BSD-3-Clause License

"""
WebSocket client for the LifeOS Agent environment.

Subclasses openenv.core.EnvClient to provide typed interaction with
the LifeOS environment server over WebSocket.

Usage:
    from client import create_env

    env = create_env("http://localhost:8001").sync()
    with env:
        result = env.reset(seed=42)
        print(result.observation.scenario_description)
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import LifeOSAction, LifeOSObservation, LifeOSState


class LifeOSEnvClient(EnvClient[LifeOSAction, LifeOSObservation, LifeOSState]):
    """Typed WebSocket client for the LifeOS Agent environment.

    Converts between Python model objects and the JSON payloads expected
    by the OpenEnv server protocol.
    """

    def _step_payload(self, action: LifeOSAction) -> Dict[str, Any]:
        """Convert a LifeOSAction to JSON for the server.

        Args:
            action: The structured action to send.

        Returns:
            Dictionary payload matching the server's expected schema.
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[LifeOSObservation]:
        """Convert server JSON response to a typed StepResult.

        Args:
            payload: Raw JSON dictionary from the server.

        Returns:
            StepResult containing LifeOSObservation, reward, and done flag.
        """
        observation = LifeOSObservation(**payload.get("observation", payload))
        reward = payload.get("reward", 0.0)
        done = payload.get("done", False)
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> LifeOSState:
        """Convert server JSON state response to a typed LifeOSState.

        Args:
            payload: Raw JSON dictionary from the server.

        Returns:
            LifeOSState with current episode tracking information.
        """
        return LifeOSState(**payload)


def create_env(base_url: str = "http://localhost:8001") -> LifeOSEnvClient:
    """Factory function to create a LifeOS environment client.

    Args:
        base_url: URL of the running LifeOS environment server.

    Returns:
        LifeOSEnvClient instance. Call .sync() for synchronous usage.

    Example:
        env = create_env("http://localhost:8001").sync()
        with env:
            result = env.reset(seed=42)
            result = env.step(LifeOSAction(...))
    """
    return LifeOSEnvClient(base_url=base_url)
