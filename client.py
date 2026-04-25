"""LifeOS Agent – OpenEnv WebSocket client."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import LifeOSAction, LifeOSObservation, LifeOSState


class LifeOSEnvClient(EnvClient[LifeOSAction, LifeOSObservation, LifeOSState]):
    """Async WebSocket client for the LifeOS Agent environment.

    Example (async)::

        async with LifeOSEnvClient(base_url="http://localhost:8000") as env:
            result = await env.reset(seed=42)
            while not result.done:
                action = agent.decide(result.observation)
                result = await env.step(action)

    Example (sync wrapper)::

        env = LifeOSEnvClient(base_url="http://localhost:8000").sync()
        with env:
            result = env.reset()
            result = env.step(action)
    """

    # ── payload / parse overrides ───────────────────────────────────────

    def _step_payload(self, action: LifeOSAction) -> Dict[str, Any]:
        """Serialise a :class:`LifeOSAction` to the JSON dict sent to the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[LifeOSObservation]:
        """Parse the server response into a :class:`StepResult`.

        Expected *payload* keys:
            - ``observation`` – dict matching :class:`LifeOSObservation`
            - ``reward``      – float (step reward)
            - ``done``        – bool  (episode termination flag)
        """
        obs_data = payload.get("observation", {})
        observation = LifeOSObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> LifeOSState:
        """Parse the server's state response into a :class:`LifeOSState`."""
        return LifeOSState(**payload.get("state", payload))


# ── convenience factory ─────────────────────────────────────────────────────


def create_env(base_url: str = "http://localhost:8001") -> LifeOSEnvClient:
    """Create a :class:`LifeOSEnvClient` pointed at *base_url*.

    The caller is responsible for connecting (either ``await client.connect()``
    or using the async context manager).
    """
    return LifeOSEnvClient(base_url=base_url)
