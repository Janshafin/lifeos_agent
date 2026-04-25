"""LifeOS Agent – Pydantic models for the OpenEnv RL environment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Action space ────────────────────────────────────────────────────────────

class LifeOSAction(BaseModel):
    """A single action the agent can take within a LifeOS scenario."""

    action_type: Literal[
        "send_message",
        "reschedule",
        "book_alternative",
        "delegate",
        "decline",
        "escalate",
        "negotiate",
    ]
    target_person: str
    content: str
    reasoning: str
    urgency: Literal[
        "immediate",
        "within_hour",
        "today",
        "tomorrow",
    ]


# ── Observation space ───────────────────────────────────────────────────────

class LifeOSObservation(BaseModel):
    """What the agent perceives after each step."""

    scenario_description: str
    active_conflicts: list[str]
    persona_responses: dict[str, str]
    time_pressure: str
    step_number: int
    reward_breakdown: dict[str, float]


# ── Internal environment state ──────────────────────────────────────────────

class LifeOSState(BaseModel):
    """Mutable state tracked by the environment across steps."""

    scenario_id: str
    difficulty: Literal["easy", "medium", "hard"]
    step_count: int = 0
    max_steps: int = 10
    conflicts_total: int = 3
    conflicts_resolved: int = 0
    done: bool = False
    cumulative_reward: float = 0.0
    reward_history: list[float] = Field(default_factory=list)
