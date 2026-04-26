# Copyright (c) LifeOS Team 2026. All rights reserved.
# BSD-3-Clause License

"""
Pydantic data models for the LifeOS Agent environment.

Three core schemas define the contract between agent and environment:
  - LifeOSAction: what the agent decides to do
  - LifeOSObservation: what the agent sees after acting
  - LifeOSState: internal environment bookkeeping
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core import Observation as _OpenEnvObservation
except ImportError:
    # Fallback for standalone usage (e.g. Gradio UI without openenv)
    _OpenEnvObservation = BaseModel


class LifeOSAction(BaseModel):
    """A structured action the agent takes to resolve a crisis.

    The agent must choose an action type, target a specific person,
    craft actual message content, explain its reasoning, and declare
    urgency. This forces specificity — no vague hand-waving allowed.
    """

    action_type: str = Field(
        ...,
        description=(
            "Type of action to take. One of: send_message, reschedule, "
            "book_alternative, delegate, decline, escalate, negotiate"
        ),
    )
    target_person: str = Field(
        ...,
        description="Name of the person this action targets (e.g. 'Partner_Jamie', 'Boss_Karen')",
    )
    content: str = Field(
        ...,
        description="The actual message or action content. Must be specific and actionable.",
    )
    reasoning: str = Field(
        ...,
        description="Why this is the right action right now. Must be substantive (>40 chars).",
    )
    urgency: str = Field(
        ...,
        description="How urgent this action is. One of: immediate, within_hour, today, tomorrow",
    )


class LifeOSObservation(_OpenEnvObservation):
    """What the agent observes after taking an action.

    Contains the crisis scenario, active conflicts still unresolved,
    how personas responded, time pressure level, and a full breakdown
    of the reward across all 5 independent components.
    """

    scenario_description: str = Field(
        ...,
        description="Full text description of the current crisis scenario.",
    )
    active_conflicts: list[str] = Field(
        ...,
        description="List of conflict identifiers still active in this episode.",
    )
    persona_responses: dict[str, str] = Field(
        ...,
        description="Mapping of persona name to their response after the agent's action.",
    )
    time_pressure: str = Field(
        ...,
        description="Current time pressure level: low, medium, high, critical.",
    )
    step_number: int = Field(
        ...,
        description="Current step number within the episode (0-indexed).",
    )
    reward_breakdown: dict[str, float] = Field(
        ...,
        description=(
            "Decomposed reward across 5 components: conflict_addressed, "
            "stakeholder_reached, action_specificity, format_compliance, "
            "no_escalation."
        ),
    )


class LifeOSState(BaseModel):
    """Internal environment state for episode tracking.

    Tracks scenario identity, difficulty tier, step progress,
    conflict resolution status, and cumulative reward history.
    """

    scenario_id: str = Field(
        ...,
        description="Unique identifier for the current scenario (e.g. 'easy_01').",
    )
    difficulty: str = Field(
        ...,
        description="Difficulty tier: easy, medium, or hard.",
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken so far in this episode.",
    )
    max_steps: int = Field(
        default=10,
        description="Maximum steps allowed per episode.",
    )
    conflicts_total: int = Field(
        default=3,
        description="Total number of conflicts in the current scenario.",
    )
    conflicts_resolved: int = Field(
        default=0,
        description="Number of conflicts the agent has addressed.",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended.",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Sum of all rewards received this episode.",
    )
    reward_history: list[float] = Field(
        default_factory=list,
        description="List of rewards received at each step.",
    )
