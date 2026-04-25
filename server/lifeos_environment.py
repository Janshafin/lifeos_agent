"""
LifeOS Agent Environment – core OpenEnv RL environment.

Presents life-management crisis scenarios at three difficulty tiers and
rewards agents for conflict-resolution quality via five independent,
interpretable reward functions.
"""

from __future__ import annotations

import re
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Observation, State

try:
    from ..models import LifeOSAction, LifeOSObservation, LifeOSState
except ImportError:
    from models import LifeOSAction, LifeOSObservation, LifeOSState


# ── Scenario bank ───────────────────────────────────────────────────────────

SCENARIOS: list[dict[str, Any]] = [
    # ── EASY  (1 conflict each) ─────────────────────────────────────────────
    {
        "id": "easy_01",
        "title": "Meeting Overrun",
        "difficulty": "easy",
        "trigger": "Meeting overrun by 30 minutes, next meeting starts now",
        "conflicts": ["scheduling"],
        "personas": {
            "Alice": "Your next meeting organiser – punctual, expects you on time.",
            "Bob": "Current meeting lead – tends to ramble, unaware of your schedule.",
        },
        "success_criteria": [
            "Inform the next meeting organiser of the delay",
            "Gracefully exit the current meeting",
        ],
    },
    {
        "id": "easy_02",
        "title": "Missed Client Call",
        "difficulty": "easy",
        "trigger": "Client called during lunch, need to call back within the hour",
        "conflicts": ["client_call"],
        "personas": {
            "Client_Sarah": "Senior client – values responsiveness, easily offended.",
            "Manager_Tom": "Your manager – wants to know about all client interactions.",
        },
        "success_criteria": [
            "Return the client call promptly",
            "Inform your manager about the interaction",
        ],
    },
    {
        "id": "easy_03",
        "title": "Team Help Request",
        "difficulty": "easy",
        "trigger": "Team member asks for urgent help, you're in a deadline",
        "conflicts": ["team_request"],
        "personas": {
            "Junior_Dev": "New hire – anxious, blocked on a critical bug.",
            "PM_Rachel": "Project manager – tracking your deliverable closely.",
        },
        "success_criteria": [
            "Acknowledge the team member's request",
            "Protect your own deadline",
        ],
    },
    # ── MEDIUM  (2 conflicts each) ──────────────────────────────────────────
    {
        "id": "medium_01",
        "title": "Flight Delay Cascade",
        "difficulty": "medium",
        "trigger": (
            "Flight delayed 3 hours, partner waiting at airport, "
            "dinner reservation tonight"
        ),
        "conflicts": ["travel_delay", "dinner_reservation"],
        "personas": {
            "Partner_Jamie": "Your partner – worried, already at the airport.",
            "Restaurant_Host": "Upscale restaurant – strict cancellation policy.",
            "Airline_Agent": "Customer service – overworked, limited rebooking options.",
        },
        "success_criteria": [
            "Update partner on new arrival time",
            "Reschedule or cancel the dinner reservation",
            "Explore rebooking options with the airline",
        ],
    },
    {
        "id": "medium_02",
        "title": "Triple Collision",
        "difficulty": "medium",
        "trigger": (
            "Boss demands report now, child school called about emergency, "
            "important call in 1hr"
        ),
        "conflicts": ["boss_report", "child_emergency"],
        "personas": {
            "Boss_Karen": "VP – impatient, expects immediate compliance.",
            "School_Nurse": "School staff – calm but firm, child has minor injury.",
            "Client_VP": "Senior stakeholder – calling in one hour for deal review.",
        },
        "success_criteria": [
            "Address the child emergency immediately",
            "Negotiate a short extension on the report",
            "Prepare for the upcoming client call",
        ],
    },
    {
        "id": "medium_03",
        "title": "Double-Booked Clients",
        "difficulty": "medium",
        "trigger": (
            "Two client meetings double-booked at same time, "
            "both are senior stakeholders"
        ),
        "conflicts": ["client_meeting_A", "client_meeting_B"],
        "personas": {
            "Client_A_VP": "Fortune-500 VP – low tolerance for schedule changes.",
            "Client_B_Director": "Key account director – relationship is fragile.",
            "Your_EA": "Executive assistant – apologetic, made the booking error.",
        },
        "success_criteria": [
            "Reschedule one meeting without damaging the relationship",
            "Attend or delegate the other meeting",
        ],
    },
    # ── HARD  (3 conflicts each) ────────────────────────────────────────────
    {
        "id": "hard_01",
        "title": "Travel Meltdown",
        "difficulty": "hard",
        "trigger": (
            "Flight cancelled, 9am meeting tomorrow in another city, "
            "partner at restaurant, hotel sold out, "
            "boss doesn't know you might miss it"
        ),
        "conflicts": ["cancelled_flight", "partner_dinner", "hotel_booking"],
        "personas": {
            "Partner_Jamie": "At the restaurant alone – upset and worried.",
            "Boss_Mark": "C-suite – expects you in person tomorrow morning.",
            "Airline_Agent": "Only alternative is a red-eye with a layover.",
            "Hotel_Concierge": "All hotels near the venue are fully booked.",
        },
        "success_criteria": [
            "Secure alternative travel to the meeting city",
            "Communicate with partner about the situation",
            "Inform boss proactively with a backup plan",
            "Find overnight accommodation",
        ],
    },
    {
        "id": "hard_02",
        "title": "Team Crisis",
        "difficulty": "hard",
        "trigger": (
            "Team member quit without notice, client deliverable due today, "
            "your own presentation in 2 hours, intern needs guidance"
        ),
        "conflicts": ["team_quit", "client_deliverable", "presentation_prep"],
        "personas": {
            "Client_Director": "Expecting deliverable by EOD – no extensions.",
            "Intern_Alex": "Overwhelmed – first week on the job.",
            "CTO": "Attending your presentation – high visibility.",
            "HR_Lead": "Needs to process the resignation paperwork.",
        },
        "success_criteria": [
            "Redistribute the quitting member's work",
            "Deliver or negotiate the client deliverable",
            "Prepare adequately for the presentation",
            "Provide minimal guidance to the intern",
        ],
    },
    {
        "id": "hard_03",
        "title": "Budget Crisis",
        "difficulty": "hard",
        "trigger": (
            "Budget cut announced mid-project, 3 client contracts at risk, "
            "team morale collapsed, board presentation in 48 hours"
        ),
        "conflicts": ["budget_cut", "client_contracts", "team_morale"],
        "personas": {
            "CFO": "Delivered the budget cut – open to creative proposals.",
            "Client_A": "Largest account – threatening to leave if scope shrinks.",
            "Client_B": "Mid-tier account – flexible but needs reassurance.",
            "Client_C": "New account – considering competitors.",
            "Team_Lead": "Demoralised – key engineers talking about quitting.",
        },
        "success_criteria": [
            "Propose a revised budget allocation",
            "Retain at least two of three client contracts",
            "Stabilise team morale with a concrete action plan",
            "Prepare a credible board presentation narrative",
        ],
    },
]

# Lookup for fast access
_SCENARIOS_BY_ID: dict[str, dict[str, Any]] = {s["id"]: s for s in SCENARIOS}
_EASY = [s for s in SCENARIOS if s["difficulty"] == "easy"]
_MEDIUM = [s for s in SCENARIOS if s["difficulty"] == "medium"]
_HARD = [s for s in SCENARIOS if s["difficulty"] == "hard"]

# ── Constants for reward functions ──────────────────────────────────────────

_ACTION_VERBS = re.compile(
    r"\b(reschedule|inform|contact|book|cancel|delegate|arrange)\b",
    re.IGNORECASE,
)
_TIME_REFS = re.compile(
    r"\b(\d{1,2}:\d{2}|\d{1,2}\s*(am|pm|AM|PM)"
    r"|tomorrow|tonight|today|within\s+\d+\s*(hour|minute|min|hr)"
    r"|immediately|right\s+now|asap|by\s+\d{1,2})\b",
    re.IGNORECASE,
)
_GENERIC_PHRASES = re.compile(
    r"(I will try|I apologize for any inconvenience|I'll do my best)",
    re.IGNORECASE,
)


# ── Environment ─────────────────────────────────────────────────────────────


class LifeOSEnvironment(Environment):
    """OpenEnv RL environment that simulates life-management crises.

    The agent must resolve interpersonal / logistical conflicts through
    targeted communication actions.  Reward is decomposed into five
    interpretable, independently computed components.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── lifecycle ───────────────────────────────────────────────────────

    def __init__(self) -> None:
        super().__init__()
        # Curriculum tracking
        self._total_episodes: int = 0
        self._curriculum_stage: str = "easy"  # easy → medium → hard

        # Per-episode state
        self._scenario: dict[str, Any] = {}
        self._active_conflicts: list[str] = []
        self._persona_responses: dict[str, str] = {}
        self._previous_content: str = ""
        self._state = LifeOSState(scenario_id="", difficulty="easy")

    # ── reset ───────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LifeOSObservation:
        """Pick a scenario based on curriculum stage and return the first observation."""
        self._reset_rubric()
        self._advance_curriculum()

        scenario = self._pick_scenario(seed)
        self._scenario = scenario
        self._active_conflicts = list(scenario["conflicts"])
        self._persona_responses = {
            name: "Awaiting your action."
            for name in scenario["personas"]
        }
        self._previous_content = ""

        self._state = LifeOSState(
            scenario_id=scenario["id"],
            difficulty=scenario["difficulty"],
            step_count=0,
            max_steps=10,
            conflicts_total=len(scenario["conflicts"]),
            conflicts_resolved=0,
            done=False,
            cumulative_reward=0.0,
            reward_history=[],
        )

        return self._build_observation(reward_breakdown={})

    # ── step ────────────────────────────────────────────────────────────

    def step(
        self,
        action: LifeOSAction,  # type: ignore[override]
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> tuple[LifeOSObservation, float, bool]:
        """Execute one agent action, compute decomposed reward, advance state."""
        self._state.step_count += 1

        # Anti-hacking: penalise repeated identical content
        if action.content == self._previous_content:
            reward_breakdown = {
                "conflict_addressed": 0.0,
                "stakeholder_reached": 0.0,
                "action_specificity": 0.0,
                "format_compliance": 0.0,
                "no_escalation": 0.0,
            }
            total_reward = 0.0
        else:
            reward_breakdown = {
                "conflict_addressed": self._reward_conflict_addressed(action),
                "stakeholder_reached": self._reward_stakeholder_reached(action),
                "action_specificity": self._reward_action_specificity(action),
                "format_compliance": self._reward_format_compliance(action),
                "no_escalation": self._reward_no_escalation(action),
            }
            total_reward = sum(reward_breakdown.values())

        self._previous_content = action.content

        # Update resolved conflicts (simple keyword check)
        self._update_resolved_conflicts(action)

        # Simulate persona responses
        self._simulate_persona_responses(action)

        # Accumulate reward
        self._state.cumulative_reward += total_reward
        self._state.reward_history.append(total_reward)

        # Termination conditions
        done = (
            self._state.step_count >= self._state.max_steps
            or self._state.conflicts_resolved >= self._state.conflicts_total
        )
        self._state.done = done

        observation = self._build_observation(reward_breakdown)
        return observation, total_reward, done

    # ── state property ──────────────────────────────────────────────────

    @property
    def state(self) -> LifeOSState:
        """Return a copy of the current environment state."""
        return self._state.model_copy()

    # ── reward functions (five independent components) ──────────────────

    def _reward_conflict_addressed(self, action: LifeOSAction) -> float:
        """0–0.30: Does the action content reference an active conflict keyword?"""
        content_lower = action.content.lower()
        for conflict in self._active_conflicts:
            # Check each word of the conflict identifier
            keywords = conflict.replace("_", " ").split()
            if any(kw in content_lower for kw in keywords):
                return 0.30
        return 0.0

    def _reward_stakeholder_reached(self, action: LifeOSAction) -> float:
        """0–0.25: Is the target person one of the scenario's personas?"""
        if action.target_person in self._scenario.get("personas", {}):
            return 0.25
        return 0.0

    def _reward_action_specificity(self, action: LifeOSAction) -> float:
        """0–0.20: Content contains both a time reference AND an action verb."""
        has_verb = bool(_ACTION_VERBS.search(action.content))
        has_time = bool(_TIME_REFS.search(action.content))
        if has_verb and has_time:
            return 0.20
        if has_verb or has_time:
            return 0.10
        return 0.0

    def _reward_format_compliance(self, action: LifeOSAction) -> float:
        """0–0.15: Reasoning is substantive (>30 chars) and urgency is valid."""
        valid_urgencies = {"immediate", "within_hour", "today", "tomorrow"}
        reasoning_ok = len(action.reasoning) > 30
        urgency_ok = action.urgency in valid_urgencies
        if reasoning_ok and urgency_ok:
            return 0.15
        if reasoning_ok or urgency_ok:
            return 0.07
        return 0.0

    def _reward_no_escalation(self, action: LifeOSAction) -> float:
        """0–0.10: Penalise generic filler phrases."""
        if _GENERIC_PHRASES.search(action.content):
            return 0.0
        return 0.10

    # ── helpers ──────────────────────────────────────────────────────────

    def _advance_curriculum(self) -> None:
        """Move the curriculum stage forward based on total episodes completed."""
        self._total_episodes += 1
        if self._total_episodes <= 3:
            self._curriculum_stage = "easy"
        elif self._total_episodes <= 6:
            self._curriculum_stage = "medium"
        else:
            self._curriculum_stage = "hard"

    def _pick_scenario(self, seed: Optional[int] = None) -> dict[str, Any]:
        """Select a scenario matching the current curriculum stage."""
        import random

        pool = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}
        candidates = pool.get(self._curriculum_stage, _EASY)

        rng = random.Random(seed)
        return rng.choice(candidates)

    def _update_resolved_conflicts(self, action: LifeOSAction) -> None:
        """Mark a conflict as resolved if the action addresses it directly."""
        content_lower = action.content.lower()
        newly_resolved: list[str] = []
        for conflict in self._active_conflicts:
            keywords = conflict.replace("_", " ").split()
            if any(kw in content_lower for kw in keywords):
                newly_resolved.append(conflict)

        for conflict in newly_resolved:
            self._active_conflicts.remove(conflict)
            self._state.conflicts_resolved += 1

    def _simulate_persona_responses(self, action: LifeOSAction) -> None:
        """Generate simple deterministic persona responses based on the action."""
        target = action.target_person
        personas = self._scenario.get("personas", {})

        if target in personas:
            # Targeted persona acknowledges the action
            self._persona_responses[target] = (
                f"Acknowledged your {action.action_type}. "
                f"I'll factor this into my plans."
            )
        else:
            # Non-targeted personas remain waiting
            for name in personas:
                if name not in self._persona_responses or self._persona_responses[name] == "Awaiting your action.":
                    self._persona_responses[name] = "Still waiting for your response."

    def _build_observation(
        self,
        reward_breakdown: dict[str, float],
    ) -> LifeOSObservation:
        """Construct an observation from the current environment state."""
        scenario = self._scenario
        time_pressure = self._infer_time_pressure()

        return LifeOSObservation(
            scenario_description=scenario.get("trigger", ""),
            active_conflicts=list(self._active_conflicts),
            persona_responses=dict(self._persona_responses),
            time_pressure=time_pressure,
            step_number=self._state.step_count,
            reward_breakdown=reward_breakdown,
        )

    def _infer_time_pressure(self) -> str:
        """Derive a human-readable time-pressure label from the state."""
        remaining = self._state.max_steps - self._state.step_count
        if remaining <= 2:
            return "critical – almost out of steps"
        if remaining <= 5:
            return "high – limited steps remaining"
        return "moderate – enough steps to plan carefully"
