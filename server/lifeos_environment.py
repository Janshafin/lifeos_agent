"""
LifeOS Agent Environment – core OpenEnv RL environment.

Presents life-management crisis scenarios at three difficulty tiers and
rewards agents for conflict-resolution quality via five independent,
interpretable reward functions.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

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
        "trigger": (
            "Your current meeting has overrun by 30 minutes. Your next meeting "
            "starts right now with an important client who is already waiting."
        ),
        "conflicts": ["scheduling_overlap"],
        "personas": {
            "Alice_Client": "Your next meeting organiser – punctual, expects you on time, easily offended by lateness.",
            "Bob_Colleague": "Current meeting lead – tends to ramble, unaware of your schedule.",
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
        "trigger": (
            "An important client called during your meeting. You must call "
            "back within the hour or risk losing the deal."
        ),
        "conflicts": ["missed_client_call"],
        "personas": {
            "Client_Sarah": "Senior client – values responsiveness, easily offended by delays.",
            "Manager_Tom": "Your manager – wants to know about all client interactions immediately.",
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
        "trigger": (
            "A team member needs urgent help with a critical blocker, but you "
            "are exactly 1 hour from your own hard deadline."
        ),
        "conflicts": ["team_request_conflict"],
        "personas": {
            "Junior_Dev": "New hire – anxious, blocked on a critical bug, first week on the job.",
            "PM_Rachel": "Project manager – tracking your deliverable closely, zero tolerance for slips.",
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
            "Your flight is delayed by 3 hours. Your partner is already "
            "waiting at the airport to pick you up. You have a dinner "
            "reservation in 2 hours that will be forfeited if you no-show."
        ),
        "conflicts": ["travel_delay", "dinner_reservation"],
        "personas": {
            "Partner_Jamie": "Your partner – worried, already at the airport waiting for you.",
            "Restaurant_Host": "Upscale restaurant – strict cancellation policy, no refunds.",
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
            "Your boss demands a report delivered in 1 hour. Your child's "
            "school just called about an emergency – your child fell and may "
            "need stitches. A critical client call starts in 45 minutes."
        ),
        "conflicts": ["boss_report_deadline", "child_emergency"],
        "personas": {
            "Boss_Karen": "VP – impatient, expects immediate compliance, tracks deadlines to the minute.",
            "School_Nurse": "School staff – calm but firm, your child has a minor injury needing attention.",
            "Client_VP": "Senior stakeholder – calling in 45 minutes for a deal review, cannot be rescheduled.",
        },
        "success_criteria": [
            "Address the child emergency immediately",
            "Negotiate a short extension on the report",
            "Prepare for the upcoming client call",
        ],
    },
    {
        "id": "medium_03",
        "title": "Double-Booked VPs",
        "difficulty": "medium",
        "trigger": (
            "You have double-booked two VP-level meetings at the exact same "
            "time. Both VPs expect you to attend in person. Neither meeting "
            "can be easily rescheduled."
        ),
        "conflicts": ["client_meeting_A", "client_meeting_B"],
        "personas": {
            "VP_Product": "Fortune-500 VP – low tolerance for schedule changes, controls your budget.",
            "VP_Engineering": "Key account director – relationship is fragile, considering competitors.",
            "Your_EA": "Executive assistant – apologetic, made the booking error, eager to help fix it.",
        },
        "success_criteria": [
            "Reschedule one meeting without damaging the relationship",
            "Attend or delegate the other meeting",
        ],
    },
    # ── HARD  (3–4 conflicts each) ──────────────────────────────────────────
    {
        "id": "hard_01",
        "title": "Travel Meltdown",
        "difficulty": "hard",
        "trigger": (
            "Your flight has been cancelled. You have a 9am meeting tomorrow "
            "in another city that you absolutely cannot miss. Your partner is "
            "sitting alone at a restaurant waiting for you right now. Every "
            "hotel near the meeting venue is sold out. Your boss doesn't know "
            "you might miss the meeting."
        ),
        "conflicts": [
            "cancelled_flight",
            "partner_dinner",
            "hotel_booking",
            "boss_communication",
        ],
        "personas": {
            "Partner_Jamie": "At the restaurant alone – upset, worried, texting you repeatedly.",
            "Boss_Mark": "C-suite – expects you in person tomorrow morning, no excuses.",
            "Airline_Agent": "Only alternative is a red-eye with a layover through Dallas.",
            "Hotel_Concierge": "All hotels near the venue are fully booked for a conference.",
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
            "Your key team member quit without notice this morning. A major "
            "client deliverable is due at 5pm today – they were leading it. "
            "Your own presentation to the CTO is in 2 hours. An intern who "
            "started this week is stuck and needs guidance right now."
        ),
        "conflicts": [
            "team_member_quit",
            "client_deliverable",
            "presentation_prep",
        ],
        "personas": {
            "Client_Director": "Expecting deliverable by 5pm EOD – no extensions, contract depends on it.",
            "Intern_Alex": "Overwhelmed – first week on the job, blocked on a critical task.",
            "CTO": "Attending your presentation in 2 hours – high visibility, career-defining.",
            "HR_Lead": "Needs to process the resignation paperwork and begin backfill.",
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
            "Budget cuts of 30% were just announced mid-project. Three client "
            "contracts are now at risk of cancellation. Team morale has "
            "collapsed – two senior engineers are openly discussing leaving. "
            "You have a board presentation in 48 hours."
        ),
        "conflicts": [
            "budget_cut",
            "client_contracts",
            "team_morale",
        ],
        "personas": {
            "CFO": "Delivered the budget cut – open to creative proposals if you act fast.",
            "Client_A": "Largest account – threatening to leave if project scope shrinks.",
            "Client_B": "Mid-tier account – flexible but needs reassurance immediately.",
            "Client_C": "New account – actively considering competitor proposals.",
            "Team_Lead": "Demoralised – key engineers talking about quitting this week.",
        },
        "success_criteria": [
            "Propose a revised budget allocation to the CFO",
            "Retain at least two of three client contracts",
            "Stabilise team morale with a concrete action plan",
            "Prepare a credible board presentation narrative",
        ],
    },
]

# Lookup helpers
_SCENARIOS_BY_ID: dict[str, dict[str, Any]] = {s["id"]: s for s in SCENARIOS}
_EASY = [s for s in SCENARIOS if s["difficulty"] == "easy"]
_MEDIUM = [s for s in SCENARIOS if s["difficulty"] == "medium"]
_HARD = [s for s in SCENARIOS if s["difficulty"] == "hard"]


# ── Constants for reward functions ──────────────────────────────────────────

ACTION_VERBS = [
    "reschedule", "inform", "contact", "book", "cancel", "delegate",
    "arrange", "call", "email", "message", "notify", "confirm", "move",
    "propose",
]

TIME_REFS = [
    "minute", "hour", "today", "tomorrow", "morning", "evening",
    "now", "immediately", "asap", "am", "pm",
]

GENERIC_PHRASES = [
    "i will try my best",
    "i apologize for any inconvenience",
    "i'll do my best",
    "i'm sorry for the trouble",
    "as soon as possible",
    "i will get back to you",
]

VALID_URGENCIES = {"immediate", "within_hour", "today", "tomorrow"}


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
        self._curriculum: dict[str, Any] = {
            "easy_done": 0,
            "medium_done": 0,
            "stage": "easy",
        }

        # Per-episode state
        self._scenario: dict[str, Any] = {}
        self._active_conflicts: list[str] = []
        self._persona_responses: dict[str, str] = {}
        self._prev_content: str = ""
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

        # Advance curriculum
        if self._curriculum["easy_done"] < 8:
            self._curriculum["stage"] = "easy"
            self._curriculum["easy_done"] += 1
        elif self._curriculum["medium_done"] < 8:
            self._curriculum["stage"] = "medium"
            self._curriculum["medium_done"] += 1
        else:
            self._curriculum["stage"] = "hard"

        # Pick scenario from current stage
        pool = {
            "easy": _EASY,
            "medium": _MEDIUM,
            "hard": _HARD,
        }
        candidates = pool.get(self._curriculum["stage"], _EASY)
        rng = random.Random(seed)
        scenario = rng.choice(candidates)

        self._scenario = scenario
        self._active_conflicts = list(scenario["conflicts"])
        self._persona_responses = {
            name: "Awaiting your action."
            for name in scenario["personas"]
        }
        self._prev_content = ""

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

        # ── Anti-hacking: duplicate content ─────────────────────────────
        if action.content.strip() == self._prev_content.strip() and self._prev_content != "":
            reward_breakdown = {
                "conflict_addressed": 0.0,
                "stakeholder_reached": 0.0,
                "action_specificity": 0.0,
                "format_compliance": 0.0,
                "no_escalation": 0.0,
            }
            total_reward = 0.0
            self._prev_content = action.content
            self._state.reward_history.append(total_reward)
            done = self._state.step_count >= self._state.max_steps
            self._state.done = done
            return self._build_observation(reward_breakdown), total_reward, done

        # ── Anti-hacking: too short ─────────────────────────────────────
        if len(action.content) < 30:
            reward_breakdown = {
                "conflict_addressed": 0.0,
                "stakeholder_reached": 0.0,
                "action_specificity": 0.0,
                "format_compliance": 0.0,
                "no_escalation": 0.0,
            }
            total_reward = 0.0
            self._prev_content = action.content
            self._state.reward_history.append(total_reward)
            done = self._state.step_count >= self._state.max_steps
            self._state.done = done
            return self._build_observation(reward_breakdown), total_reward, done

        # ── Compute five independent reward components ──────────────────
        reward_breakdown = {
            "conflict_addressed": self._reward_conflict_addressed(action, self._scenario),
            "stakeholder_reached": self._reward_stakeholder_reached(action, self._scenario),
            "action_specificity": self._reward_action_specificity(action),
            "format_compliance": self._reward_format_compliance(action),
            "no_escalation": self._reward_no_escalation(action),
        }
        total_reward = min(sum(reward_breakdown.values()), 1.0)

        # Store for duplicate detection
        self._prev_content = action.content

        # Update resolved conflicts
        self._update_resolved_conflicts(action)

        # Simulate persona responses
        self._simulate_persona_responses(action)

        # Accumulate reward
        self._state.cumulative_reward += total_reward
        self._state.reward_history.append(total_reward)

        # Termination
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

    # ── Five independent reward functions ────────────────────────────────

    def _reward_conflict_addressed(
        self, action: LifeOSAction, scenario: dict[str, Any]
    ) -> float:
        """0–0.30: Does the action content reference an active conflict keyword?"""
        content_lower = action.content.lower()
        for conflict in self._active_conflicts:
            keywords = conflict.replace("_", " ").split()
            if any(kw in content_lower for kw in keywords):
                return 0.30
        # Partial credit for substantive content
        if len(action.content) > 30:
            return 0.05
        return 0.0

    def _reward_stakeholder_reached(
        self, action: LifeOSAction, scenario: dict[str, Any]
    ) -> float:
        """0–0.25: Is the target person one of the scenario's personas?"""
        if action.target_person in scenario.get("personas", {}):
            return 0.25
        # Partial credit for non-empty target
        if action.target_person.strip():
            return 0.05
        return 0.0

    def _reward_action_specificity(self, action: LifeOSAction) -> float:
        """0–0.20: Content contains both a time reference AND an action verb."""
        content_lower = action.content.lower()
        has_verb = any(verb in content_lower for verb in ACTION_VERBS)
        has_time = any(ref in content_lower for ref in TIME_REFS)
        if has_verb and has_time:
            return 0.20
        if has_verb or has_time:
            return 0.10
        return 0.0

    def _reward_format_compliance(self, action: LifeOSAction) -> float:
        """0–0.15: Reasoning is substantive (>40 chars) and urgency is valid."""
        reasoning_ok = len(action.reasoning) > 40
        urgency_ok = action.urgency in VALID_URGENCIES
        if reasoning_ok and urgency_ok:
            return 0.15
        if reasoning_ok or urgency_ok:
            return 0.07
        return 0.0

    def _reward_no_escalation(self, action: LifeOSAction) -> float:
        """0–0.10: Penalise generic filler phrases."""
        content_lower = action.content.lower()
        for phrase in GENERIC_PHRASES:
            if phrase in content_lower:
                return 0.0
        return 0.10

    # ── helpers ──────────────────────────────────────────────────────────

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
            self._persona_responses[target] = (
                f"Acknowledged your {action.action_type}. "
                f"I'll factor this into my plans."
            )
        else:
            for name in personas:
                if (
                    name not in self._persona_responses
                    or self._persona_responses[name] == "Awaiting your action."
                ):
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
