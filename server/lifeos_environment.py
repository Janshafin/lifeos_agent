# Copyright (c) LifeOS Team 2026. All rights reserved.
# BSD-3-Clause License

"""
LifeOS Agent Environment — Core RL environment.

Trains LLMs to handle cascading personal life crises through structured
actions across 9 scenarios in 3 difficulty tiers, scored by 5 independent
reward functions with curriculum learning and anti-reward-hacking guards.

Subclasses openenv.core.Environment for full OpenEnv compatibility.
"""

from __future__ import annotations

import random
from typing import Any

from openenv.core import Environment

try:
    from ..models import LifeOSAction, LifeOSObservation, LifeOSState
except (ImportError, ModuleNotFoundError):
    from models import LifeOSAction, LifeOSObservation, LifeOSState


# ══════════════════════════════════════════════════════════════════════════
# SCENARIOS — 9 total across 3 difficulty tiers
# ══════════════════════════════════════════════════════════════════════════

SCENARIOS: dict[str, list[dict[str, Any]]] = {
    "easy": [
        {
            "id": "easy_01",
            "title": "Meeting Overrun",
            "difficulty": "easy",
            "trigger": (
                "Your current meeting has overrun by 30 minutes. Your next "
                "meeting starts right now with an important client who is "
                "already waiting in the conference room."
            ),
            "conflicts": ["scheduling_overlap"],
            "personas": {
                "Alice_Client": "punctual, values professionalism, expects you on time",
                "Bob_Colleague": "long-winded, unaware of your schedule, mid-presentation",
            },
            "success_criteria": [
                "Inform Alice about the delay with a specific timeframe",
                "Gracefully exit the overrun meeting without offending Bob",
            ],
        },
        {
            "id": "easy_02",
            "title": "Missed Client Call",
            "difficulty": "easy",
            "trigger": (
                "An important client called while you were in a meeting. They "
                "left a voicemail saying they need to discuss a contract change "
                "urgently. You must call back within the hour or risk losing "
                "the deal entirely."
            ),
            "conflicts": ["missed_client_call"],
            "personas": {
                "Client_Director": "impatient, high-value account, considering competitors",
                "PM_Rachel": "your project manager, needs to know about contract changes",
            },
            "success_criteria": [
                "Call the client back with a specific plan",
                "Loop in the project manager on potential contract changes",
            ],
        },
        {
            "id": "easy_03",
            "title": "Team Blocker",
            "difficulty": "easy",
            "trigger": (
                "A team member needs urgent help with a critical blocker, but "
                "you are exactly 1 hour from your own deadline on a separate "
                "deliverable. They cannot proceed without your input."
            ),
            "conflicts": ["team_request_conflict"],
            "personas": {
                "Junior_Dev": "stressed, blocked for 3 hours, feels ignored",
                "PM_Rachel": "tracking both deliverables, needs status updates",
            },
            "success_criteria": [
                "Unblock the team member with actionable guidance",
                "Protect your own deadline with a concrete plan",
            ],
        },
    ],
    "medium": [
        {
            "id": "medium_01",
            "title": "Travel Delay Cascade",
            "difficulty": "medium",
            "trigger": (
                "Your flight has been delayed by 3 hours. Your partner is "
                "already at the airport waiting to pick you up. You have a "
                "dinner reservation at an exclusive restaurant in 2 hours "
                "that took 3 months to book — they will give away your table "
                "after 15 minutes."
            ),
            "conflicts": ["flight_delay", "dinner_reservation_at_risk"],
            "personas": {
                "Partner_Jamie": "excited about dinner, drove 45 minutes to airport, easily upset",
                "Restaurant_Host": "strict policy, 3-month waitlist, no exceptions",
            },
            "success_criteria": [
                "Inform partner with empathy and a backup plan",
                "Contact restaurant to save or reschedule the reservation",
            ],
        },
        {
            "id": "medium_02",
            "title": "Work-Family Collision",
            "difficulty": "medium",
            "trigger": (
                "Your boss needs a critical report delivered in 1 hour. "
                "Your child's school just called — your kid fell on the "
                "playground and needs to be picked up immediately. You also "
                "have a client call starting in 45 minutes that you are "
                "leading."
            ),
            "conflicts": ["boss_report_deadline", "family_emergency"],
            "personas": {
                "Boss_Karen": "demanding, no tolerance for missed deadlines, expects results",
                "School_Nurse": "concerned, legally needs a guardian present within 30 minutes",
                "Client_VP": "important client, this call determines contract renewal",
            },
            "success_criteria": [
                "Address the family emergency as top priority",
                "Delegate or reschedule work commitments with specific handoff plans",
            ],
        },
        {
            "id": "medium_03",
            "title": "Double-Booked VPs",
            "difficulty": "medium",
            "trigger": (
                "You are double-booked for two VP-level meetings that start "
                "right now. VP of Sales expects you to present Q3 numbers. "
                "VP of Engineering expects you to demo the new feature. Both "
                "meetings are in different buildings. Both VPs have short "
                "tempers and will take it personally if you skip theirs."
            ),
            "conflicts": ["vp_sales_meeting", "vp_engineering_meeting"],
            "personas": {
                "VP_Sales": "competitive, takes attendance personally, holds grudges",
                "VP_Engineering": "technical, prepared extensive demo, booked this 2 weeks ago",
                "Your_Manager": "caught in the middle, trying to avoid political fallout",
            },
            "success_criteria": [
                "Attend or delegate one meeting with a credible representative",
                "Personally handle the higher-stakes meeting with preparation",
            ],
        },
    ],
    "hard": [
        {
            "id": "hard_01",
            "title": "Total Travel Meltdown",
            "difficulty": "hard",
            "trigger": (
                "Your flight has been cancelled entirely — no rebooking available "
                "until tomorrow afternoon. You have a 9am board meeting tomorrow "
                "in another city that you are presenting at. Your partner is "
                "sitting alone at a restaurant across town — you were supposed "
                "to be there 40 minutes ago and they are furious. Every hotel "
                "near the meeting venue is sold out. Your boss does not know "
                "any of this yet."
            ),
            "conflicts": [
                "flight_cancelled",
                "partner_waiting",
                "hotel_unavailable",
                "boss_uninformed",
            ],
            "personas": {
                "Partner_Jamie": "furious, has been texting for 40 minutes, considering leaving",
                "Boss_Karen": "expects you in person tomorrow, will question your reliability",
                "Airline_Agent": "overwhelmed, processing hundreds of cancellations",
                "Hotel_Concierge": "apologetic but fully booked, might know alternatives",
            },
            "success_criteria": [
                "Message partner immediately with honesty and a concrete plan",
                "Inform boss with a backup plan (virtual presentation option)",
                "Find alternative transportation (red-eye bus, rental car, train)",
                "Secure accommodation near the meeting venue",
            ],
        },
        {
            "id": "hard_02",
            "title": "Team Collapse",
            "difficulty": "hard",
            "trigger": (
                "Your key team member quit without notice this morning. A major "
                "client deliverable is due at 5pm today — they were the lead on "
                "it. You have your own board presentation to prepare for in 2 "
                "hours. The intern who was shadowing the departing employee is "
                "completely stuck and panicking. HR needs you to do an exit "
                "interview."
            ),
            "conflicts": [
                "team_member_quit",
                "client_deliverable",
                "presentation_prep",
            ],
            "personas": {
                "Client_Director": "expecting delivery at 5pm sharp, contract depends on it",
                "Intern_Alex": "panicking, has some context but not enough to finish alone",
                "CTO": "concerned about team stability, wants a retention plan by EOD",
                "HR_Lead": "needs exit paperwork completed, compliance deadline",
            },
            "success_criteria": [
                "Take ownership of the client deliverable with a specific plan",
                "Give the intern actionable guidance to contribute",
                "Brief the CTO with a team continuity plan",
            ],
        },
        {
            "id": "hard_03",
            "title": "Budget Crisis Firestorm",
            "difficulty": "hard",
            "trigger": (
                "Budget cuts of 30% were just announced mid-project. Three "
                "active client contracts are now at risk because you cannot "
                "deliver the promised scope. Your team of 8 just heard the "
                "news and morale has collapsed — two senior engineers are "
                "updating their resumes. You have a board presentation in 48 "
                "hours where you need to present a revised plan. A tech "
                "journalist has somehow found out and is asking for comment."
            ),
            "conflicts": [
                "budget_cuts",
                "client_contracts_at_risk",
                "team_morale_collapsed",
                "press_inquiry",
            ],
            "personas": {
                "CFO": "made the cut decision, open to hearing revised plans with ROI data",
                "Client_A_Lead": "biggest account, will leave if scope is reduced without negotiation",
                "Senior_Engineer_1": "top performer, has an offer from a competitor, on the fence",
                "Journalist": "tech press, writing a story, deadline in 24 hours",
                "Board_Chair": "needs confidence that leadership has a plan",
            },
            "success_criteria": [
                "Negotiate with CFO using data-driven revised scope",
                "Proactively contact at-risk clients with renegotiation options",
                "Retain key engineers with concrete commitments",
                "Manage press inquiry to control the narrative",
            ],
        },
    ],
}

# ══════════════════════════════════════════════════════════════════════════
# REWARD CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

ACTION_VERBS = [
    "reschedule", "inform", "contact", "book", "cancel", "delegate",
    "arrange", "call", "email", "message", "notify", "confirm", "move",
    "propose", "apologize", "explain", "update", "brief", "coordinate",
    "negotiate", "offer", "send", "draft", "prepare", "escalate",
    "rebook", "transfer", "assign", "prioritize", "defer",
]

TIME_REFERENCES = [
    "minute", "hour", "today", "tomorrow", "morning", "afternoon",
    "evening", "tonight", "now", "immediately", "asap", "urgent",
    "9am", "5pm", "am", "pm", "within", "deadline", "by",
    "noon", "midnight", "eod", "eob", "before", "after",
]

GENERIC_PHRASES = [
    "i will try my best",
    "i apologize for any inconvenience",
    "i ll do my best",
    "i m sorry for the trouble",
    "as soon as possible",
    "i will get back to you",
    "i understand your concern",
    "i will look into this",
]

VALID_URGENCY = ["immediate", "within_hour", "today", "tomorrow"]

TIME_PRESSURE_BY_DIFFICULTY = {
    "easy": "medium",
    "medium": "high",
    "hard": "critical",
}


# ══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CLASS
# ══════════════════════════════════════════════════════════════════════════


class LifeOSEnvironment(Environment):
    """OpenEnv-compliant RL environment for personal crisis management.

    The agent observes a crisis scenario with active conflicts, persona
    descriptions, and time pressure. It must respond with a structured
    action: choosing an action type, targeting a specific person, crafting
    actual message content, explaining its reasoning, and declaring urgency.

    Reward is decomposed into 5 independent functions:
      - conflict_addressed (0.30): does the action reference a real conflict?
      - stakeholder_reached (0.25): is the target a real persona?
      - action_specificity (0.20): does content have verbs + time references?
      - format_compliance (0.15): is reasoning substantive + urgency valid?
      - no_escalation (0.10): are generic filler phrases absent?

    Curriculum learning progresses: easy (ep 1-8) → medium (9-16) → hard (17+).
    """

    def __init__(self) -> None:
        super().__init__()
        self._scenario: dict[str, Any] = {}
        self._step_count: int = 0
        self._max_steps: int = 10
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._reward_history: list[float] = []
        self._prev_content: str = ""
        self._last_breakdown: dict[str, float] = {}
        self._episode_count: int = 0

        # Curriculum tracking
        self._curriculum = {
            "easy_done": 0,
            "medium_done": 0,
            "stage": "easy",
        }

    # ──────────────────────────────────────────────────────────────────
    # CURRICULUM LOGIC
    # ──────────────────────────────────────────────────────────────────

    def _get_curriculum_stage(self) -> str:
        """Determine difficulty tier based on episode count."""
        if self._episode_count < 8:
            return "easy"
        elif self._episode_count < 16:
            return "medium"
        else:
            return "hard"

    def _pick_scenario(self, seed: int | None = None) -> dict[str, Any]:
        """Pick a scenario from the current curriculum stage."""
        stage = self._get_curriculum_stage()
        self._curriculum["stage"] = stage
        pool = SCENARIOS[stage]

        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        return rng.choice(pool)

    # ──────────────────────────────────────────────────────────────────
    # RESET
    # ──────────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> LifeOSObservation:
        """Reset the environment with a new crisis scenario.

        Args:
            seed: Optional random seed for reproducible scenario selection.

        Returns:
            Initial observation with scenario description and active conflicts.
        """
        self._episode_count += 1
        self._scenario = self._pick_scenario(seed)
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._reward_history = []
        self._prev_content = ""
        self._last_breakdown = {
            "conflict_addressed": 0.0,
            "stakeholder_reached": 0.0,
            "action_specificity": 0.0,
            "format_compliance": 0.0,
            "no_escalation": 0.0,
        }

        # Build the full scenario description
        persona_descriptions = "; ".join(
            f"{name} ({desc})"
            for name, desc in self._scenario["personas"].items()
        )
        description = (
            f"{self._scenario['trigger']}\n\n"
            f"People involved: {persona_descriptions}\n\n"
            f"Active conflicts: {', '.join(self._scenario['conflicts'])}\n\n"
            f"Success criteria: {'; '.join(self._scenario['success_criteria'])}"
        )

        # Initial persona responses
        initial_responses = {
            name: "Waiting for your response..."
            for name in self._scenario["personas"]
        }

        difficulty = self._scenario["difficulty"]

        return LifeOSObservation(
            scenario_description=description,
            active_conflicts=list(self._scenario["conflicts"]),
            persona_responses=initial_responses,
            time_pressure=TIME_PRESSURE_BY_DIFFICULTY.get(difficulty, "medium"),
            step_number=0,
            reward_breakdown=dict(self._last_breakdown),
            reward=None,
            done=False,
        )

    # ──────────────────────────────────────────────────────────────────
    # FIVE INDEPENDENT REWARD FUNCTIONS
    # ──────────────────────────────────────────────────────────────────

    def _reward_conflict_addressed(
        self, action: LifeOSAction, scenario: dict[str, Any]
    ) -> float:
        """Check if the action references an actual active conflict.

        Extracts keywords from conflict names (e.g. 'flight_cancelled' →
        ['flight', 'cancelled']) and checks if any appear in the content.

        Returns:
            0.30 if a conflict keyword is found in content.
            0.05 if content is substantive (>30 chars) but no keyword match.
            0.00 otherwise.
        """
        content_lower = action.content.lower()
        for conflict in scenario["conflicts"]:
            keywords = conflict.replace("_", " ").split()
            for keyword in keywords:
                if len(keyword) > 2 and keyword in content_lower:
                    return 0.30
        # Partial credit for substantive content
        if len(action.content.strip()) > 30:
            return 0.05
        return 0.0

    def _reward_stakeholder_reached(
        self, action: LifeOSAction, scenario: dict[str, Any]
    ) -> float:
        """Check if the action targets a real persona in the scenario.

        Returns:
            0.25 if target_person matches a persona name (case-insensitive).
            0.05 if target_person is non-empty but doesn't match.
            0.00 if target_person is empty.
        """
        target_lower = action.target_person.lower().strip()
        if not target_lower:
            return 0.0
        for persona_name in scenario["personas"]:
            if persona_name.lower() in target_lower or target_lower in persona_name.lower():
                return 0.25
        # Partial credit for having any target
        if target_lower:
            return 0.05
        return 0.0

    def _reward_action_specificity(self, action: LifeOSAction) -> float:
        """Check if the content contains specific action verbs and time references.

        Returns:
            0.20 if both an action verb and a time reference are present.
            0.10 if only one is present.
            0.00 if neither is present.
        """
        content_lower = action.content.lower()
        has_verb = any(verb in content_lower for verb in ACTION_VERBS)
        has_time = any(time_ref in content_lower for time_ref in TIME_REFERENCES)

        if has_verb and has_time:
            return 0.20
        elif has_verb or has_time:
            return 0.10
        return 0.0

    def _reward_format_compliance(self, action: LifeOSAction) -> float:
        """Check if reasoning is substantive and urgency is valid.

        Returns:
            0.15 if reasoning > 40 chars AND urgency is valid.
            0.07 if only one condition is met.
            0.00 if neither is met.
        """
        good_reasoning = len(action.reasoning.strip()) > 40
        good_urgency = action.urgency in VALID_URGENCY

        if good_reasoning and good_urgency:
            return 0.15
        elif good_reasoning or good_urgency:
            return 0.07
        return 0.0

    def _reward_no_escalation(self, action: LifeOSAction) -> float:
        """Penalize generic filler phrases that LLMs default to.

        Returns:
            0.10 if no generic phrases are found.
            0.00 if any generic phrase is detected.
        """
        content_lower = action.content.lower()
        for phrase in GENERIC_PHRASES:
            if phrase in content_lower:
                return 0.0
        return 0.10

    # ──────────────────────────────────────────────────────────────────
    # STEP
    # ──────────────────────────────────────────────────────────────────

    def step(
        self, action: LifeOSAction, **kwargs: Any,
    ) -> LifeOSObservation:
        """Execute an action and return observation with reward and done.

        Applies anti-reward-hacking guards first, then computes all 5
        reward components independently.

        Args:
            action: The agent's structured action.

        Returns:
            LifeOSObservation with reward and done fields set.
        """
        # Auto-reset if no scenario loaded (HTTP mode creates fresh env per request)
        if not self._scenario:
            self.reset()

        self._step_count += 1
        done = self._step_count >= self._max_steps
        self._done = done
        scenario = self._scenario
        difficulty = scenario.get("difficulty", "easy")

        # ── Anti-reward-hacking guard: duplicate content ──
        if (
            self._prev_content
            and action.content.strip() == self._prev_content.strip()
        ):
            zero_breakdown = {k: 0.0 for k in self._last_breakdown}
            self._last_breakdown = zero_breakdown
            self._reward_history.append(0.0)
            return self._build_observation(scenario, zero_breakdown, difficulty, 0.0, done)

        # ── Anti-reward-hacking guard: minimum content length ──
        if len(action.content.strip()) < 30:
            zero_breakdown = {k: 0.0 for k in self._last_breakdown}
            self._last_breakdown = zero_breakdown
            self._reward_history.append(0.0)
            return self._build_observation(scenario, zero_breakdown, difficulty, 0.0, done)

        # ── Compute all 5 reward components independently ──
        breakdown = {
            "conflict_addressed": self._reward_conflict_addressed(action, scenario),
            "stakeholder_reached": self._reward_stakeholder_reached(action, scenario),
            "action_specificity": self._reward_action_specificity(action),
            "format_compliance": self._reward_format_compliance(action),
            "no_escalation": self._reward_no_escalation(action),
        }

        total_reward = min(sum(breakdown.values()), 1.0)

        # ── Update state ──
        self._prev_content = action.content
        self._last_breakdown = breakdown
        self._cumulative_reward += total_reward
        self._reward_history.append(total_reward)

        return self._build_observation(scenario, breakdown, difficulty, total_reward, done)

    # ──────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────

    def _build_observation(
        self,
        scenario: dict[str, Any],
        breakdown: dict[str, float],
        difficulty: str,
        reward_value: float = 0.0,
        is_done: bool = False,
    ) -> LifeOSObservation:
        """Build an observation from current state."""
        # Generate contextual persona responses based on step
        responses = {}
        for name, personality in scenario.get("personas", {}).items():
            if self._step_count <= 1:
                responses[name] = "Waiting for your response..."
            elif sum(breakdown.values()) > 0.5:
                responses[name] = f"{name} acknowledges your message positively."
            else:
                responses[name] = f"{name} is still waiting for a substantive response."

        return LifeOSObservation(
            scenario_description=scenario.get("trigger", ""),
            active_conflicts=list(scenario.get("conflicts", [])),
            persona_responses=responses,
            time_pressure=TIME_PRESSURE_BY_DIFFICULTY.get(difficulty, "medium"),
            step_number=self._step_count,
            reward_breakdown=dict(breakdown),
            reward=reward_value,
            done=is_done,
        )

    @property
    def state(self) -> LifeOSState:
        """Return the current internal environment state."""
        return LifeOSState(
            scenario_id=self._scenario.get("id", "none"),
            difficulty=self._scenario.get("difficulty", "easy"),
            step_count=self._step_count,
            max_steps=self._max_steps,
            conflicts_total=len(self._scenario.get("conflicts", [])),
            conflicts_resolved=0,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            reward_history=list(self._reward_history),
        )
