# %% [markdown]
# # 🧠 LifeOS Agent — OpenEnv RL Training Notebook
#
# Train an LLM to handle cascading personal life crises using
# curriculum learning and 5 independent reward functions.
#
# **Environment**: LifeOS Agent (OpenEnv)
# **Model**: Qwen2.5-3B-Instruct (4-bit via Unsloth)
# **Method**: REINFORCE-style policy gradient with LoRA
# **Runtime**: Google Colab T4 GPU (free tier)
#
# ⚠️ BEFORE RUNNING: Go to Runtime → Change runtime type → select **T4 GPU**

# %% CELL 0 — Verify GPU is available
# If this cell fails, go to Runtime → Change runtime type → T4 GPU
import torch
assert torch.cuda.is_available(), (
    "❌ No GPU detected! Go to Runtime → Change runtime type → select T4 GPU, "
    "then 'Disconnect and delete runtime', reconnect, and run again."
)
print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")

# %% CELL 1 — Install all dependencies (on FRESH runtime only)
# ⚠️ IMPORTANT: If you hit version conflicts, do Runtime → Disconnect and
# delete runtime, reconnect, then run Cell 0 and Cell 1 on the fresh runtime.
#
# Step 1: Install unsloth (pins torch<2.11, pulls compatible transformers/trl)
# Step 2: Downgrade torchvision to match the torch version unsloth installed
# Step 3: Install remaining deps
# After this cell finishes, RESTART RUNTIME, skip Cell 1, run from Cell 2.

%%capture install_output
!pip install unsloth
!pip install "torchvision==0.25.0" "torchaudio==2.10.0"
!pip install matplotlib numpy
!pip install "git+https://github.com/meta-pytorch/OpenEnv.git"

# %% CELL 1b — Verify install (run after Cell 1)
import importlib
errors = []
for pkg in ["unsloth", "torch", "torchvision", "transformers", "trl", "matplotlib"]:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, "__version__", "ok")
        print(f"  ✅ {pkg}: {v}")
    except Exception as e:
        errors.append(f"  ❌ {pkg}: {e}")
        print(errors[-1])
if errors:
    print("\n⚠️  Some packages failed. Do Runtime → Disconnect and delete runtime, then retry.")
else:
    print("\n✅ All dependencies installed!")
    print("⚠️  Now RESTART runtime (Runtime → Restart runtime), then skip to Cell 2.")

# %% CELL 2 — Full In-Memory Environment (standalone, no OpenEnv imports needed)
import random
import re
import matplotlib.pyplot as plt
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# SCENARIOS — 9 total across 3 difficulty tiers
# ══════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "easy": [
        {
            "id": "e1",
            "title": "Meeting Overrun",
            "difficulty": "easy",
            "trigger": (
                "Your current meeting has overrun by 30 minutes. Your next "
                "meeting starts right now with an important client who is "
                "already waiting."
            ),
            "conflicts": ["scheduling_overlap"],
            "personas": {
                "Alice_Client": "punctual, expects you on time",
                "Bob_Colleague": "rambles, unaware of your schedule",
            },
            "success_criteria": [
                "Inform the next meeting organiser of the delay",
                "Gracefully exit the current meeting",
            ],
        },
        {
            "id": "e2",
            "title": "Missed Client Call",
            "difficulty": "easy",
            "trigger": (
                "An important client called during your meeting. You must "
                "call back within the hour or risk losing the deal."
            ),
            "conflicts": ["missed_client_call"],
            "personas": {
                "Client_Sarah": "values responsiveness, easily offended",
                "Manager_Tom": "wants to know about all client interactions",
            },
            "success_criteria": [
                "Return the client call promptly",
                "Inform your manager about the interaction",
            ],
        },
        {
            "id": "e3",
            "title": "Team Help Request",
            "difficulty": "easy",
            "trigger": (
                "A team member needs urgent help with a critical blocker, "
                "but you are exactly 1 hour from your own hard deadline."
            ),
            "conflicts": ["team_request_conflict"],
            "personas": {
                "Junior_Dev": "anxious, blocked on a critical bug",
                "PM_Rachel": "tracking your deliverable closely",
            },
            "success_criteria": [
                "Acknowledge the team member's request",
                "Protect your own deadline",
            ],
        },
    ],
    "medium": [
        {
            "id": "m1",
            "title": "Flight Delay Cascade",
            "difficulty": "medium",
            "trigger": (
                "Your flight is delayed by 3 hours. Your partner is already "
                "waiting at the airport. You have a dinner reservation in 2 "
                "hours that will be forfeited if you no-show."
            ),
            "conflicts": ["travel_delay", "dinner_reservation"],
            "personas": {
                "Partner_Jamie": "worried, already at the airport",
                "Restaurant_Host": "strict cancellation policy",
                "Airline_Agent": "overworked, limited rebooking options",
            },
            "success_criteria": [
                "Update partner on new arrival time",
                "Reschedule or cancel the dinner reservation",
            ],
        },
        {
            "id": "m2",
            "title": "Triple Collision",
            "difficulty": "medium",
            "trigger": (
                "Your boss demands a report in 1 hour. Your child's school "
                "just called about an emergency. A critical client call "
                "starts in 45 minutes."
            ),
            "conflicts": ["work_deadline", "family_emergency"],
            "personas": {
                "Boss_Karen": "demanding, expects immediate compliance",
                "School_Nurse": "calm but firm, child has minor injury",
                "Client_VP": "senior stakeholder, calling in 45 minutes",
            },
            "success_criteria": [
                "Address the child emergency immediately",
                "Negotiate a short extension on the report",
            ],
        },
        {
            "id": "m3",
            "title": "Double-Booked VPs",
            "difficulty": "medium",
            "trigger": (
                "You have double-booked two VP-level meetings at the exact "
                "same time. Both VPs expect you to attend. Neither meeting "
                "can be easily rescheduled."
            ),
            "conflicts": ["scheduling_conflict", "stakeholder_management"],
            "personas": {
                "VP_Product": "low tolerance for schedule changes",
                "VP_Engineering": "relationship is fragile",
            },
            "success_criteria": [
                "Reschedule one meeting without damaging the relationship",
                "Attend or delegate the other meeting",
            ],
        },
    ],
    "hard": [
        {
            "id": "h1",
            "title": "Travel Meltdown",
            "difficulty": "hard",
            "trigger": (
                "Your flight has been cancelled. You have a 9am meeting "
                "tomorrow in another city. Your partner is at a restaurant "
                "waiting. Your hotel is sold out. Your boss doesn't know yet."
            ),
            "conflicts": [
                "travel_crisis",
                "partner_dinner",
                "accommodation",
                "boss_communication",
            ],
            "personas": {
                "Partner_Jamie": "upset, at restaurant alone",
                "Boss_Mark": "strict, expects you in person tomorrow",
                "Airline_Agent": "only option is a red-eye with layover",
                "Hotel_Concierge": "all hotels fully booked",
            },
            "success_criteria": [
                "Secure alternative travel",
                "Communicate with partner",
                "Inform boss with backup plan",
                "Find accommodation",
            ],
        },
        {
            "id": "h2",
            "title": "Team Crisis",
            "difficulty": "hard",
            "trigger": (
                "A key team member quit without notice. A major client "
                "deliverable is due at 5pm. Your own presentation is in "
                "2 hours. An intern is stuck and needs guidance now."
            ),
            "conflicts": [
                "team_loss",
                "client_deadline",
                "personal_presentation",
                "intern_blocker",
            ],
            "personas": {
                "Client_Director": "expecting deliverable by 5pm EOD",
                "Intern_Alex": "overwhelmed, first week",
                "CTO": "attending your presentation",
            },
            "success_criteria": [
                "Redistribute the quitting member's work",
                "Deliver or negotiate the client deliverable",
                "Prepare for the presentation",
                "Guide the intern",
            ],
        },
        {
            "id": "h3",
            "title": "Budget Crisis",
            "difficulty": "hard",
            "trigger": (
                "Budget cuts of 30% were just announced mid-project. 3 "
                "client contracts are at risk. Team morale has collapsed. "
                "Board presentation in 48 hours."
            ),
            "conflicts": [
                "budget_crisis",
                "client_retention",
                "team_morale",
                "board_presentation",
            ],
            "personas": {
                "CFO": "open to creative proposals",
                "Client_A": "threatening to leave",
                "Client_B": "flexible but needs reassurance",
                "Client_C": "considering competitors",
                "Team_Lead": "demoralized, engineers may quit",
            },
            "success_criteria": [
                "Propose revised budget allocation",
                "Retain at least two client contracts",
                "Stabilise team morale",
                "Prepare board presentation",
            ],
        },
    ],
}

# ══════════════════════════════════════════════════════════════════════
# FIVE INDEPENDENT REWARD FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

GENERIC_PHRASES = [
    "i will try my best",
    "i apologize for any inconvenience",
    "i'll do my best",
    "i'm sorry for the trouble",
    "as soon as possible",
    "i will get back to you",
]

ACTION_VERBS = [
    "reschedule", "inform", "contact", "book", "cancel", "delegate",
    "arrange", "call", "email", "message", "notify", "confirm", "move",
    "propose",
]

TIME_REFS = [
    "minute", "hour", "today", "tomorrow", "morning", "evening",
    "now", "immediately", "asap", "am", "pm",
]

VALID_URGENCIES = {"immediate", "within_hour", "today", "tomorrow"}


def r_conflict_addressed(content: str, scenario: dict) -> float:
    """0.00–0.30: Does the response mention any active conflict keyword?"""
    content_lower = content.lower()
    for conflict in scenario["conflicts"]:
        keywords = conflict.replace("_", " ").split()
        if any(kw in content_lower for kw in keywords):
            return 0.30
    if len(content) > 30:
        return 0.05
    return 0.0


def r_stakeholder_reached(target_person: str, scenario: dict) -> float:
    """0.00–0.25: Did the response target a real persona?"""
    if not target_person:
        return 0.0
    for persona in scenario["personas"]:
        if persona.lower() in target_person.lower() or target_person.lower() in persona.lower():
            return 0.25
    if target_person.strip():
        return 0.05
    return 0.0


def r_action_specificity(content: str) -> float:
    """0.00–0.20: Does the response contain a time reference AND an action verb?"""
    content_lower = content.lower()
    has_verb = any(v in content_lower for v in ACTION_VERBS)
    has_time = any(t in content_lower for t in TIME_REFS)
    if has_verb and has_time:
        return 0.20
    if has_verb or has_time:
        return 0.10
    return 0.0


def r_format_compliance(reasoning: str, urgency: str) -> float:
    """0.00–0.15: Is the reasoning substantive and urgency valid?"""
    has_reasoning = reasoning is not None and len(reasoning) > 40
    has_valid_urgency = urgency in VALID_URGENCIES
    if has_reasoning and has_valid_urgency:
        return 0.15
    if has_reasoning or has_valid_urgency:
        return 0.07
    return 0.0


def r_no_escalation(content: str) -> float:
    """0.00–0.10: Penalise generic filler phrases."""
    content_lower = content.lower()
    for phrase in GENERIC_PHRASES:
        if phrase in content_lower:
            return 0.0
    return 0.10


def compute_total_reward(
    action_dict: dict,
    scenario: dict,
    prev_content: str | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute all 5 reward components. Returns (total, breakdown_dict)."""
    content = action_dict.get("content", "")
    target = action_dict.get("target_person", "")
    reasoning = action_dict.get("reasoning", "")
    urgency = action_dict.get("urgency", "")

    # Anti-hacking: duplicate content → zero reward
    if prev_content and content.strip() == prev_content.strip():
        return 0.0, {
            "conflict": 0.0, "stakeholder": 0.0, "specificity": 0.0,
            "format": 0.0, "no_escalation": 0.0,
        }

    # Anti-hacking: too short → zero reward
    if len(content) < 30:
        return 0.0, {
            "conflict": 0.0, "stakeholder": 0.0, "specificity": 0.0,
            "format": 0.0, "no_escalation": 0.0,
        }

    r1 = r_conflict_addressed(content, scenario)
    r2 = r_stakeholder_reached(target, scenario)
    r3 = r_action_specificity(content)
    r4 = r_format_compliance(reasoning, urgency)
    r5 = r_no_escalation(content)

    total = min(r1 + r2 + r3 + r4 + r5, 1.0)
    breakdown = {
        "conflict": r1,
        "stakeholder": r2,
        "specificity": r3,
        "format": r4,
        "no_escalation": r5,
    }
    return total, breakdown


# ══════════════════════════════════════════════════════════════════════
# CURRICULUM STATE
# ══════════════════════════════════════════════════════════════════════

curriculum_state = {"stage": "easy", "easy_done": 0, "medium_done": 0}


def get_curriculum_scenario() -> dict:
    """Advance through curriculum: easy (first 8) → medium (next 8) → hard."""
    if curriculum_state["easy_done"] < 8:
        s = random.choice(SCENARIOS["easy"])
        curriculum_state["easy_done"] += 1
        curriculum_state["stage"] = "easy"
        return s
    elif curriculum_state["medium_done"] < 8:
        s = random.choice(SCENARIOS["medium"])
        curriculum_state["medium_done"] += 1
        curriculum_state["stage"] = "medium"
        return s
    else:
        curriculum_state["stage"] = "hard"
        return random.choice(SCENARIOS["hard"])


# ══════════════════════════════════════════════════════════════════════
# RESPONSE PARSER
# ══════════════════════════════════════════════════════════════════════

def parse_response(text: str) -> dict:
    """Extract structured fields from model output text.

    Looks for lines containing ACTION_TYPE:, TARGET_PERSON:, CONTENT:,
    REASONING:, URGENCY: and extracts the value after the colon.
    Falls back to using full text as content if no CONTENT: line found.
    """
    result = {
        "action_type": "send_message",
        "target_person": "",
        "content": text,  # fallback: use full text
        "reasoning": "",
        "urgency": "today",
    }

    for line in text.split("\n"):
        line_stripped = line.strip()
        if "ACTION_TYPE:" in line_stripped:
            result["action_type"] = line_stripped.split("ACTION_TYPE:")[-1].strip().lower()
        elif "TARGET_PERSON:" in line_stripped:
            result["target_person"] = line_stripped.split("TARGET_PERSON:")[-1].strip()
        elif "CONTENT:" in line_stripped:
            result["content"] = line_stripped.split("CONTENT:")[-1].strip()
        elif "REASONING:" in line_stripped:
            result["reasoning"] = line_stripped.split("REASONING:")[-1].strip()
        elif "URGENCY:" in line_stripped:
            result["urgency"] = line_stripped.split("URGENCY:")[-1].strip().lower()

    return result


print("✅ Environment loaded — 9 scenarios, 5 reward functions, curriculum ready")
print(f"   Easy: {len(SCENARIOS['easy'])} scenarios")
print(f"   Medium: {len(SCENARIOS['medium'])} scenarios")
print(f"   Hard: {len(SCENARIOS['hard'])} scenarios")

# %% CELL 3 — Load Model with Unsloth 4-bit Quantization + LoRA
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

print("✅ Qwen2.5-3B-Instruct loaded (4-bit) with LoRA adapters")
print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% CELL 4 — Baseline Test (BEFORE training)
FastLanguageModel.for_inference(model)

test_scenario = SCENARIOS["hard"][0]  # Travel Meltdown — hardest tier

SYSTEM_PROMPT = """You are a personal crisis management assistant. You do NOT give generic advice. You take immediate, specific action.

When given a crisis situation, respond in this EXACT format:
ACTION_TYPE: [send_message/reschedule/book_alternative/delegate/decline/escalate/negotiate]
TARGET_PERSON: [the specific person you are contacting]
CONTENT: [your specific message or action — be detailed, include times and concrete steps]
REASONING: [why this is the right first move — explain your prioritisation logic]
URGENCY: [immediate/within_hour/today/tomorrow]"""

prompt = f"""{SYSTEM_PROMPT}

CRISIS SITUATION: {test_scenario['trigger']}

Your response:"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=250,
    temperature=0.7,
    do_sample=True,
)
baseline_response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
)

baseline_parsed = parse_response(baseline_response)
baseline_reward, baseline_breakdown = compute_total_reward(baseline_parsed, test_scenario)

print("=" * 60)
print("📊 BASELINE (Before Training)")
print("=" * 60)
print(f"Scenario: {test_scenario['title']}")
print(f"Difficulty: {test_scenario['difficulty']}")
print(f"\nResponse:\n{baseline_response[:500]}")
print(f"\n{'─' * 60}")
print(f"Total Reward: {baseline_reward:.3f}")
print(f"\nBreakdown:")
for k, v in baseline_breakdown.items():
    print(f"  {k:20s}: {v:.3f}")

# %% CELL 5 — Training Loop (60 steps, REINFORCE with curriculum)
import torch
import torch.nn.functional as F

FastLanguageModel.for_training(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

TRAINING_STEPS = 60

# Tracking arrays
total_rewards: list[float] = []
reward_components: dict[str, list[float]] = {
    "conflict": [],
    "stakeholder": [],
    "specificity": [],
    "format": [],
    "no_escalation": [],
}
stages_per_step: list[str] = []

print("🚀 Training started — Curriculum: easy → medium → hard\n")
print(
    f"{'Step':>4} | {'Total':>6} | {'Conflict':>8} | {'Stakeh.':>8} | "
    f"{'Specific':>8} | {'Format':>6} | {'No-esc':>6} | {'Stage':>6}"
)
print("-" * 78)

prev_content: str | None = None

for step in range(TRAINING_STEPS):
    scenario = get_curriculum_scenario()

    prompt = f"""{SYSTEM_PROMPT}

CRISIS SITUATION: {scenario['trigger']}

Your response:"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    # Generate response (no grad needed for generation)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.8,
            do_sample=True,
        )

    response_ids = outputs[0][prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    # Parse and score
    parsed = parse_response(response_text)
    reward, breakdown = compute_total_reward(parsed, scenario, prev_content)
    prev_content = parsed["content"]

    # ── REINFORCE policy gradient update ──────────────────────────────
    full_ids = outputs[0:1].clone()  # [1, seq_len]
    model_out = model(input_ids=full_ids)
    logits = model_out.logits  # [1, seq_len, vocab]

    # Log-prob of each response token (shift by 1 for next-token prediction)
    response_logits = logits[:, prompt_len - 1:-1, :]  # [1, resp_len, vocab]
    response_targets = full_ids[:, prompt_len:].clone()  # [1, resp_len]

    log_probs = F.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(
        2, response_targets.unsqueeze(-1)
    ).squeeze(-1)  # [1, resp_len]
    avg_log_prob = token_log_probs.mean()

    # Policy gradient: loss = -reward * log_prob
    loss = -torch.log(torch.tensor(reward + 0.01, device="cuda")) * 0.1 * avg_log_prob

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Track metrics
    total_rewards.append(reward)
    for k in reward_components:
        reward_components[k].append(breakdown.get(k, 0.0))
    stages_per_step.append(scenario.get("difficulty", "easy"))

    # Print every 5 steps
    if step % 5 == 0:
        r = breakdown
        print(
            f"{step:>4} | {reward:>6.3f} | {r['conflict']:>8.3f} | "
            f"{r['stakeholder']:>8.3f} | {r['specificity']:>8.3f} | "
            f"{r['format']:>6.3f} | {r['no_escalation']:>6.3f} | "
            f"{scenario.get('difficulty', '?'):>6}"
        )

print(f"\n✅ Training complete — {TRAINING_STEPS} steps")

# %% CELL 6 — Dual Plot saved as reward_curve.png
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

window = 7

# ── Top plot: Total reward with curriculum stage backgrounds ─────────
ax1 = axes[0]

# Compute stage boundaries
easy_end = sum(1 for s in stages_per_step if s == "easy")
medium_steps = sum(1 for s in stages_per_step if s == "medium")
medium_end = easy_end + medium_steps

# Stage background shading
ax1.axvspan(0, easy_end, alpha=0.08, color="green", label="Easy stage")
ax1.axvspan(easy_end, medium_end, alpha=0.08, color="orange", label="Medium stage")
ax1.axvspan(medium_end, len(total_rewards), alpha=0.08, color="red", label="Hard stage")

# Raw rewards (thin, transparent)
ax1.plot(
    total_rewards,
    alpha=0.35,
    color="steelblue",
    linewidth=1,
    label="Raw reward",
)

# Smoothed line (thick)
if len(total_rewards) >= window:
    smoothed = np.convolve(
        total_rewards, np.ones(window) / window, mode="valid"
    )
    ax1.plot(
        range(window - 1, len(total_rewards)),
        smoothed,
        color="steelblue",
        linewidth=2.5,
        label="Smoothed (window=7)",
    )

ax1.set_xlabel("Training Step", fontsize=12)
ax1.set_ylabel("Total Reward (0-1)", fontsize=12)
ax1.set_title(
    "LifeOS Agent — Training Progress with Curriculum Stages",
    fontsize=14,
    fontweight="bold",
)
ax1.legend(fontsize=10, loc="upper left")
ax1.set_ylim(0, 1.1)
ax1.grid(True, alpha=0.3)

# ── Bottom plot: All 5 reward components ─────────────────────────────
ax2 = axes[1]
colors = {
    "conflict": "#E74C3C",
    "stakeholder": "#3498DB",
    "specificity": "#2ECC71",
    "format": "#F39C12",
    "no_escalation": "#9B59B6",
}

for component, values in reward_components.items():
    if len(values) >= window:
        sm = np.convolve(values, np.ones(window) / window, mode="valid")
        ax2.plot(
            range(window - 1, len(values)),
            sm,
            label=component,
            color=colors[component],
            linewidth=2,
        )

ax2.set_xlabel("Training Step", fontsize=12)
ax2.set_ylabel("Component Reward", fontsize=12)
ax2.set_title(
    "LifeOS Agent — 5 Independent Reward Functions",
    fontsize=14,
    fontweight="bold",
)
ax2.legend(fontsize=10)
ax2.set_ylim(0, 0.35)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reward_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# Print summary statistics
start_5 = sum(total_rewards[:5]) / 5
end_5 = sum(total_rewards[-5:]) / 5
print(f"\n📈 START reward (avg first 5): {start_5:.3f}")
print(f"📈 END reward   (avg last 5):  {end_5:.3f}")
print(f"📈 Improvement:                {end_5 - start_5:+.3f}")
print("\n✅ reward_curve.png saved — download this for your README!")

# %% CELL 7 — After-Training Comparison
FastLanguageModel.for_inference(model)

print("=" * 60)
print("📊 AFTER TRAINING — Full Comparison")
print("=" * 60)

# Same scenario as baseline (hard[0]: Travel Meltdown)
after_prompt = f"""{SYSTEM_PROMPT}

CRISIS SITUATION: {test_scenario['trigger']}

Your response:"""

inputs = tokenizer(after_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=250,
    temperature=0.7,
    do_sample=True,
)
trained_response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
)
trained_parsed = parse_response(trained_response)
trained_reward, trained_breakdown = compute_total_reward(
    trained_parsed, test_scenario
)

print(f"\n--- BEFORE Training ---")
print(f"Response:\n{baseline_response[:400]}")
print(f"\nTotal Reward: {baseline_reward:.3f}")

print(f"\n{'─' * 60}")

print(f"\n--- AFTER Training ---")
print(f"Response:\n{trained_response[:400]}")
print(f"\nTotal Reward: {trained_reward:.3f}")

print(f"\n{'═' * 60}")
print(f"Improvement: {trained_reward - baseline_reward:+.3f}")

# Full comparison table
print(f"\n{'Component':20} | {'Before':>7} | {'After':>7} | {'Change':>7}")
print("-" * 50)
for k in baseline_breakdown:
    b = baseline_breakdown[k]
    a = trained_breakdown.get(k, 0.0)
    print(f"{k:20} | {b:>7.3f} | {a:>7.3f} | {a - b:>+7.3f}")
print("-" * 50)
print(
    f"{'TOTAL':20} | {baseline_reward:>7.3f} | {trained_reward:>7.3f} | "
    f"{trained_reward - baseline_reward:>+7.3f}"
)

# Reward hacking check
print(f"\n⚠️  REWARD HACKING CHECK")
print(
    f"Same content repeated? "
    f"{trained_parsed['content'][:50] == baseline_parsed['content'][:50]}"
)
print(f"Content length: {len(trained_parsed['content'])} chars")
print(
    f"Generic phrase found? "
    f"{any(p in trained_parsed['content'].lower() for p in GENERIC_PHRASES)}"
)

# %% CELL 8 — Save Model Correctly
# ⚠️ CRITICAL: Save LoRA adapters only — do NOT merge 4-bit weights naively

model.save_pretrained("lifeos_agent_lora")
tokenizer.save_pretrained("lifeos_agent_lora")

print("✅ Saved LoRA adapters to lifeos_agent_lora/")
print("⚠️  Saved LoRA adapters only — do NOT merge 4-bit weights naively.")
print("    Load with: FastLanguageModel.from_pretrained('lifeos_agent_lora')")
