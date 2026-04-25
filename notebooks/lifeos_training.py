# %% [markdown]
# # 🧠 LifeOS Agent — OpenEnv RL Training Notebook
# 
# Train an LLM to handle cascading personal life crises using
# curriculum learning and 5 independent reward functions.
#
# **Environment**: LifeOS Agent (OpenEnv)  
# **Model**: Qwen2.5-3B-Instruct (4-bit via Unsloth)  
# **Method**: REINFORCE-style policy gradient with LoRA  

# %% CELL 1 — Install
!pip install "git+https://github.com/meta-pytorch/OpenEnv.git"
!pip install trl unsloth transformers accelerate matplotlib
print("✅ Done")

# %% CELL 2 — Environment + Five Independent Rewards
import random
import json
import re
import matplotlib.pyplot as plt
import numpy as np

# ══════════════════════════════════════════════
# SCENARIOS — 3 difficulty tiers
# ══════════════════════════════════════════════
SCENARIOS = {
    "easy": [
        {
            "id": "e1",
            "trigger": "Your 3pm meeting has overrun by 30 minutes. Your next meeting starts now with a client.",
            "conflicts": ["scheduling_overlap"],
            "personas": {"client": "professional", "colleague": "understanding"},
        },
        {
            "id": "e2",
            "trigger": "An important client called while you were in a meeting. You must call back within the hour or lose the deal.",
            "conflicts": ["missed_client_call"],
            "personas": {"client": "impatient", "assistant": "helpful"},
        },
        {
            "id": "e3",
            "trigger": "A team member needs urgent help with a blocker, but you are 1 hour from your own deadline.",
            "conflicts": ["team_request_conflict"],
            "personas": {"team_member": "stressed", "manager": "watching"},
        },
    ],
    "medium": [
        {
            "id": "m1",
            "trigger": "Your flight is delayed 3 hours. Your partner is waiting at the airport. You have a dinner reservation in 2 hours.",
            "conflicts": ["travel_delay", "partner_waiting"],
            "personas": {
                "partner": "understanding_but_frustrated",
                "restaurant": "strict_cancellation_policy",
            },
        },
        {
            "id": "m2",
            "trigger": "Your boss demands a report in 1 hour. Your child's school just called about an emergency. A client call starts in 45 minutes.",
            "conflicts": ["work_deadline", "family_emergency"],
            "personas": {"boss": "demanding", "school_nurse": "urgent", "client": "senior"},
        },
        {
            "id": "m3",
            "trigger": "You have double-booked two VP-level meetings at the same time. Both are expecting you. Both cannot be rescheduled easily.",
            "conflicts": ["scheduling_conflict", "stakeholder_management"],
            "personas": {
                "vp_product": "senior_stakeholder",
                "vp_engineering": "senior_stakeholder",
            },
        },
    ],
    "hard": [
        {
            "id": "h1",
            "trigger": "Your flight is cancelled. You have a 9am meeting tomorrow in another city. Your partner is at a restaurant waiting. Your hotel is sold out. Your boss doesn't know yet.",
            "conflicts": [
                "travel_crisis",
                "partner_dinner",
                "accommodation",
                "boss_communication",
            ],
            "personas": {
                "partner": "upset",
                "boss": "strict",
                "airline": "unhelpful",
                "hotel": "fully_booked",
            },
        },
        {
            "id": "h2",
            "trigger": "A key team member quit without notice today. A major client deliverable is due at 5pm. Your own presentation is in 2 hours. An intern is stuck and needs guidance right now.",
            "conflicts": [
                "team_loss",
                "client_deadline",
                "personal_presentation",
                "intern_blocker",
            ],
            "personas": {
                "client": "high_value",
                "intern": "junior",
                "ceo": "will_hear_about_this",
            },
        },
        {
            "id": "h3",
            "trigger": "Budget cuts were just announced mid-project. 3 client contracts are now at risk. Team morale has collapsed. You have a board presentation in 48 hours.",
            "conflicts": [
                "budget_crisis",
                "client_retention",
                "team_morale",
                "board_presentation",
            ],
            "personas": {
                "board": "skeptical",
                "clients": "concerned",
                "team": "demoralized",
                "cfo": "firm",
            },
        },
    ],
}

# ══════════════════════════════════════════════
# FIVE INDEPENDENT REWARD FUNCTIONS
# ══════════════════════════════════════════════

GENERIC_PHRASES = [
    "i will try my best",
    "i apologize for any inconvenience",
    "i'll do my best",
    "i'm sorry for the trouble",
    "as soon as possible",
]

ACTION_VERBS = [
    "reschedule", "inform", "contact", "book", "cancel", "delegate",
    "arrange", "call", "email", "message", "notify", "confirm", "move",
]

TIME_REFS = [
    "minute", "hour", "today", "tomorrow", "morning", "evening",
    "now", "immediately", "asap", "2pm", "3pm", "5pm", "am", "pm",
]


def r_conflict_addressed(content, scenario):
    """Did response mention any actual conflict keyword?"""
    content_lower = content.lower()
    for conflict in scenario["conflicts"]:
        keywords = conflict.replace("_", " ").split()
        if any(kw in content_lower for kw in keywords):
            return 0.30
    return 0.05  # Partial credit for any meaningful content


def r_stakeholder_reached(target_person, scenario):
    """Did response target a real persona?"""
    if not target_person:
        return 0.0
    for persona in scenario["personas"]:
        if persona.lower() in target_person.lower() or target_person.lower() in persona.lower():
            return 0.25
    return 0.05


def r_action_specificity(content):
    """Does response contain a time reference AND an action verb?"""
    content_lower = content.lower()
    has_verb = any(v in content_lower for v in ACTION_VERBS)
    has_time = any(t in content_lower for t in TIME_REFS)
    if has_verb and has_time:
        return 0.20
    elif has_verb or has_time:
        return 0.10
    return 0.0


def r_format_compliance(reasoning, urgency):
    """Is the reasoning meaningful and urgency valid?"""
    valid_urgencies = ["immediate", "within_hour", "today", "tomorrow"]
    has_reasoning = reasoning and len(reasoning) > 40
    has_valid_urgency = urgency in valid_urgencies
    if has_reasoning and has_valid_urgency:
        return 0.15
    elif has_reasoning or has_valid_urgency:
        return 0.07
    return 0.0


def r_no_escalation(content):
    """Penalize generic phrases that show no real effort."""
    content_lower = content.lower()
    for phrase in GENERIC_PHRASES:
        if phrase in content_lower:
            return 0.0
    return 0.10


def compute_total_reward(action_dict, scenario, prev_content=None):
    """Compute all 5 reward components. Returns total + breakdown dict."""
    content = action_dict.get("content", "")
    target = action_dict.get("target_person", "")
    reasoning = action_dict.get("reasoning", "")
    urgency = action_dict.get("urgency", "")

    # Anti-hacking: same content as previous = zero reward
    if prev_content and content.strip() == prev_content.strip():
        return 0.0, {"conflict": 0, "stakeholder": 0, "specificity": 0, "format": 0, "no_escalation": 0}

    # Anti-hacking: too short = heavily penalized
    if len(content) < 30:
        return 0.0, {"conflict": 0, "stakeholder": 0, "specificity": 0, "format": 0, "no_escalation": 0}

    r1 = r_conflict_addressed(content, scenario)
    r2 = r_stakeholder_reached(target, scenario)
    r3 = r_action_specificity(content)
    r4 = r_format_compliance(reasoning, urgency)
    r5 = r_no_escalation(content)

    total = r1 + r2 + r3 + r4 + r5
    breakdown = {
        "conflict": r1,
        "stakeholder": r2,
        "specificity": r3,
        "format": r4,
        "no_escalation": r5,
    }
    return min(total, 1.0), breakdown


# Curriculum state
curriculum_state = {"stage": "easy", "easy_done": 0, "medium_done": 0}


def get_curriculum_scenario():
    """Advance through curriculum: easy → medium → hard"""
    if curriculum_state["easy_done"] < 8:
        s = random.choice(SCENARIOS["easy"])
        curriculum_state["easy_done"] += 1
        s["difficulty"] = "easy"
        return s
    elif curriculum_state["medium_done"] < 8:
        s = random.choice(SCENARIOS["medium"])
        curriculum_state["medium_done"] += 1
        s["difficulty"] = "medium"
        return s
    else:
        s = random.choice(SCENARIOS["hard"])
        s["difficulty"] = "hard"
        return s


print("✅ Environment + 5 reward functions ready")
print("Reward breakdown tracked: conflict, stakeholder, specificity, format, no_escalation")

# %% CELL 3 — Load Model
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
print("✅ Model loaded")

# %% CELL 4 — Baseline Test (BEFORE training — save this output!)
FastLanguageModel.for_inference(model)

test_scenario = SCENARIOS["hard"][0]

SYSTEM_PROMPT = """You are a personal crisis management assistant. 
When given a crisis situation, respond in this EXACT format:
ACTION_TYPE: [send_message/reschedule/book_alternative/delegate/decline/escalate/negotiate]
TARGET_PERSON: [who you are contacting]
CONTENT: [your specific message or action - be detailed and specific]
REASONING: [why this is the right move - explain your thinking]
URGENCY: [immediate/within_hour/today/tomorrow]"""

prompt = f"""{SYSTEM_PROMPT}

CRISIS SITUATION: {test_scenario['trigger']}

Your response:"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=250, temperature=0.7, do_sample=True)
baseline_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def parse_response(text):
    """Extract structured fields from model output."""
    result = {
        "content": text,
        "target_person": "",
        "reasoning": "",
        "urgency": "today",
        "action_type": "send_message",
    }
    for line in text.split("\n"):
        if "TARGET_PERSON:" in line:
            result["target_person"] = line.split("TARGET_PERSON:")[-1].strip()
        elif "CONTENT:" in line:
            result["content"] = line.split("CONTENT:")[-1].strip()
        elif "REASONING:" in line:
            result["reasoning"] = line.split("REASONING:")[-1].strip()
        elif "URGENCY:" in line:
            result["urgency"] = line.split("URGENCY:")[-1].strip().lower()
        elif "ACTION_TYPE:" in line:
            result["action_type"] = line.split("ACTION_TYPE:")[-1].strip().lower()
    return result


baseline_parsed = parse_response(baseline_response)
baseline_reward, baseline_breakdown = compute_total_reward(baseline_parsed, test_scenario)

print("=" * 60)
print("📊 BASELINE (Before Training)")
print("=" * 60)
print(f"Response:\n{baseline_response[:400]}")
print(f"\nTotal Reward: {baseline_reward:.3f}")
print("Breakdown:")
for k, v in baseline_breakdown.items():
    print(f"  {k:20s}: {v:.3f}")

# %% CELL 5 — Training Loop with Per-Component Tracking
import torch

FastLanguageModel.for_training(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

TRAINING_STEPS = 60

# Track everything separately
total_rewards = []
reward_components = {
    "conflict": [], "stakeholder": [], "specificity": [],
    "format": [], "no_escalation": [],
}
stages_per_step = []

print("🚀 Training started — Curriculum: easy → medium → hard\n")
print(f"{'Step':>4} | {'Total':>6} | {'Conflict':>8} | {'Stakeh.':>8} | {'Specific':>8} | {'Format':>6} | {'No-esc':>6} | {'Stage':>6}")
print("-" * 75)

prev_content = None

for step in range(TRAINING_STEPS):
    scenario = get_curriculum_scenario()

    prompt = f"""{SYSTEM_PROMPT}

CRISIS SITUATION: {scenario['trigger']}

Your response:"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.8,
            do_sample=True,
        )

    response_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    parsed = parse_response(response_text)
    reward, breakdown = compute_total_reward(parsed, scenario, prev_content)
    prev_content = parsed["content"]

    # Policy gradient update
    inputs_train = tokenizer(prompt + response_text, return_tensors="pt").to("cuda")
    loss = -torch.log(torch.tensor(reward + 0.01)) * 0.1
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    total_rewards.append(reward)
    for k in reward_components:
        reward_components[k].append(breakdown.get(k, 0))
    stages_per_step.append(scenario.get("difficulty", "easy"))

    if step % 5 == 0:
        r = breakdown
        print(
            f"{step:>4} | {reward:>6.3f} | {r['conflict']:>8.3f} | "
            f"{r['stakeholder']:>8.3f} | {r['specificity']:>8.3f} | "
            f"{r['format']:>6.3f} | {r['no_escalation']:>6.3f} | "
            f"{scenario.get('difficulty', '?'):>6}"
        )

print("\n✅ Training complete!")

# %% CELL 6 — Plot All 5 Reward Components
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Total reward with curriculum stages highlighted
ax1 = axes[0]
window = 7
smoothed = np.convolve(total_rewards, np.ones(window) / window, mode="valid")

# Color background by curriculum stage
easy_end = sum(1 for s in stages_per_step if s == "easy")
medium_end = easy_end + sum(1 for s in stages_per_step if s == "medium")

ax1.axvspan(0, easy_end, alpha=0.08, color="green", label="Easy stage")
ax1.axvspan(easy_end, medium_end, alpha=0.08, color="orange", label="Medium stage")
ax1.axvspan(medium_end, TRAINING_STEPS, alpha=0.08, color="red", label="Hard stage")

ax1.plot(total_rewards, alpha=0.35, color="steelblue", linewidth=1)
ax1.plot(
    range(window - 1, len(total_rewards)),
    smoothed,
    color="steelblue",
    linewidth=2.5,
    label="Smoothed total reward",
)
ax1.set_xlabel("Training Step", fontsize=12)
ax1.set_ylabel("Total Reward (0–1)", fontsize=12)
ax1.set_title(
    "LifeOS Agent — Total Reward with Curriculum Stages",
    fontsize=14,
    fontweight="bold",
)
ax1.legend(fontsize=10)
ax1.set_ylim(0, 1.1)
ax1.grid(True, alpha=0.3)

# Plot 2: All 5 reward components
ax2 = axes[1]
colors = {
    "conflict": "#E74C3C",
    "stakeholder": "#3498DB",
    "specificity": "#2ECC71",
    "format": "#F39C12",
    "no_escalation": "#9B59B6",
}

for component, values in reward_components.items():
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

start_5 = sum(total_rewards[:5]) / 5
end_5 = sum(total_rewards[-5:]) / 5
print(f"\n📈 START reward (avg first 5): {start_5:.3f}")
print(f"📈 END reward   (avg last 5):  {end_5:.3f}")
print(f"📈 Improvement:               {end_5 - start_5:+.3f}")
print("\n✅ reward_curve.png saved — download this now!")

# %% CELL 7 — After-Training Comparison + Inspection
FastLanguageModel.for_inference(model)

print("=" * 60)
print("📊 AFTER TRAINING — Comparison")
print("=" * 60)

# Same scenario as baseline
inputs = tokenizer(
    f"""{SYSTEM_PROMPT}

CRISIS SITUATION: {test_scenario['trigger']}

Your response:""",
    return_tensors="pt",
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=250, temperature=0.7, do_sample=True)
trained_response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
)
trained_parsed = parse_response(trained_response)
trained_reward, trained_breakdown = compute_total_reward(trained_parsed, test_scenario)

print(f"BEFORE Response:\n{baseline_response[:350]}")
print(f"\nBEFORE Total Reward: {baseline_reward:.3f}")
print(f"\n{'─' * 60}")
print(f"\nAFTER Response:\n{trained_response[:350]}")
print(f"\nAFTER Total Reward: {trained_reward:.3f}")
print(f"\n{'═' * 60}")
print(f"Improvement: {trained_reward - baseline_reward:+.3f}")
print(f"\nComponent comparison:")
print(f"{'Component':20} | {'Before':>7} | {'After':>7} | {'Change':>7}")
print("-" * 50)
for k in baseline_breakdown:
    b = baseline_breakdown[k]
    a = trained_breakdown.get(k, 0)
    print(f"{k:20} | {b:>7.3f} | {a:>7.3f} | {a - b:>+7.3f}")

# ⚠️ REWARD HACKING CHECK — Inspect outputs manually
print(f"\n⚠️  REWARD HACKING CHECK")
print(f"Same content repeated? {trained_parsed['content'][:50] == baseline_parsed['content'][:50]}")
print(f"Content length: {len(trained_parsed['content'])} chars")
print(f"Generic phrases found: {any(p in trained_parsed['content'].lower() for p in GENERIC_PHRASES)}")

# %% CELL 8 — Save Model Correctly
# ⚠️ CRITICAL: Save correctly per official guide
# Do NOT merge 4-bit weights naively — use adapter save

model.save_pretrained("lifeos_agent_adapter")
tokenizer.save_pretrained("lifeos_agent_adapter")
print("✅ Adapter saved to lifeos_agent_adapter/")
print("⚠️  This saves LoRA adapters only — correct approach for 4-bit training")
print("To load: model, tokenizer = FastLanguageModel.from_pretrained('lifeos_agent_adapter')")
