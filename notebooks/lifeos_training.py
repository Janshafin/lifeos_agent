# ════════════════════════════════════════════════════════════
# LifeOS Agent — Post-Training with GRPO-Style RL
# OpenEnv Hackathon 2026
# ════════════════════════════════════════════════════════════
#
# PRE-TRAINING vs POST-TRAINING:
# Pre-training: Done by Qwen team on Qwen2.5-3B-Instruct
# Post-training (THIS NOTEBOOK): We use GRPO-style reinforcement
# learning to improve the model's crisis resolution ability
# using our LifeOS environment's 5 reward functions.
#
# Stack: HuggingFace transformers + PEFT + bitsandbytes (NO Unsloth)
# Model: Qwen/Qwen2.5-3B-Instruct
# Hardware: Google Colab T4 GPU
# ════════════════════════════════════════════════════════════

# %% CELL 1 — Install Dependencies
# !pip install -q trl transformers peft bitsandbytes accelerate
# !pip install -q matplotlib numpy datasets torch

import os
import random
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "sans-serif"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print("✅ All dependencies ready")

# %% CELL 2 — Standalone Environment (all 9 scenarios, 5 rewards, curriculum)

SCENARIOS = {
    "easy": [
        {"id":"easy_01","title":"Meeting Overrun","difficulty":"easy",
         "trigger":"Your current meeting has overrun by 30 minutes. Your next meeting starts right now with an important client who is already waiting in the conference room.",
         "conflicts":["scheduling_overlap"],
         "personas":{"Alice_Client":"punctual, values professionalism","Bob_Colleague":"long-winded, unaware of your schedule"},
         "success_criteria":["Inform Alice about delay with specific timeframe","Exit overrun meeting gracefully"]},
        {"id":"easy_02","title":"Missed Client Call","difficulty":"easy",
         "trigger":"An important client called while you were in a meeting. They need to discuss a contract change urgently. You must call back within the hour or risk losing the deal.",
         "conflicts":["missed_client_call"],
         "personas":{"Client_Director":"impatient, high-value account","PM_Rachel":"your project manager, needs updates"},
         "success_criteria":["Call client back with specific plan","Loop in PM on contract changes"]},
        {"id":"easy_03","title":"Team Blocker","difficulty":"easy",
         "trigger":"A team member needs urgent help with a critical blocker, but you are 1 hour from your own deadline. They cannot proceed without your input.",
         "conflicts":["team_request_conflict"],
         "personas":{"Junior_Dev":"stressed, blocked for 3 hours","PM_Rachel":"tracking both deliverables"},
         "success_criteria":["Unblock team member with actionable guidance","Protect your own deadline"]},
    ],
    "medium": [
        {"id":"medium_01","title":"Travel Delay Cascade","difficulty":"medium",
         "trigger":"Your flight has been delayed by 3 hours. Your partner is at the airport waiting. You have a dinner reservation in 2 hours that took 3 months to book.",
         "conflicts":["flight_delay","dinner_reservation_at_risk"],
         "personas":{"Partner_Jamie":"excited about dinner, drove 45 min to airport","Restaurant_Host":"strict policy, 3-month waitlist"},
         "success_criteria":["Inform partner with empathy and backup plan","Contact restaurant to save reservation"]},
        {"id":"medium_02","title":"Work-Family Collision","difficulty":"medium",
         "trigger":"Your boss needs a critical report in 1 hour. Your child's school called — your kid fell and needs pickup immediately. Client call starts in 45 minutes.",
         "conflicts":["boss_report_deadline","family_emergency"],
         "personas":{"Boss_Karen":"demanding, no tolerance for missed deadlines","School_Nurse":"needs guardian within 30 minutes","Client_VP":"contract renewal depends on this call"},
         "success_criteria":["Address family emergency as top priority","Delegate work commitments"]},
        {"id":"medium_03","title":"Double-Booked VPs","difficulty":"medium",
         "trigger":"You are double-booked for two VP-level meetings starting right now. VP of Sales expects Q3 numbers. VP of Engineering expects a feature demo. Both will take it personally if you skip.",
         "conflicts":["vp_sales_meeting","vp_engineering_meeting"],
         "personas":{"VP_Sales":"competitive, holds grudges","VP_Engineering":"technical, booked 2 weeks ago","Your_Manager":"caught in the middle"},
         "success_criteria":["Attend or delegate one meeting credibly","Handle higher-stakes meeting personally"]},
    ],
    "hard": [
        {"id":"hard_01","title":"Total Travel Meltdown","difficulty":"hard",
         "trigger":"Your flight has been cancelled entirely. You have a 9am board meeting tomorrow in another city. Your partner is at a restaurant waiting — you are 40 minutes late. Every hotel is sold out. Your boss does not know.",
         "conflicts":["flight_cancelled","partner_waiting","hotel_unavailable","boss_uninformed"],
         "personas":{"Partner_Jamie":"furious, texting for 40 minutes","Boss_Karen":"expects you in person tomorrow","Airline_Agent":"overwhelmed","Hotel_Concierge":"fully booked"},
         "success_criteria":["Message partner immediately","Inform boss with backup plan","Find alternative transport","Secure accommodation"]},
        {"id":"hard_02","title":"Team Collapse","difficulty":"hard",
         "trigger":"Your key team member quit this morning. Client deliverable due at 5pm today. Board presentation in 2 hours. The intern is stuck and panicking.",
         "conflicts":["team_member_quit","client_deliverable","presentation_prep"],
         "personas":{"Client_Director":"expecting delivery at 5pm","Intern_Alex":"panicking","CTO":"wants retention plan","HR_Lead":"needs exit paperwork"},
         "success_criteria":["Own the client deliverable","Guide the intern","Brief CTO on continuity"]},
        {"id":"hard_03","title":"Budget Crisis Firestorm","difficulty":"hard",
         "trigger":"30% budget cuts announced mid-project. Three client contracts at risk. Team morale collapsed. Board presentation in 48 hours. Press found out.",
         "conflicts":["budget_cuts","client_contracts_at_risk","team_morale_collapsed","press_inquiry"],
         "personas":{"CFO":"open to revised plans with ROI","Client_A_Lead":"biggest account, will leave","Senior_Engineer_1":"has competitor offer","Journalist":"deadline in 24h","Board_Chair":"needs confidence"},
         "success_criteria":["Negotiate with CFO using data","Contact at-risk clients","Retain key engineers","Manage press"]},
    ],
}

ACTION_VERBS = ["reschedule","inform","contact","book","cancel","delegate","arrange","call","email","message","notify","confirm","move","propose","apologize","explain","update","brief","coordinate","negotiate","offer","send","draft","prepare","escalate","rebook","transfer","assign","prioritize","defer"]
TIME_REFS = ["minute","hour","today","tomorrow","morning","afternoon","evening","tonight","now","immediately","asap","urgent","9am","5pm","am","pm","within","deadline","by","noon","midnight","eod","eob","before","after"]
GENERIC_PHRASES = ["i will try my best","i apologize for any inconvenience","i ll do my best","i m sorry for the trouble","as soon as possible","i will get back to you","i understand your concern","i will look into this"]
VALID_URGENCY = ["immediate","within_hour","today","tomorrow"]

def reward_conflict(content, scenario):
    cl = content.lower()
    for c in scenario["conflicts"]:
        for kw in c.replace("_"," ").split():
            if len(kw) > 2 and kw in cl:
                return 0.30
    return 0.05 if len(content.strip()) > 30 else 0.0

def reward_stakeholder(target, scenario):
    t = target.lower().strip()
    if not t: return 0.0
    for p in scenario["personas"]:
        if p.lower() in t or t in p.lower():
            return 0.25
    return 0.05 if t else 0.0

def reward_specificity(content):
    cl = content.lower()
    hv = any(v in cl for v in ACTION_VERBS)
    ht = any(t in cl for t in TIME_REFS)
    if hv and ht: return 0.20
    if hv or ht: return 0.10
    return 0.0

def reward_format(reasoning, urgency):
    gr_ = len(reasoning.strip()) > 40
    gu = urgency in VALID_URGENCY
    if gr_ and gu: return 0.15
    if gr_ or gu: return 0.07
    return 0.0

def reward_no_generic(content):
    cl = content.lower()
    return 0.0 if any(p in cl for p in GENERIC_PHRASES) else 0.10

def compute_full_reward(parsed, scenario, prev_content=""):
    content = parsed.get("content","")
    # Anti-hacking: duplicate
    if prev_content and content.strip() == prev_content.strip():
        return 0.0, {k: 0.0 for k in ["conflict_addressed","stakeholder_reached","action_specificity","format_compliance","no_escalation"]}
    # Anti-hacking: too short
    if len(content.strip()) < 30:
        return 0.0, {k: 0.0 for k in ["conflict_addressed","stakeholder_reached","action_specificity","format_compliance","no_escalation"]}
    bd = {
        "conflict_addressed": reward_conflict(content, scenario),
        "stakeholder_reached": reward_stakeholder(parsed.get("target_person",""), scenario),
        "action_specificity": reward_specificity(content),
        "format_compliance": reward_format(parsed.get("reasoning",""), parsed.get("urgency","")),
        "no_escalation": reward_no_generic(content),
    }
    return min(sum(bd.values()), 1.0), bd

SYSTEM_PROMPT = """You are a crisis management AI. You must resolve this situation with a specific, actionable response.

Respond with EXACTLY these 5 fields, one per line:
action_type: [send_message/reschedule/book_alternative/delegate/decline/escalate/negotiate]
target_person: [exact name of person to contact from the scenario]
content: [your actual message — be specific with times, actions, names. minimum 30 characters]
reasoning: [why this is the right move right now — must be detailed, over 40 characters]
urgency: [immediate/within_hour/today/tomorrow]"""

def parse_response(text):
    fields = {"action_type":"send_message","target_person":"","content":"","reasoning":"","urgency":"immediate"}
    current_field = None
    buffer = []
    for line in text.strip().split("\n"):
        ls = line.strip()
        ll = ls.lower()
        matched = False
        for key in fields:
            if ll.startswith(key + ":") or ll.startswith(key.replace("_"," ") + ":"):
                if current_field and buffer:
                    fields[current_field] = " ".join(buffer).strip()
                current_field = key
                val = ls[ls.index(":")+1:].strip().strip("\"'")
                buffer = [val] if val else []
                matched = True
                break
        if not matched and current_field:
            buffer.append(ls)
    if current_field and buffer:
        fields[current_field] = " ".join(buffer).strip()
    # Normalize urgency
    for u in VALID_URGENCY:
        if u in fields["urgency"].lower():
            fields["urgency"] = u
            break
    else:
        fields["urgency"] = "immediate"
    return fields

def get_curriculum_stage(step):
    if step < 8: return "easy"
    if step < 16: return "medium"
    return "hard"

def get_curriculum_scenario(step):
    stage = get_curriculum_stage(step)
    return random.choice(SCENARIOS[stage])

def build_prompt(scenario):
    personas = "; ".join(f"{n} ({d})" for n, d in scenario["personas"].items())
    conflicts = ", ".join(scenario["conflicts"])
    criteria = "; ".join(scenario["success_criteria"])
    return f"""{SYSTEM_PROMPT}

CRISIS: {scenario['trigger']}

People involved: {personas}
Active conflicts: {conflicts}
Success criteria: {criteria}"""

print("✅ Environment ready: 9 scenarios, 5 reward functions, 3 anti-hacking guards, curriculum")

# %% CELL 3 — Load Model with 8-bit Quantization + LoRA (NO Unsloth)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading {MODEL_NAME} with 8-bit quantization...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("✅ Model loaded with LoRA — ready for post-training")

def generate_response(prompt, temp=0.7, max_tokens=300):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=temp,
            do_sample=True, top_p=0.9, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# %% CELL 4 — Baseline Test BEFORE Post-Training

HARD_SCENARIO = SCENARIOS["hard"][0]  # Total Travel Meltdown

print("=" * 55)
print("PRE-TRAINING BASELINE")
print("Scenario: Total Travel Meltdown (hardest)")
print("=" * 55)

baseline_prompt = build_prompt(HARD_SCENARIO)
baseline_response = generate_response(baseline_prompt)
print(f"\nModel output:\n{baseline_response}\n")

baseline_parsed = parse_response(baseline_response)
baseline_reward, baseline_breakdown = compute_full_reward(baseline_parsed, HARD_SCENARIO)

print("BASELINE RESULTS (Before Post-Training):")
print("━" * 45)
MAX_VALS = {"conflict_addressed":0.30,"stakeholder_reached":0.25,"action_specificity":0.20,"format_compliance":0.15,"no_escalation":0.10}
for comp, score in baseline_breakdown.items():
    mx = MAX_VALS[comp]
    pct = score / mx * 100 if mx > 0 else 0
    print(f"  {comp:<22} {score:.2f} / {mx:.2f}  ({pct:3.0f}%)")
print("━" * 45)
print(f"  TOTAL BASELINE REWARD: {baseline_reward:.3f} / 1.00")
print("━" * 45)

# Store for comparison
stored_baseline_reward = baseline_reward
stored_baseline_breakdown = dict(baseline_breakdown)
stored_baseline_response = baseline_response[:500]

# %% CELL 5 — Post-Training Loop (60 steps, GRPO-style RL)

TRAINING_STEPS = 60
LR = 2e-5

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_rewards = []
comp_history = {"conflict_addressed":[],"stakeholder_reached":[],"action_specificity":[],"format_compliance":[],"no_escalation":[]}
stages = []
prev_content = ""

print("=" * 65)
print("POST-TRAINING WITH GRPO-STYLE RL")
print("Curriculum: Easy (1-8) → Medium (9-16) → Hard (17+)")
print("=" * 65)

model.train()

for step in range(TRAINING_STEPS):
    stage = get_curriculum_stage(step)
    scenario = get_curriculum_scenario(step)

    # Generate
    prompt = build_prompt(scenario)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs, max_new_tokens=300, temperature=0.7,
        do_sample=True, top_p=0.9, pad_token_id=tokenizer.pad_token_id,
    )
    response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Parse and reward
    parsed = parse_response(response_text)
    total, bd = compute_full_reward(parsed, scenario, prev_content)
    prev_content = parsed.get("content", "")

    # GRPO-style: reward-weighted supervised loss on own outputs
    if total > 0.2:
        labels = outputs[0].clone()
        labels[:inputs["input_ids"].shape[1]] = -100
        with torch.amp.autocast("cuda"):
            loss_out = model(input_ids=outputs[0].unsqueeze(0), labels=labels.unsqueeze(0))
            # Weight by reward: higher reward = stronger gradient signal
            weighted_loss = loss_out.loss * total
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Track
    total_rewards.append(total)
    for k in comp_history:
        comp_history[k].append(bd[k])
    stages.append(stage)

    if step % 5 == 0 or step == TRAINING_STEPS - 1:
        avg5 = np.mean(total_rewards[max(0,step-4):step+1])
        print(f"Step {step:3d}/{TRAINING_STEPS} | {stage:6s} | "
              f"R={total:.3f} Avg5={avg5:.3f} | "
              f"CA={bd['conflict_addressed']:.2f} SR={bd['stakeholder_reached']:.2f} "
              f"AS={bd['action_specificity']:.2f} FC={bd['format_compliance']:.2f} "
              f"NE={bd['no_escalation']:.2f} | {scenario['title']}")

model.eval()
print()
print("=" * 65)
print(f"POST-TRAINING COMPLETE — {TRAINING_STEPS} steps")
print(f"Avg reward (first 5):  {np.mean(total_rewards[:5]):.3f}")
print(f"Avg reward (last 5):   {np.mean(total_rewards[-5:]):.3f}")
print(f"Improvement:          +{np.mean(total_rewards[-5:]) - np.mean(total_rewards[:5]):.3f}")
print("=" * 65)

# %% CELL 6 — Professional Dual Plot (judge-ready)

def smooth(data, w=7):
    if len(data) < w: return data
    return np.convolve(data, np.ones(w)/w, mode="valid").tolist()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=150)
fig.patch.set_facecolor("#f8fafc")
steps = list(range(len(total_rewards)))

# ── TOP: Total reward with curriculum backgrounds ──
ax1.set_facecolor("#ffffff")

# Stage backgrounds
stage_colors = {"easy": "#dcfce7", "medium": "#fef9c3", "hard": "#fee2e2"}
prev_s = stages[0]; start = 0
for i in range(len(stages) + 1):
    s = stages[i] if i < len(stages) else None
    if s != prev_s or i == len(stages):
        ax1.axvspan(start, i, alpha=0.3, color=stage_colors[prev_s], label=prev_s.upper() if start == 0 or prev_s != stages[max(0,start-1)] else "")
        start = i
        if s: prev_s = s

# Raw dots
ax1.scatter(steps, total_rewards, alpha=0.35, s=18, color="#3b82f6", zorder=2, label="Raw reward")

# Smoothed line
sm = smooth(total_rewards, 7)
sm_x = list(range(3, 3 + len(sm)))
ax1.plot(sm_x, sm, color="#1d4ed8", linewidth=2.5, zorder=3, label="Smoothed (w=7)")

ax1.set_xlim(0, len(total_rewards))
ax1.set_ylim(0, 1.1)
ax1.set_xlabel("Training Step (Post-Training RL)", fontsize=12)
ax1.set_ylabel("Total Reward (0.0 — 1.0)", fontsize=12)
ax1.set_title("LifeOS Agent — Post-Training Progress", fontsize=16, fontweight="bold", pad=12)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(axis="y", alpha=0.3)

# Stage labels
for name, rng in [("EASY",(0,8)),("MEDIUM",(8,16)),("HARD",(16,60))]:
    mid = (min(rng[1], len(total_rewards)) + rng[0]) / 2
    if mid < len(total_rewards):
        ax1.text(mid, 1.05, name, ha="center", fontsize=10, fontweight="bold",
                 color={"EASY":"#16a34a","MEDIUM":"#ca8a04","HARD":"#dc2626"}[name])

# ── BOTTOM: 5 components ──
ax2.set_facecolor("#ffffff")
comp_colors = {"conflict_addressed":"#E74C3C","stakeholder_reached":"#3498DB",
               "action_specificity":"#2ECC71","format_compliance":"#F39C12","no_escalation":"#9B59B6"}
comp_labels = {"conflict_addressed":"Conflict (max 0.30)","stakeholder_reached":"Stakeholder (max 0.25)",
               "action_specificity":"Specificity (max 0.20)","format_compliance":"Format (max 0.15)","no_escalation":"No Generic (max 0.10)"}

for comp, color in comp_colors.items():
    vals = comp_history[comp]
    sm_c = smooth(vals, 7)
    sm_cx = list(range(3, 3 + len(sm_c)))
    ax2.plot(sm_cx, sm_c, color=color, linewidth=2, label=comp_labels[comp])

ax2.set_xlim(0, len(total_rewards))
ax2.set_ylim(0, 0.35)
ax2.set_xlabel("Training Step", fontsize=12)
ax2.set_ylabel("Component Reward Score", fontsize=12)
ax2.set_title("LifeOS Agent — 5 Independent Reward Components", fontsize=14, fontweight="bold", pad=12)
ax2.legend(loc="upper left", fontsize=9, ncol=2)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout(pad=3.0)
plt.savefig("reward_curve.png", dpi=150, bbox_inches="tight")
plt.savefig("components_curve.png", dpi=150, bbox_inches="tight")
plt.show()

print()
print("━" * 50)
print("POST-TRAINING RESULTS SUMMARY")
print("━" * 50)
print(f"Start reward (avg first 5 steps): {np.mean(total_rewards[:5]):.3f}")
print(f"End reward   (avg last 5 steps):  {np.mean(total_rewards[-5:]):.3f}")
print(f"Improvement:                     +{np.mean(total_rewards[-5:]) - np.mean(total_rewards[:5]):.3f}")
print("━" * 50)
print("⬇️  DOWNLOAD reward_curve.png NOW — needed for README")
print("⬇️  DOWNLOAD components_curve.png NOW — needed for README")

# %% CELL 7 — After Training Comparison

print("=" * 65)
print("POST-TRAINING EVALUATION — Total Travel Meltdown")
print("=" * 65)

after_response = generate_response(build_prompt(HARD_SCENARIO))
print(f"\nTrained model output:\n{after_response}\n")

after_parsed = parse_response(after_response)
after_reward, after_breakdown = compute_full_reward(after_parsed, HARD_SCENARIO)

print("BEFORE POST-TRAINING:")
print(stored_baseline_response[:300])
print()
print("AFTER POST-TRAINING:")
print(after_response[:300])
print()

print("IMPROVEMENT TABLE:")
print("━" * 60)
print(f"{'Component':<22} | {'Before':>6} | {'After':>6} | {'Change':>7} | Better?")
print("━" * 60)
for comp in stored_baseline_breakdown:
    b = stored_baseline_breakdown[comp]
    a = after_breakdown[comp]
    diff = a - b
    better = "✅" if diff > 0 else ("→" if diff == 0 else "❌")
    print(f"  {comp:<20} | {b:>6.2f} | {a:>6.2f} | {diff:>+6.2f} | {better}")
print("━" * 60)
diff_total = after_reward - stored_baseline_reward
better_total = "✅" if diff_total > 0 else "❌"
print(f"  {'TOTAL':<20} | {stored_baseline_reward:>6.3f} | {after_reward:>6.3f} | {diff_total:>+6.3f} | {better_total}")
print("━" * 60)

# Reward hacking audit
content = after_parsed.get("content","")
print()
print("REWARD HACKING AUDIT:")
print("━" * 40)
chk1 = len(content.strip()) >= 30
chk2 = not any(p in content.lower() for p in GENERIC_PHRASES)
chk3 = any(v in content.lower() for v in ACTION_VERBS)
print(f"  Content length: {len(content)} chars {'✅' if chk1 else '❌'} (min 30)")
print(f"  No generic phrases: {'✅' if chk2 else '❌'}")
print(f"  Has action verbs: {'✅' if chk3 else '❌'}")
print(f"  All checks passed: {'✅' if (chk1 and chk2 and chk3) else '❌'}")
print("━" * 40)


# %% CELL 8 — Save LoRA Adapters Correctly

# IMPORTANT: Save LoRA adapters ONLY
# Do NOT merge 4-bit/8-bit weights — damages model quality
# Per official hackathon guide: "LoRA saved incorrectly kills submissions"

SAVE_DIR = "lifeos_agent_lora_adapters"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print()
print("✅ LoRA adapters saved correctly")
print("⚠️  These are adapters only — NOT merged weights")
print(f"   Saved to: {SAVE_DIR}/")
print()
print("To load later:")
print("  from transformers import AutoModelForCausalLM, AutoTokenizer")
print("  from peft import PeftModel")
print(f"  base = AutoModelForCausalLM.from_pretrained('{MODEL_NAME}', load_in_8bit=True, device_map='auto')")
print(f"  model = PeftModel.from_pretrained(base, '{SAVE_DIR}')")
print(f"  tokenizer = AutoTokenizer.from_pretrained('{SAVE_DIR}')")
print()
print("To push to HuggingFace Hub:")
print("  model.push_to_hub('YOUR-USERNAME/lifeos-agent-lora')")
print("  tokenizer.push_to_hub('YOUR-USERNAME/lifeos-agent-lora')")
print()
print("🎉 POST-TRAINING COMPLETE — Download reward_curve.png and components_curve.png!")
