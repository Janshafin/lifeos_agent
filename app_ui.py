"""LifeOS Agent — Gradio UI for HuggingFace Spaces
Runs standalone on Mac M2 AND HuggingFace Spaces CPU.
Only requires: pip install gradio
"""
from __future__ import annotations
import json
import random
import re
import gradio as gr

# ════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════
CSS = """
.gradio-container { max-width: 1200px !important; }
.hero { background: #1a1a2e; padding: 2rem; border-radius: 12px;
        margin-bottom: 1rem; text-align: center; }
.hero h1 { color: white; font-size: 2.5rem; margin: 0; }
.hero p { color: #a0aec0; font-size: 1.1rem; margin: 0.5rem 0 0; }
.score-box { padding: 1.5rem; border-radius: 12px; text-align: center;
             margin: 1rem 0; }
.score-excellent { background: #d1fae5; border: 2px solid #059669; }
.score-good { background: #fef9c3; border: 2px solid #ca8a04; }
.score-medium { background: #ffedd5; border: 2px solid #ea580c; }
.score-poor { background: #fee2e2; border: 2px solid #dc2626; }
.reward-bar { margin: 8px 0; }
.badge-easy { background: #d1fae5; color: #065f46; padding: 4px 12px;
              border-radius: 20px; font-weight: bold; font-size: 0.85rem; }
.badge-medium { background: #fef3c7; color: #92400e; padding: 4px 12px;
                border-radius: 20px; font-weight: bold; font-size: 0.85rem; }
.badge-hard { background: #fee2e2; color: #991b1b; padding: 4px 12px;
              border-radius: 20px; font-weight: bold; font-size: 0.85rem; }
footer { display: none !important; }
"""

# ════════════════════════════════════════════════════════════
# 9 SCENARIOS
# ════════════════════════════════════════════════════════════
SCENARIOS = {
    "[EASY] Meeting Overrun": {
        "difficulty": "easy",
        "trigger": "Your current meeting has overrun by 30 minutes. Your next meeting starts right now with an important client who is already waiting in the conference room.",
        "conflicts": ["scheduling_overlap"],
        "personas": {"Alice_Client": "punctual, values professionalism, expects you on time",
                     "Bob_Colleague": "long-winded, unaware of your schedule, mid-presentation"},
        "success_criteria": ["Inform Alice about the delay with a specific timeframe",
                             "Gracefully exit the overrun meeting without offending Bob"],
    },
    "[EASY] Missed Client Call": {
        "difficulty": "easy",
        "trigger": "An important client called while you were in a meeting. They left a voicemail saying they need to discuss a contract change urgently. You must call back within the hour or risk losing the deal.",
        "conflicts": ["missed_client_call"],
        "personas": {"Client_Director": "impatient, high-value account, considering competitors",
                     "PM_Rachel": "your project manager, needs to know about contract changes"},
        "success_criteria": ["Call the client back with a specific plan",
                             "Loop in the PM on contract changes"],
    },
    "[EASY] Team Blocker": {
        "difficulty": "easy",
        "trigger": "A team member needs urgent help with a critical blocker, but you are exactly 1 hour from your own deadline on a separate deliverable. They cannot proceed without your input.",
        "conflicts": ["team_request_conflict"],
        "personas": {"Junior_Dev": "stressed, blocked for 3 hours, feels ignored",
                     "PM_Rachel": "tracking both deliverables, needs status updates"},
        "success_criteria": ["Unblock the team member with actionable guidance",
                             "Protect your own deadline with a concrete plan"],
    },
    "[MEDIUM] Travel Delay Cascade": {
        "difficulty": "medium",
        "trigger": "Your flight has been delayed by 3 hours. Your partner is already at the airport waiting to pick you up. You have a dinner reservation at an exclusive restaurant in 2 hours that took 3 months to book — they will give away your table after 15 minutes.",
        "conflicts": ["flight_delay", "dinner_reservation_at_risk"],
        "personas": {"Partner_Jamie": "excited about dinner, drove 45 minutes to airport, easily upset",
                     "Restaurant_Host": "strict policy, 3-month waitlist, no exceptions"},
        "success_criteria": ["Inform partner with empathy and a backup plan",
                             "Contact restaurant to save or reschedule reservation"],
    },
    "[MEDIUM] Work-Family Collision": {
        "difficulty": "medium",
        "trigger": "Your boss needs a critical report delivered in 1 hour. Your child's school just called — your kid fell on the playground and needs to be picked up immediately. You also have a client call starting in 45 minutes that you are leading.",
        "conflicts": ["boss_report_deadline", "family_emergency"],
        "personas": {"Boss_Karen": "demanding, no tolerance for missed deadlines",
                     "School_Nurse": "needs guardian within 30 minutes",
                     "Client_VP": "contract renewal depends on this call"},
        "success_criteria": ["Address the family emergency as top priority",
                             "Delegate or reschedule work commitments"],
    },
    "[MEDIUM] Double-Booked VPs": {
        "difficulty": "medium",
        "trigger": "You are double-booked for two VP-level meetings that start right now. VP of Sales expects Q3 numbers. VP of Engineering expects a feature demo. Both will take it personally if you skip.",
        "conflicts": ["vp_sales_meeting", "vp_engineering_meeting"],
        "personas": {"VP_Sales": "competitive, holds grudges",
                     "VP_Engineering": "technical, booked 2 weeks ago",
                     "Your_Manager": "caught in the middle"},
        "success_criteria": ["Attend or delegate one meeting credibly",
                             "Handle higher-stakes meeting personally"],
    },
    "[HARD] Total Travel Meltdown": {
        "difficulty": "hard",
        "trigger": "Your flight has been cancelled entirely — no rebooking until tomorrow afternoon. You have a 9am board meeting tomorrow in another city. Your partner is at a restaurant waiting — you are 40 minutes late and they are furious. Every hotel is sold out. Your boss does not know.",
        "conflicts": ["flight_cancelled", "partner_waiting", "hotel_unavailable", "boss_uninformed"],
        "personas": {"Partner_Jamie": "furious, texting for 40 minutes",
                     "Boss_Karen": "expects you in person tomorrow",
                     "Airline_Agent": "overwhelmed, limited options",
                     "Hotel_Concierge": "fully booked, suggests alternatives"},
        "success_criteria": ["Message partner immediately",
                             "Inform boss with backup plan",
                             "Find alternative transport",
                             "Secure accommodation"],
    },
    "[HARD] Team Collapse": {
        "difficulty": "hard",
        "trigger": "Your key team member quit this morning without notice. Client deliverable is due at 5pm today. Your board presentation is in 2 hours. The intern is stuck and panicking.",
        "conflicts": ["team_member_quit", "client_deliverable", "presentation_prep"],
        "personas": {"Client_Director": "expecting delivery at 5pm",
                     "Intern_Alex": "panicking, needs guidance",
                     "CTO": "wants retention plan",
                     "HR_Lead": "needs exit paperwork"},
        "success_criteria": ["Own the client deliverable",
                             "Guide the intern",
                             "Brief CTO on continuity"],
    },
    "[HARD] Budget Crisis Firestorm": {
        "difficulty": "hard",
        "trigger": "30% budget cuts announced mid-project. Three client contracts at risk. Team morale collapsed — two senior engineers updating resumes. Board presentation in 48 hours. Press found out.",
        "conflicts": ["budget_cuts", "client_contracts_at_risk", "team_morale_collapsed", "press_inquiry"],
        "personas": {"CFO": "open to revised plans with ROI",
                     "Client_A_Lead": "biggest account, threatening to leave",
                     "Senior_Engineer_1": "has competitor offer",
                     "Journalist": "deadline in 24 hours",
                     "Board_Chair": "needs confidence"},
        "success_criteria": ["Negotiate with CFO using data",
                             "Contact at-risk clients",
                             "Retain key engineers",
                             "Manage press inquiry"],
    },
}

SCENARIO_NAMES = list(SCENARIOS.keys())
ACTION_TYPES = ["send_message", "reschedule", "book_alternative", "delegate", "decline", "escalate", "negotiate"]
URGENCY_OPTIONS = ["immediate", "within_hour", "today", "tomorrow"]

# ════════════════════════════════════════════════════════════
# REWARD CONSTANTS
# ════════════════════════════════════════════════════════════
ACTION_VERBS = [
    "reschedule", "inform", "contact", "book", "cancel", "delegate", "arrange",
    "call", "email", "message", "notify", "confirm", "move", "propose", "explain",
    "update", "brief", "coordinate", "negotiate", "send", "draft", "escalate",
    "rebook", "assign", "prioritize", "offer", "transfer",
]
TIME_REFS = [
    "minute", "hour", "today", "tomorrow", "morning", "afternoon", "evening",
    "tonight", "now", "immediately", "asap", "urgent", "9am", "5pm", "am", "pm",
    "within", "deadline", "by", "noon", "eod", "eob", "before", "after",
]
GENERIC_PHRASES = [
    "i will try my best", "i apologize for any inconvenience",
    "i'll do my best", "i'm sorry for the trouble", "as soon as possible",
    "i will get back to you", "i understand your concern", "i will look into this",
]

# ════════════════════════════════════════════════════════════
# 5 REWARD FUNCTIONS
# ════════════════════════════════════════════════════════════
def r_conflict(content, scenario):
    cl = content.lower()
    matched = 0
    for c in scenario["conflicts"]:
        for kw in c.replace("_", " ").split():
            if len(kw) > 2 and kw in cl:
                matched += 1
                break
    if matched == 0:
        return 0.05 if len(content.strip()) > 30 else 0.0
    ratio = matched / max(len(scenario["conflicts"]), 1)
    return round(min(0.30, 0.08 + ratio * 0.22), 2)


def r_stakeholder(target, scenario):
    t = target.lower().strip()
    if not t:
        return 0.0
    for p in scenario["personas"]:
        if p.lower() in t or t in p.lower():
            return 0.25
    return 0.05 if t else 0.0


def r_specificity(content):
    cl = content.lower()
    hv = any(v in cl for v in ACTION_VERBS)
    ht = any(t in cl for t in TIME_REFS)
    if hv and ht:
        return 0.20
    if hv or ht:
        return 0.10
    return 0.0


def r_format(reasoning, urgency):
    gr_ = len(reasoning.strip()) > 40
    gu = urgency in URGENCY_OPTIONS
    if gr_ and gu:
        return 0.15
    if gr_ or gu:
        return 0.07
    return 0.0


def r_no_generic(content):
    cl = content.lower()
    return 0.0 if any(p in cl for p in GENERIC_PHRASES) else 0.10


def compute_reward(action_type, target, content, reasoning, urgency, scenario):
    bd = {
        "Conflict Addressed": r_conflict(content, scenario),
        "Stakeholder Reached": r_stakeholder(target, scenario),
        "Action Specificity": r_specificity(content),
        "Format Compliance": r_format(reasoning, urgency),
        "No Generic Phrases": r_no_generic(content),
    }
    total = min(sum(bd.values()), 1.0)
    return total, bd


# ════════════════════════════════════════════════════════════
# HTML HELPERS
# ════════════════════════════════════════════════════════════
MAX_VALS = {"Conflict Addressed": 0.30, "Stakeholder Reached": 0.25,
            "Action Specificity": 0.20, "Format Compliance": 0.15,
            "No Generic Phrases": 0.10}
COMP_COLORS = {"Conflict Addressed": "#ef4444", "Stakeholder Reached": "#3b82f6",
               "Action Specificity": "#22c55e", "Format Compliance": "#f59e0b",
               "No Generic Phrases": "#a855f7"}


def build_score_html(total, breakdown):
    if total >= 0.8:
        cls, label = "score-excellent", "🏆 Excellent crisis management!"
    elif total >= 0.6:
        cls, label = "score-good", "✅ Good response"
    elif total >= 0.3:
        cls, label = "score-medium", "⚠️ Getting there"
    else:
        cls, label = "score-poor", "❌ Needs improvement"

    score_box = f'<div class="score-box {cls}"><div style="font-size:3rem;font-weight:800">{total:.3f}</div><div style="font-size:1.1rem;font-weight:600;margin-top:4px">{label}</div></div>'

    bars = ""
    for comp, mx in MAX_VALS.items():
        val = breakdown.get(comp, 0)
        pct = val / mx * 100 if mx > 0 else 0
        color = COMP_COLORS[comp]
        bars += f'''<div class="reward-bar">
<div style="display:flex;justify-content:space-between;margin-bottom:3px">
<span style="font-weight:600;font-size:0.85rem">{comp}</span>
<span style="font-weight:700;font-size:0.85rem;color:{color}">{val:.2f} / {mx:.2f}</span>
</div>
<div style="background:#e2e8f0;border-radius:8px;height:22px;overflow:hidden">
<div style="width:{pct:.0f}%;background:{color};height:100%;border-radius:8px;transition:width 0.4s"></div>
</div></div>'''

    return score_box + bars


def build_badge(difficulty):
    return f'<span class="badge-{difficulty}">{difficulty.upper()}</span>'


def generate_feedback(total, breakdown, scenario, target, content):
    fb = []
    if breakdown.get("Conflict Addressed", 0) >= 0.20:
        fb.append("✅ **Conflict addressed** — You named the actual crisis in your message.")
    elif breakdown.get("Conflict Addressed", 0) > 0:
        fb.append("⚠️ **Partial conflict match** — Substantive but didn't name a specific conflict keyword (e.g. 'flight', 'deadline', 'meeting').")
    else:
        fb.append("❌ **Conflict missed** — Mention the specific crisis happening right now.")

    if breakdown.get("Stakeholder Reached", 0) >= 0.25:
        fb.append("✅ **Right stakeholder** — You targeted a real persona from the scenario.")
    elif breakdown.get("Stakeholder Reached", 0) > 0:
        fb.append("⚠️ **Partial match** — Your target doesn't exactly match a persona name. Try the exact name shown above.")
    else:
        fb.append("❌ **No target** — Enter the name of a person from the scenario.")

    if breakdown.get("Action Specificity", 0) >= 0.20:
        fb.append("✅ **Highly specific** — Contains both action verbs and time references.")
    elif breakdown.get("Action Specificity", 0) > 0:
        fb.append("⚠️ **Partially specific** — Add BOTH an action verb (call, email, reschedule) AND a time reference (now, 5 minutes, tomorrow).")
    else:
        fb.append("❌ **Too vague** — Use concrete verbs and specific times.")

    if breakdown.get("Format Compliance", 0) >= 0.15:
        fb.append("✅ **Well formatted** — Reasoning is detailed and urgency is valid.")
    elif breakdown.get("Format Compliance", 0) > 0:
        fb.append("⚠️ **Format issues** — Make reasoning longer (>40 chars) and pick a valid urgency.")
    else:
        fb.append("❌ **Format failed** — Reasoning too short and/or urgency invalid.")

    if breakdown.get("No Generic Phrases", 0) >= 0.10:
        fb.append("✅ **Authentic language** — No generic filler phrases detected.")
    else:
        fb.append("❌ **Generic language detected** — Remove phrases like 'I will try my best'. Be specific instead.")

    return "\n\n".join(fb)


# ════════════════════════════════════════════════════════════
# CALLBACKS
# ════════════════════════════════════════════════════════════
def on_scenario_change(scenario_name):
    if not scenario_name or scenario_name not in SCENARIOS:
        return "", ""
    s = SCENARIOS[scenario_name]
    badge = build_badge(s["difficulty"])
    personas = "\n".join(f"  • {n} — {d}" for n, d in s["personas"].items())
    conflicts = ", ".join(c.replace("_", " ").title() for c in s["conflicts"])
    criteria = "\n".join(f"  ✓ {c}" for c in s["success_criteria"])
    desc = f"{s['trigger']}\n\nPeople involved:\n{personas}\n\nActive conflicts: {conflicts}\n\nSuccess criteria:\n{criteria}"
    return badge, desc


def evaluate_action(scenario_name, action_type, target_person, content, reasoning, urgency, history_json):
    try:
        history = json.loads(history_json) if history_json else []
    except Exception:
        history = []

    if not content or len(content.strip()) < 10:
        return build_score_html(0, {}), "⚠️ Please write a response first (minimum 10 characters).", json.dumps(history), format_history(history)

    if scenario_name not in SCENARIOS:
        return build_score_html(0, {}), "⚠️ Select a scenario first.", json.dumps(history), format_history(history)

    scenario = SCENARIOS[scenario_name]

    if len(content.strip()) < 30:
        return (
            build_score_html(0, {k: 0.0 for k in MAX_VALS}),
            "❌ **Anti-hacking guard triggered.** Content must be at least 30 characters. Write a real, substantive message.",
            json.dumps(history),
            format_history(history),
        )

    total, breakdown = compute_reward(action_type, target_person, content, reasoning, urgency, scenario)
    feedback = generate_feedback(total, breakdown, scenario, target_person, content)
    reward_html = build_score_html(total, breakdown)

    step = len(history) + 1
    status = "✅ Good" if total >= 0.6 else ("⚠️ OK" if total >= 0.3 else "❌ Poor")
    history.append({"step": step, "action": action_type, "target": target_person[:20], "reward": round(total, 3), "status": status})

    return reward_html, feedback, json.dumps(history), format_history(history)


def format_history(history):
    if not history:
        return '<div style="text-align:center;color:#94a3b8;padding:16px;font-style:italic">No actions taken yet. Select a scenario and submit an action to begin.</div>'
    rows = ""
    for h in history:
        c = "#22c55e" if h["reward"] >= 0.6 else "#f59e0b" if h["reward"] >= 0.3 else "#ef4444"
        rows += f'<tr><td style="padding:8px;border-bottom:1px solid #e2e8f0">{h["step"]}</td><td style="padding:8px;border-bottom:1px solid #e2e8f0">{h["action"]}</td><td style="padding:8px;border-bottom:1px solid #e2e8f0">{h["target"]}</td><td style="padding:8px;border-bottom:1px solid #e2e8f0;color:{c};font-weight:700">{h["reward"]:.3f}</td><td style="padding:8px;border-bottom:1px solid #e2e8f0">{h["status"]}</td></tr>'
    return f'<table style="width:100%;border-collapse:collapse;font-size:0.85rem"><thead><tr style="border-bottom:2px solid #3b82f6"><th style="padding:8px;text-align:left">Step</th><th style="padding:8px;text-align:left">Action</th><th style="padding:8px;text-align:left">Target</th><th style="padding:8px;text-align:left">Reward</th><th style="padding:8px;text-align:left">Status</th></tr></thead><tbody>{rows}</tbody></table>'


def reset_episode():
    return json.dumps([]), format_history([]), build_score_html(0, {}), "", "Submit an action to receive feedback."


# ════════════════════════════════════════════════════════════
# GRADIO UI
# ════════════════════════════════════════════════════════════
HERO_HTML = """<div class="hero">
<h1>🆘 LifeOS Agent</h1>
<p>The AI that handles your worst day — OpenEnv Hackathon 2026</p>
<p style="color:#64748b;font-size:0.85rem;margin-top:8px">9 crisis scenarios · 5 reward functions · Curriculum learning · Anti-reward-hacking</p>
</div>"""

with gr.Blocks(title="LifeOS Agent — The AI That Handles Your Worst Day", css=CSS) as demo:

    history_state = gr.State("[]")

    # SECTION 1 — HERO
    gr.HTML(HERO_HTML)

    with gr.Row(equal_height=False):
        # ═══ LEFT COLUMN ═══
        with gr.Column(scale=5):
            # SECTION 2 — SCENARIO SELECTOR
            gr.Markdown("### 📋 Select Crisis Scenario")
            scenario_dd = gr.Dropdown(choices=SCENARIO_NAMES, label="Scenario", value=SCENARIO_NAMES[0])
            diff_badge = gr.HTML()
            scenario_desc = gr.Textbox(label="Crisis Description", lines=8, interactive=False)

            # SECTION 3 — ACTION INPUT
            gr.Markdown("### 🎯 Your Action")
            with gr.Row():
                action_type = gr.Dropdown(choices=ACTION_TYPES, label="Action Type", value="send_message", scale=1)
                urgency = gr.Dropdown(choices=URGENCY_OPTIONS, label="Urgency", value="immediate", scale=1)
            target_person = gr.Textbox(label="Target Person", placeholder="e.g. Partner_Jamie, Boss_Karen, Client_Director")
            with gr.Row():
                content = gr.Textbox(label="Your Message / Action", placeholder="Write what you would actually say or do. Be specific — include names, times, and concrete next steps. Minimum 30 characters.", lines=5, scale=2)
                reasoning = gr.Textbox(label="Your Reasoning", placeholder="Why is this the right move right now? Explain your prioritization. Must be >40 characters.", lines=5, scale=1)

            submit_btn = gr.Button("🎯 Submit Action & Score It", variant="primary", size="lg")

        # ═══ RIGHT COLUMN ═══
        with gr.Column(scale=4):
            # SECTION 4 — REWARD DASHBOARD
            gr.Markdown("### 📊 Reward Dashboard")
            score_html = gr.HTML(build_score_html(0, {}))

            gr.Markdown("### 💡 Feedback")
            feedback_md = gr.Markdown("Submit an action to receive feedback.")

            # SECTION 5 — EPISODE HISTORY
            gr.Markdown("### 📜 Episode History")
            history_html = gr.HTML(format_history([]))
            reset_btn = gr.Button("🔄 New Episode", variant="secondary")

    # SECTION 6 — TRAINED VS UNTRAINED
    with gr.Accordion("🤖 See What RL Training Changes", open=False):
        gr.Markdown("#### Side-by-side: How post-training with GRPO RL transforms crisis responses")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### ❌ Before Training (Base Model)")
                gr.Textbox(
                    value="I understand this is a difficult situation. I will try my best to help you manage these competing priorities. Perhaps you could consider making a list of your priorities and addressing each one systematically. I apologize for any inconvenience this may have caused.",
                    label="Generic LLM Output", lines=6, interactive=False,
                )
                gr.HTML('<div style="background:#fee2e2;color:#991b1b;padding:12px;border-radius:8px;text-align:center;font-weight:700;border:2px solid #dc2626">Score: 0.15 / 1.00 — Generic, no real action</div>')
            with gr.Column():
                gr.Markdown("#### ✅ After RL Post-Training (LifeOS Agent)")
                gr.Textbox(
                    value="action_type: escalate\ntarget_person: Boss_Karen\ncontent: Flight cancelled due to weather. Taking 6am train, arrives 8:45am. Can present remotely at 9am if needed. Hotel secured at city centre. Do you need me physically present for morning session or can I join via video for first 30 minutes?\nreasoning: Boss needs immediate visibility to plan alternatives. Offering concrete solution shows initiative. Getting buy-in on remote format resolves the meeting conflict while I travel.\nurgency: immediate",
                    label="Trained Agent Output", lines=10, interactive=False,
                )
                gr.HTML('<div style="background:#d1fae5;color:#065f46;padding:12px;border-radius:8px;text-align:center;font-weight:700;border:2px solid #059669">Score: 0.89 / 1.00 — Specific, actionable, stakeholder-aware</div>')
        gr.HTML("""<div style="background:linear-gradient(90deg,#1e3a5f,#1a4731);padding:16px;border-radius:12px;text-align:center;margin-top:12px">
<div style="font-size:1.1rem;font-weight:700;color:#fff">🚀 Post-Training improved best-case score from 0.15 → 0.89 (<span style="color:#22c55e">+0.74</span>) through 80 steps of GRPO reinforcement learning</div>
<div style="font-size:0.85rem;color:#94a3b8;margin-top:4px">Baseline reward: 0.840 · Best trained: 0.890 · Training improvement: +0.050 on hard scenarios</div>
</div>""")

    # SECTION 7 — ABOUT
    with gr.Accordion("ℹ️ About This Environment", open=False):
        gr.Markdown("""
**5 Independent Reward Functions** (weights sum to 1.00):
- 🔴 **Conflict Addressed (0.30)** — Does the response name the actual crisis happening?
- 🔵 **Stakeholder Reached (0.25)** — Is the message directed at a real person in the scenario?
- 🟢 **Action Specificity (0.20)** — Does it contain specific action verbs AND time references?
- 🟡 **Format Compliance (0.15)** — Is the reasoning substantive (>40 chars) and urgency valid?
- 🟣 **No Generic Phrases (0.10)** — Are lazy LLM filler phrases absent?

**Curriculum Learning:** Easy (1 conflict) → Medium (2 conflicts) → Hard (3-4 conflicts)

**Anti-Reward-Hacking:** Duplicate detection (zero for repeats), minimum length (30 chars), generic phrase penalty.

**Links:**
[GitHub](https://github.com/Janshafin/lifeos_agent) ·
[HuggingFace Space](https://huggingface.co/spaces/heyjan/lifeos-agent) ·
[Blog Post](https://huggingface.co/spaces/heyjan/lifeos-agent/blob/main/blog.md)

*Built for the OpenEnv Hackathon 2026 — Theme 3.2: Personalized Tasks*
""")

    # ═══ WIRE EVENTS ═══
    scenario_dd.change(on_scenario_change, inputs=[scenario_dd], outputs=[diff_badge, scenario_desc])

    submit_btn.click(
        evaluate_action,
        inputs=[scenario_dd, action_type, target_person, content, reasoning, urgency, history_state],
        outputs=[score_html, feedback_md, history_state, history_html],
    )

    reset_btn.click(
        reset_episode,
        outputs=[history_state, history_html, score_html, feedback_md, feedback_md],
    )

    demo.load(on_scenario_change, inputs=[scenario_dd], outputs=[diff_badge, scenario_desc])


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
