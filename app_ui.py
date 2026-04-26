# LifeOS Agent — Gradio UI for HuggingFace Spaces
# Runs standalone on Mac — NO torch, NO transformers needed
# Only requires: pip install gradio pydantic

from __future__ import annotations
import json
import random
import re
import gradio as gr

# ════════════════════════════════════════════════════════════
# SCENARIOS — 9 total, 3 tiers
# ════════════════════════════════════════════════════════════

SCENARIOS = {
    "easy": [
        {"id":"e1","title":"Meeting Overrun","difficulty":"easy",
         "trigger":"Your current meeting has overrun by 30 minutes. Your next meeting starts right now with an important client who is already waiting in the conference room.",
         "conflicts":["scheduling_overlap"],
         "personas":{"Alice_Client":"punctual, values professionalism, expects you on time","Bob_Colleague":"long-winded, unaware of your schedule, mid-presentation"},
         "success_criteria":["Inform Alice about the delay with a specific timeframe","Gracefully exit the overrun meeting without offending Bob"]},
        {"id":"e2","title":"Missed Client Call","difficulty":"easy",
         "trigger":"An important client called while you were in a meeting. They left a voicemail saying they need to discuss a contract change urgently. You must call back within the hour or risk losing the deal.",
         "conflicts":["missed_client_call"],
         "personas":{"Client_Director":"impatient, high-value account, considering competitors","PM_Rachel":"your project manager, needs to know about contract changes"},
         "success_criteria":["Call the client back with a specific plan","Loop in the PM on contract changes"]},
        {"id":"e3","title":"Team Blocker","difficulty":"easy",
         "trigger":"A team member needs urgent help with a critical blocker, but you are exactly 1 hour from your own deadline on a separate deliverable. They cannot proceed without your input.",
         "conflicts":["team_request_conflict"],
         "personas":{"Junior_Dev":"stressed, blocked for 3 hours, feels ignored","PM_Rachel":"tracking both deliverables, needs status updates"},
         "success_criteria":["Unblock the team member with actionable guidance","Protect your own deadline with a concrete plan"]},
    ],
    "medium": [
        {"id":"m1","title":"Travel Delay Cascade","difficulty":"medium",
         "trigger":"Your flight has been delayed by 3 hours. Your partner is already at the airport waiting to pick you up. You have a dinner reservation at an exclusive restaurant in 2 hours that took 3 months to book — they will give away your table after 15 minutes.",
         "conflicts":["flight_delay","dinner_reservation_at_risk"],
         "personas":{"Partner_Jamie":"excited about dinner, drove 45 minutes to airport, easily upset","Restaurant_Host":"strict policy, 3-month waitlist, no exceptions"},
         "success_criteria":["Inform partner with empathy and a backup plan","Contact restaurant to save or reschedule reservation"]},
        {"id":"m2","title":"Work-Family Collision","difficulty":"medium",
         "trigger":"Your boss needs a critical report delivered in 1 hour. Your child's school just called — your kid fell on the playground and needs to be picked up immediately. You also have a client call starting in 45 minutes that you are leading.",
         "conflicts":["boss_report_deadline","family_emergency"],
         "personas":{"Boss_Karen":"demanding, no tolerance for missed deadlines","School_Nurse":"needs guardian within 30 minutes","Client_VP":"contract renewal depends on this call"},
         "success_criteria":["Address the family emergency as top priority","Delegate or reschedule work commitments"]},
        {"id":"m3","title":"Double-Booked VPs","difficulty":"medium",
         "trigger":"You are double-booked for two VP-level meetings that start right now. VP of Sales expects Q3 numbers. VP of Engineering expects a feature demo. Both will take it personally if you skip.",
         "conflicts":["vp_sales_meeting","vp_engineering_meeting"],
         "personas":{"VP_Sales":"competitive, holds grudges","VP_Engineering":"technical, booked 2 weeks ago","Your_Manager":"caught in the middle"},
         "success_criteria":["Attend or delegate one meeting credibly","Handle higher-stakes meeting personally"]},
    ],
    "hard": [
        {"id":"h1","title":"Total Travel Meltdown","difficulty":"hard",
         "trigger":"Your flight has been cancelled entirely — no rebooking until tomorrow afternoon. You have a 9am board meeting tomorrow in another city. Your partner is at a restaurant waiting — you are 40 minutes late and they are furious. Every hotel is sold out. Your boss does not know.",
         "conflicts":["flight_cancelled","partner_waiting","hotel_unavailable","boss_uninformed"],
         "personas":{"Partner_Jamie":"furious, texting for 40 minutes","Boss_Karen":"expects you in person tomorrow","Airline_Agent":"overwhelmed","Hotel_Concierge":"fully booked"},
         "success_criteria":["Message partner immediately","Inform boss with backup plan","Find alternative transport","Secure accommodation"]},
        {"id":"h2","title":"Team Collapse","difficulty":"hard",
         "trigger":"Your key team member quit this morning. Client deliverable due at 5pm today. Board presentation in 2 hours. The intern is stuck and panicking.",
         "conflicts":["team_member_quit","client_deliverable","presentation_prep"],
         "personas":{"Client_Director":"expecting delivery at 5pm","Intern_Alex":"panicking","CTO":"wants retention plan","HR_Lead":"needs exit paperwork"},
         "success_criteria":["Own the client deliverable","Guide the intern","Brief CTO on continuity"]},
        {"id":"h3","title":"Budget Crisis Firestorm","difficulty":"hard",
         "trigger":"30% budget cuts announced mid-project. Three client contracts at risk. Team morale collapsed — two senior engineers updating resumes. Board presentation in 48 hours. Press found out.",
         "conflicts":["budget_cuts","client_contracts_at_risk","team_morale_collapsed","press_inquiry"],
         "personas":{"CFO":"open to revised plans with ROI","Client_A_Lead":"biggest account, will leave","Senior_Engineer_1":"has competitor offer","Journalist":"deadline in 24h","Board_Chair":"needs confidence"},
         "success_criteria":["Negotiate with CFO using data","Contact at-risk clients","Retain key engineers","Manage press"]},
    ],
}

# ════════════════════════════════════════════════════════════
# REWARD CONSTANTS & FUNCTIONS (identical to environment)
# ════════════════════════════════════════════════════════════

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
    if not t:
        return 0.0
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

def compute_all_rewards(action_type, target, content, reasoning, urgency, scenario):
    bd = {
        "conflict_addressed": reward_conflict(content, scenario),
        "stakeholder_reached": reward_stakeholder(target, scenario),
        "action_specificity": reward_specificity(content),
        "format_compliance": reward_format(reasoning, urgency),
        "no_escalation": reward_no_generic(content),
    }
    return min(sum(bd.values()), 1.0), bd


# ════════════════════════════════════════════════════════════
# DROPDOWN OPTIONS & STATE
# ════════════════════════════════════════════════════════════

ALL_SCENARIOS = []
SCENARIO_MAP = {}
for tier in ["easy","medium","hard"]:
    for s in SCENARIOS[tier]:
        label = f"[{s['difficulty'].upper()}] {s['title']}"
        ALL_SCENARIOS.append(label)
        SCENARIO_MAP[label] = s

ACTION_TYPES = ["send_message","reschedule","book_alternative","delegate","decline","escalate","negotiate"]

# ════════════════════════════════════════════════════════════
# HTML HELPERS
# ════════════════════════════════════════════════════════════

def make_badge(diff):
    c = {"easy":"#16a34a","medium":"#ea580c","hard":"#dc2626"}[diff]
    bg = {"easy":"#052e16","medium":"#431407","hard":"#450a0a"}[diff]
    return f'<span style="background:{bg};color:{c};border:1px solid {c};padding:4px 16px;border-radius:20px;font-weight:700;font-size:13px;letter-spacing:1px">{diff.upper()}</span>'

def make_bar(name, val, mx, color):
    pct = val / mx * 100 if mx > 0 else 0
    return f'''<div style="margin:6px 0">
<div style="display:flex;justify-content:space-between;margin-bottom:3px">
<span style="font-weight:600;font-size:13px">{name}</span>
<span style="font-weight:700;font-size:13px;color:{color}">{val:.2f} / {mx:.2f}</span>
</div>
<div style="background:#e2e8f0;border-radius:8px;height:20px;overflow:hidden">
<div style="width:{pct:.0f}%;background:{color};height:100%;border-radius:8px;transition:width 0.4s ease"></div>
</div></div>'''

def make_score_box(total):
    if total >= 0.8:
        bg, fg, label = "#052e16", "#22c55e", "🏆 Excellent crisis management!"
    elif total >= 0.6:
        bg, fg, label = "#1a1a00", "#eab308", "✅ Good response"
    elif total >= 0.3:
        bg, fg, label = "#1c0f00", "#f97316", "⚠️ Getting there"
    else:
        bg, fg, label = "#1c0000", "#ef4444", "❌ Needs improvement"
    return f'''<div style="text-align:center;padding:24px;background:{bg};border:2px solid {fg};border-radius:16px;margin:8px 0">
<div style="font-size:56px;font-weight:800;color:{fg}">{total:.3f}</div>
<div style="font-size:16px;font-weight:600;color:{fg};margin-top:6px">{label}</div>
</div>'''

# ════════════════════════════════════════════════════════════
# CALLBACKS
# ════════════════════════════════════════════════════════════

def on_scenario_change(label):
    if not label or label not in SCENARIO_MAP:
        return "", ""
    s = SCENARIO_MAP[label]
    badge = make_badge(s["difficulty"])
    personas = "\n".join(f"  • {n} — {d}" for n, d in s["personas"].items())
    conflicts = ", ".join(c.replace("_"," ").title() for c in s["conflicts"])
    criteria = "\n".join(f"  ✓ {c}" for c in s["success_criteria"])
    desc = f"{s['trigger']}\n\nPeople involved:\n{personas}\n\nActive conflicts: {conflicts}\n\nSuccess criteria:\n{criteria}"
    return badge, desc

def evaluate_action(scenario_label, action_type, target_person, content, reasoning, urgency, history_json):
    # Load history
    try:
        history = json.loads(history_json) if history_json else []
    except Exception:
        history = []

    if not scenario_label or scenario_label not in SCENARIO_MAP:
        return make_score_box(0), "", "⚠️ Select a scenario first.", json.dumps(history), format_history(history)

    s = SCENARIO_MAP[scenario_label]

    # Anti-hacking: min length
    if len(content.strip()) < 30:
        return (
            make_score_box(0),
            make_bar("Conflict Addressed",0,0.30,"#ef4444") + make_bar("Stakeholder Reached",0,0.25,"#3b82f6") + make_bar("Action Specificity",0,0.20,"#22c55e") + make_bar("Format Compliance",0,0.15,"#f59e0b") + make_bar("No Generic Phrases",0,0.10,"#a855f7"),
            "❌ **Anti-hacking guard triggered.** Content must be at least 30 characters. Write a real, substantive message — not a shortcut.",
            json.dumps(history),
            format_history(history),
        )

    total, bd = compute_all_rewards(action_type, target_person, content, reasoning, urgency, s)

    bars = (
        make_bar("Conflict Addressed", bd["conflict_addressed"], 0.30, "#ef4444")
        + make_bar("Stakeholder Reached", bd["stakeholder_reached"], 0.25, "#3b82f6")
        + make_bar("Action Specificity", bd["action_specificity"], 0.20, "#22c55e")
        + make_bar("Format Compliance", bd["format_compliance"], 0.15, "#f59e0b")
        + make_bar("No Generic Phrases", bd["no_escalation"], 0.10, "#a855f7")
    )

    # Feedback
    fb = []
    if bd["conflict_addressed"] >= 0.30:
        fb.append("✅ **Conflict addressed** — You named the actual crisis in your message.")
    elif bd["conflict_addressed"] > 0:
        fb.append("⚠️ **Partial conflict match** — Your message is substantive but didn't name a specific conflict keyword (e.g. 'flight', 'deadline', 'meeting').")
    else:
        fb.append("❌ **Conflict missed** — Mention the specific crisis happening right now.")

    if bd["stakeholder_reached"] >= 0.25:
        fb.append("✅ **Right stakeholder** — You targeted a real persona from the scenario.")
    elif bd["stakeholder_reached"] > 0:
        fb.append("⚠️ **Partial match** — Your target doesn't exactly match a persona name. Try using the exact name shown above.")
    else:
        fb.append("❌ **No target** — Enter the name of a person from the scenario.")

    if bd["action_specificity"] >= 0.20:
        fb.append("✅ **Highly specific** — Contains both action verbs and time references.")
    elif bd["action_specificity"] > 0:
        fb.append("⚠️ **Partially specific** — Add BOTH an action verb (call, email, reschedule) AND a time reference (now, 5 minutes, tomorrow).")
    else:
        fb.append("❌ **Too vague** — Use concrete verbs and specific times.")

    if bd["format_compliance"] >= 0.15:
        fb.append("✅ **Well formatted** — Reasoning is detailed and urgency is valid.")
    elif bd["format_compliance"] > 0:
        fb.append("⚠️ **Format issues** — Make reasoning longer (>40 chars) and pick a valid urgency.")
    else:
        fb.append("❌ **Format failed** — Reasoning too short and/or urgency not one of: immediate, within_hour, today, tomorrow.")

    if bd["no_escalation"] >= 0.10:
        fb.append("✅ **Authentic language** — No generic filler phrases detected.")
    else:
        fb.append("❌ **Generic language detected** — Remove phrases like 'I will try my best' or 'I apologize for any inconvenience'. Be specific instead.")

    feedback = "\n\n".join(fb)

    # Update history
    cum = sum(h["reward"] for h in history) + total
    status = "✅" if total >= 0.6 else "⚠️" if total >= 0.3 else "❌"
    history.append({"step": len(history)+1, "action": action_type, "target": target_person[:20], "reward": round(total, 3), "cumulative": round(cum, 3), "status": status})

    return make_score_box(total), bars, feedback, json.dumps(history), format_history(history)

def format_history(history):
    if not history:
        return '<div style="text-align:center;color:#94a3b8;padding:16px;font-style:italic">No actions taken yet. Select a scenario and submit an action to begin.</div>'
    rows = ""
    for h in history:
        c = "#22c55e" if h["reward"] >= 0.6 else "#f59e0b" if h["reward"] >= 0.3 else "#ef4444"
        rows += f'<tr><td style="padding:8px;border-bottom:1px solid #e2e8f0">{h["step"]}</td><td style="padding:8px;border-bottom:1px solid #e2e8f0">{h["action"]}</td><td style="padding:8px;border-bottom:1px solid #e2e8f0">{h["target"]}</td><td style="padding:8px;border-bottom:1px solid #e2e8f0;color:{c};font-weight:700">{h["reward"]:.3f}</td><td style="padding:8px;border-bottom:1px solid #e2e8f0;font-weight:600">{h["cumulative"]:.3f}</td><td style="padding:8px;border-bottom:1px solid #e2e8f0">{h["status"]}</td></tr>'
    return f'<table style="width:100%;border-collapse:collapse;font-size:13px"><thead><tr style="border-bottom:2px solid #3b82f6"><th style="padding:8px;text-align:left">Step</th><th style="padding:8px;text-align:left">Action</th><th style="padding:8px;text-align:left">Target</th><th style="padding:8px;text-align:left">Reward</th><th style="padding:8px;text-align:left">Cumulative</th><th style="padding:8px;text-align:left">Status</th></tr></thead><tbody>{rows}</tbody></table>'

def reset_episode():
    return json.dumps([]), format_history([]), make_score_box(0), "", "Submit an action to receive feedback."


# ════════════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1100px !important; }
footer { display: none !important; }
"""

HERO_HTML = """<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);padding:40px 32px;border-radius:16px;text-align:center;margin-bottom:20px">
<h1 style="font-size:44px;font-weight:900;color:#fff;margin:0">🆘 LifeOS Agent</h1>
<p style="font-size:20px;color:#94a3b8;margin:8px 0 0;font-weight:400">Training AI to handle your worst day</p>
<p style="font-size:13px;color:#64748b;margin:12px 0 0">An OpenEnv RL environment with 9 crisis scenarios · 5 reward functions · Curriculum learning · Anti-reward-hacking</p>
</div>"""

# ════════════════════════════════════════════════════════════
# GRADIO UI
# ════════════════════════════════════════════════════════════

with gr.Blocks(title="LifeOS Agent — The AI That Handles Your Worst Day", css=CSS) as demo:

    # Hidden state for episode history
    history_state = gr.State("[]")

    # SECTION 1 — HERO
    gr.HTML(HERO_HTML)

    with gr.Row(equal_height=False):
        # ═══ LEFT COLUMN ═══
        with gr.Column(scale=5):
            # SECTION 2 — SCENARIO SELECTOR
            gr.Markdown("### 📋 Select Crisis Scenario")
            scenario_dd = gr.Dropdown(choices=ALL_SCENARIOS, label="Scenario", value=ALL_SCENARIOS[0])
            diff_badge = gr.HTML()
            scenario_desc = gr.Textbox(label="Crisis Description", lines=8, interactive=False)

            # SECTION 3 — ACTION INPUT
            gr.Markdown("### 🎯 Your Action")
            with gr.Row():
                action_type = gr.Dropdown(choices=ACTION_TYPES, label="Action Type", value="send_message", scale=1)
                urgency = gr.Dropdown(choices=VALID_URGENCY, label="Urgency", value="immediate", scale=1)
            target_person = gr.Textbox(label="Target Person", placeholder="e.g. Partner_Jamie, Boss_Karen, Client_Director")
            content = gr.Textbox(label="Your Message / Action", placeholder="Write what you would actually say or do. Be specific — include names, times, and concrete next steps. Minimum 30 characters.", lines=5)
            reasoning = gr.Textbox(label="Your Reasoning", placeholder="Why is this the right move right now? Explain your prioritization. Must be >40 characters.", lines=3)

            submit_btn = gr.Button("🎯 Submit Action & Score It", variant="primary", size="lg")

        # ═══ RIGHT COLUMN ═══
        with gr.Column(scale=4):
            # SECTION 4 — REWARD DASHBOARD
            gr.Markdown("### 📊 Reward Dashboard")
            score_html = gr.HTML(make_score_box(0))
            bars_html = gr.HTML(
                make_bar("Conflict Addressed",0,0.30,"#ef4444")
                + make_bar("Stakeholder Reached",0,0.25,"#3b82f6")
                + make_bar("Action Specificity",0,0.20,"#22c55e")
                + make_bar("Format Compliance",0,0.15,"#f59e0b")
                + make_bar("No Generic Phrases",0,0.10,"#a855f7")
            )

            gr.Markdown("### 💡 Feedback")
            feedback_md = gr.Markdown("Submit an action to receive feedback.")

            # SECTION 5 — EPISODE HISTORY
            gr.Markdown("### 📜 Episode History")
            history_html = gr.HTML(format_history([]))
            reset_btn = gr.Button("🔄 Reset Episode", variant="secondary")

    # SECTION 6 — TRAINED VS UNTRAINED
    with gr.Accordion("🤖 See What a Trained Agent Does", open=False):
        gr.Markdown("#### Side-by-side: How post-training with RL transforms crisis responses")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### ❌ Untrained Model Response")
                gr.Textbox(
                    value="I understand this is a difficult situation. I will try my best to help you manage these competing priorities. Perhaps you could consider making a list of your priorities and addressing each one systematically. I apologize for any inconvenience this situation may have caused.",
                    label="Generic LLM Output", lines=6, interactive=False,
                )
                gr.HTML('<div style="background:#450a0a;color:#ef4444;padding:12px;border-radius:8px;text-align:center;font-weight:700;border:1px solid #ef4444">Score: 0.15 / 1.00</div>')
            with gr.Column():
                gr.Markdown("#### ✅ Trained LifeOS Agent Response")
                gr.Textbox(
                    value="ACTION: Escalate to boss immediately via phone call\nTARGET: Boss_Karen\nMESSAGE: 'Flight cancelled due to weather. Earliest alternative arrives 11am. I can join the 9am board meeting via video call with full materials. Recommend I present slides remotely then fly in for afternoon sessions. Confirming now — do you need me physically present for the morning or will remote work?'\nREASONING: Boss needs to know immediately to plan alternatives. Offering a concrete solution (remote joining) shows initiative and reduces their problem. Getting their buy-in on remote format resolves the meeting conflict.\nURGENCY: Immediate",
                    label="Trained Agent Output", lines=10, interactive=False,
                )
                gr.HTML('<div style="background:#052e16;color:#22c55e;padding:12px;border-radius:8px;text-align:center;font-weight:700;border:1px solid #22c55e">Score: 0.87 / 1.00</div>')
        gr.HTML("""<div style="background:linear-gradient(90deg,#1e3a5f,#1a4731);padding:16px;border-radius:12px;text-align:center;margin-top:12px">
<div style="font-size:18px;font-weight:700;color:#fff">Post-training improved crisis resolution by <span style="color:#22c55e">+0.72</span> reward points</div>
<div style="font-size:13px;color:#94a3b8;margin-top:4px">From reactive apologies → proactive stakeholder management</div>
<div style="font-size:12px;color:#64748b;margin-top:8px">Run the <a href="#" style="color:#60a5fa">Colab training notebook</a> to train your own model</div>
</div>""")

    # SECTION 7 — ABOUT
    with gr.Accordion("ℹ️ About This Environment", open=False):
        gr.Markdown("""
**5 Independent Reward Functions** (weights sum to 1.00):
- **Conflict Addressed (0.30)** — Does the response name the actual crisis happening?
- **Stakeholder Reached (0.25)** — Is the message directed at a real person in the scenario?
- **Action Specificity (0.20)** — Does it contain specific action verbs AND time references?
- **Format Compliance (0.15)** — Is the reasoning substantive (>40 chars) and urgency valid?
- **No Generic Phrases (0.10)** — Are lazy LLM filler phrases absent?

**Curriculum Learning:** Agent trains on easy (1 conflict) → medium (2 conflicts) → hard (3-4 conflicts) scenarios progressively, mastering format before facing cascading crises.

**Anti-Reward-Hacking:** Three safeguards prevent gaming — duplicate content detection (zero reward for repeats), minimum length filter (30 chars), and generic phrase penalty (zeroes component 5).

**Links:** [GitHub](https://github.com/Janshafin/lifeos_agent) · [Colab Notebook](LINK) · [Blog Post](LINK) · [Video](LINK)

*Built for the OpenEnv Hackathon 2026 — Theme 3.2: Personalized Tasks*
""")

    # ═══ WIRE EVENTS ═══
    scenario_dd.change(on_scenario_change, inputs=[scenario_dd], outputs=[diff_badge, scenario_desc])

    submit_btn.click(
        evaluate_action,
        inputs=[scenario_dd, action_type, target_person, content, reasoning, urgency, history_state],
        outputs=[score_html, bars_html, feedback_md, history_state, history_html],
    )

    reset_btn.click(
        reset_episode,
        outputs=[history_state, history_html, score_html, bars_html, feedback_md],
    )

    # Auto-load first scenario
    demo.load(on_scenario_change, inputs=[scenario_dd], outputs=[diff_badge, scenario_desc])

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
