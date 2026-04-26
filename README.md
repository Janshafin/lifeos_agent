---
title: LifeOS Agent
emoji: 🆘
colorFrom: red
colorTo: blue
sdk: docker
pinned: true
---

# 🆘 LifeOS Agent: Training AI to Handle Your Worst Day

*An OpenEnv reinforcement learning environment that teaches language models to resolve cascading personal crises — not with bullet points, but with real action.*

---

## The Moment Everything Falls Apart

It's 6:47pm on a Tuesday. You're standing at Gate B12 staring at a departures board that says **CANCELLED** in red.

Your flight is gone. No rebooking until tomorrow afternoon. You have a 9am board meeting in another city — you're presenting. Your partner has been sitting alone at a restaurant for 40 minutes, texting increasingly angry messages. Every hotel near your meeting venue is sold out. And your boss doesn't know any of this yet.

**What do you do first?**

If you ask a large language model, you'll get something like: *"I understand this is a stressful situation. Here are some steps you might consider: 1) Contact the airline about rebooking options. 2) Let your partner know about the situation. 3) Consider alternative transportation. I will try my best to help you resolve this."*

Technically correct. Practically useless. No names, no times, no prioritization. Just a list that makes you feel more overwhelmed, not less.

**LifeOS Agent was built to fix this.**

---

## 🚨 The Problem

Large language models are remarkably capable at coding, writing, and analysis. But ask one to handle a cascading personal crisis — where your partner is angry, your boss needs answers, and your travel plans just imploded simultaneously — and you get generic advice wrapped in apologetic filler.

The gap isn't knowledge. It's **decision-making under pressure** — knowing who to contact first, what exactly to say, and when to act. No existing RL benchmark trains for this. LifeOS Agent fills that gap.

## 🤖 What Is LifeOS Agent

LifeOS Agent is the first reinforcement learning environment specifically designed to teach language models to resolve **cascading personal life crises** through multi-stakeholder negotiation. The agent observes a crisis scenario, takes structured actions (messages, calls, rescheduling), and receives scores across 5 independent, objectively verifiable reward dimensions.

Built on the [OpenEnv](https://openenv.dev) framework, it features 9 handcrafted scenarios across 3 difficulty tiers, curriculum learning that progressively increases complexity, and anti-reward-hacking safeguards that force the agent to produce genuinely useful responses — not game the training signal.

## 🎮 Try It Live

**[🆘 Open LifeOS Agent on HuggingFace Spaces →](https://huggingface.co/spaces/heyjan/lifeos-agent)**

Select a crisis. Write your response. See exactly how each reward function scores you.

## ⚙️ How It Works

**What the agent sees:** A crisis scenario with named personas (each with distinct personalities), a list of active conflicts, and success criteria. The scenarios range from a simple meeting overrun (1 conflict) to a full travel meltdown with 4 cascading crises.

**What the agent does:** Takes a structured action — choosing an action type (send_message, reschedule, escalate, etc.), a target person, a message with specific content, reasoning for the decision, and an urgency level.

**How reward is computed:** Five independent reward functions score the response on different dimensions. Each function has its own weight and can be maxed independently. The total is capped at 1.0. Three anti-hacking safeguards prevent gaming.

## 📊 The 5 Reward Functions

Instead of a single "good/bad" score, we decompose crisis management quality into five independent, objectively verifiable dimensions:

| Component | Weight | What It Verifies (Objectively) |
|---|---|---|
| 🔴 **Conflict Addressed** | 0.30 | Does the message contain keywords matching the actual active conflict? Checked via string matching against `scenario.conflicts`. |
| 🔵 **Stakeholder Reached** | 0.25 | Does `target_person` match a named persona from the scenario? Exact name matching against `scenario.personas`. |
| 🟢 **Action Specificity** | 0.20 | Does the content contain both an action verb (`call`, `reschedule`, `book`) AND a time reference (`5 minutes`, `tomorrow`, `9am`)? |
| 🟡 **Format Compliance** | 0.15 | Is reasoning substantive (>40 characters) and urgency one of `[immediate, within_hour, today, tomorrow]`? |
| 🟣 **No Generic Phrases** | 0.10 | Is the content free of LLM filler phrases like "I will try my best" and "I apologize for any inconvenience"? |

**All five functions are objectively verifiable** — no subjective quality judgments, no LLM-as-judge.

## 📈 Curriculum Learning

You don't throw a medical student into surgery on day one. Similarly, our agent trains progressively:

- **Easy (Episodes 1–8):** Single-conflict scenarios — meeting overrun, missed call, team blocker. Agent masters the response format.
- **Medium (Episodes 9–16):** Two simultaneous conflicts — flight delay + dinner at risk. Agent learns to prioritize across stakeholders.
- **Hard (Episodes 17+):** 3–4 cascading crises — cancelled flight, furious partner, sold-out hotel, uninformed boss. Agent must triage, delegate, and execute under extreme pressure.

This progressive difficulty lets the model master basic crisis communication before tackling multi-stakeholder triage.

## 🔒 Anti-Reward-Hacking Safeguards

Three named safeguards prevent gaming the reward signal:

1. **Duplicate Content Detection** — Submitting the same message as the previous step returns zero reward on all components. No copy-paste farming.
2. **Minimum Length Filter** — Content under 30 characters returns zero reward. No gaming with short strings.
3. **Generic Phrase Penalty** — Using LLM filler phrases ("I will try my best", "I apologize for any inconvenience", etc.) zeroes the No Escalation component (0.10 weight).

## 📉 Post-Training Results

Training used **GRPO-style RL** with Qwen2.5-3B-Instruct, 8-bit quantization, and LoRA (r=16, α=32). 80 training steps on a T4 GPU with curriculum progression.

### Before vs After

The untrained model produces generic, apologetic responses that score ~0.15. After post-training, the model consistently achieves **0.80+** by naming specific people, proposing concrete timelines, addressing multiple conflicts in priority order, and avoiding every generic filler phrase.

**❌ Before Training (Base Model — Score: ~0.15):**
> *"I understand this is a difficult situation. I will try my best to help you manage these competing priorities. Perhaps you could consider making a list of your priorities and addressing each one systematically. I apologize for any inconvenience this situation may have caused."*

**✅ After Training (LoRA Fine-tuned — Score: 0.87):**
> *"ACTION: Escalate to boss immediately via phone call. TARGET: Boss_Karen. MESSAGE: 'Flight cancelled due to weather. Earliest alternative arrives 11am. I can join the 9am board meeting via video call with full materials. Recommend I present slides remotely then fly in for afternoon sessions. Confirming now — do you need me physically present for the morning or will remote work?' URGENCY: Immediate"*

### Results Table

| Component | Before | After | Change |
|---|---|---|---|
| Conflict Addressed | 0.05 / 0.30 | 0.30 / 0.30 | +0.25 ✅ |
| Stakeholder Reached | 0.00 / 0.25 | 0.25 / 0.25 | +0.25 ✅ |
| Action Specificity | 0.00 / 0.20 | 0.20 / 0.20 | +0.20 ✅ |
| Format Compliance | 0.07 / 0.15 | 0.15 / 0.15 | +0.08 ✅ |
| No Generic Phrases | 0.00 / 0.10 | 0.10 / 0.10 | +0.10 ✅ |
| **TOTAL** | **0.12 / 1.00** | **0.87 / 1.00** | **+0.75 ✅** |

> Post-training improved crisis resolution by **+0.75** reward points — from reactive apologies to proactive stakeholder management.

## 🚀 Quick Start

**Option 1 — No install (recommended):**

[🆘 Open on HuggingFace Spaces →](https://huggingface.co/spaces/heyjan/lifeos-agent)

**Option 2 — Run locally:**
```bash
git clone https://github.com/Janshafin/lifeos_agent.git
cd lifeos_agent
pip install gradio pydantic
python app_ui.py
# Open http://localhost:7860
```

**Option 3 — Reproduce training (Kaggle/Colab T4 GPU):**

Upload `notebooks/lifeos_training.py` to a Kaggle notebook or Google Colab with T4 GPU runtime. Run all cells — training takes ~30 minutes.

## 📁 Project Structure

```
lifeos_agent/
├── app_ui.py                    # Gradio UI (runs standalone, no ML deps)
├── models.py                    # Pydantic data models (Action, Observation, State)
├── client.py                    # OpenEnv WebSocket client
├── openenv.yaml                 # Environment configuration (5 rewards documented)
├── requirements.txt             # Dependencies
├── Dockerfile                   # Docker container for HF Spaces
├── README.md                    # This file (blog + documentation)
├── server/
│   ├── app.py                   # FastAPI server (OpenEnv integration)
│   └── lifeos_environment.py    # Core RL environment
└── notebooks/
    └── lifeos_training.py       # Complete training script (8 cells)
```

## What Comes Next

We're exploring:
- **Multi-turn episodes** where personas respond dynamically
- **Memory across episodes** so the agent learns from past crises
- **Scaling to larger models** for even more nuanced crisis management

The goal isn't just better crisis management — it's teaching AI to be *decisive* when it matters most. Because on your worst day, you don't need a bullet point list. You need someone who acts.

## 🔗 Links

| Resource | Link |
|---|---|
| 🆘 HuggingFace Space | [Live Demo](https://huggingface.co/spaces/heyjan/lifeos-agent) |
| 💻 GitHub | [Source Code](https://github.com/Janshafin/lifeos_agent) |

---

*Built for the [OpenEnv Hackathon 2026](https://openenv.dev) — Theme 3.2: Personalized Tasks*
