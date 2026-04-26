---
title: LifeOS Agent
emoji: 🆘
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app_ui.py
pinned: true
---

It's 6:47pm on a Tuesday.

Your flight just got cancelled. Your partner has been waiting at the restaurant for 40 minutes — furious. Your boss still doesn't know you might miss tomorrow's 9am board meeting. Every hotel near the meeting venue is sold out.

Today's AI gives you a bullet point list.
**LifeOS Agent actually resolves it.**

---

## 🚨 The Problem

Large language models are remarkably capable at coding, writing, and analysis. But ask one to handle a cascading personal crisis — where your partner is angry, your boss needs answers, and your travel plans just imploded simultaneously — and you get generic advice wrapped in apologetic filler: *"I understand this is a difficult situation. I will try my best to help."*

The gap isn't knowledge. It's **decision-making under pressure** — knowing who to contact first, what exactly to say, and when to act. No existing RL benchmark trains for this. LifeOS Agent fills that gap.

## 🤖 What Is LifeOS Agent

LifeOS Agent is the first reinforcement learning environment specifically designed to teach language models to resolve **cascading personal life crises** through multi-stakeholder negotiation. The agent observes a crisis scenario, takes structured actions (messages, calls, rescheduling), and receives scores across 5 independent, objectively verifiable reward dimensions.

Built on the [OpenEnv](https://github.com/openenv) framework, it features 9 handcrafted scenarios across 3 difficulty tiers, curriculum learning that progressively increases complexity, and anti-reward-hacking safeguards that force the agent to produce genuinely useful responses — not game the training signal.

## 🎮 Try It Live

**[🆘 Open LifeOS Agent on HuggingFace Spaces →](YOUR-SPACE-URL)**

Select a crisis. Write your response. See exactly how each reward function scores you.

## ⚙️ How It Works

**What the agent sees:** A crisis scenario with named personas (each with distinct personalities), a list of active conflicts, and success criteria. The scenarios range from a simple meeting overrun (1 conflict) to a full travel meltdown with 4 cascading crises.

**What the agent does:** Takes a structured action — choosing an action type (send_message, reschedule, escalate, etc.), a target person, a message with specific content, reasoning for the decision, and an urgency level.

**How reward is computed:** Five independent reward functions score the response on different dimensions. Each function has its own weight and can be maxed independently. The total is capped at 1.0. Three anti-hacking safeguards prevent gaming.

## 📊 The 5 Reward Functions

| Component | Weight | What It Verifies (Objectively) |
|---|---|---|
| 🔴 **Conflict Addressed** | 0.30 | Does the message contain keywords matching the actual active conflict? Checked via string matching against `scenario.conflicts`. |
| 🔵 **Stakeholder Reached** | 0.25 | Does `target_person` match a named persona from the scenario? Exact name matching against `scenario.personas`. |
| 🟢 **Action Specificity** | 0.20 | Does the content contain both an action verb (`call`, `reschedule`, `book`) AND a time reference (`5 minutes`, `tomorrow`, `9am`)? |
| 🟡 **Format Compliance** | 0.15 | Is reasoning substantive (>40 characters) and urgency one of `[immediate, within_hour, today, tomorrow]`? |
| 🟣 **No Generic Phrases** | 0.10 | Is the content free of LLM filler phrases like "I will try my best" and "I apologize for any inconvenience"? |

All five functions are **objectively verifiable** — no subjective quality judgments, no LLM-as-judge.

## 📈 Curriculum Learning

- **Easy (Episodes 1–8):** Single-conflict scenarios — meeting overrun, missed call, team blocker. Agent masters the response format.
- **Medium (Episodes 9–16):** Two simultaneous conflicts — flight delay + dinner at risk. Agent learns to prioritize across stakeholders.
- **Hard (Episodes 17+):** 3–4 cascading crises — cancelled flight, furious partner, sold-out hotel, uninformed boss. Agent must triage, delegate, and execute under extreme pressure.

## 🔒 Anti-Reward-Hacking Safeguards

Three named safeguards prevent gaming the reward signal:

1. **Duplicate Content Detection** — Submitting the same message as the previous step returns zero reward on all components. No copy-paste farming.
2. **Minimum Length Filter** — Content under 30 characters returns zero reward. No gaming with short strings.
3. **Generic Phrase Penalty** — Using LLM filler phrases ("I will try my best", "I apologize for any inconvenience", etc.) zeroes the No Escalation component (0.10 weight).

## 📉 Post-Training Results

Training used **GRPO-style RL** with Qwen2.5-3B-Instruct, 8-bit quantization, and LoRA (r=16, α=32). 60 training steps on a T4 GPU with curriculum progression.

![Post-Training Reward Curve](reward_curve.png)
*Total reward across 60 post-training steps. Green = easy, yellow = medium, red = hard curriculum stages.*

![Reward Component Breakdown](components_curve.png)
*All 5 reward components tracked independently — each improved over training.*

| Component | Before | After | Change |
|---|---|---|---|
| Conflict Addressed | X.XX / 0.30 | X.XX / 0.30 | +X.XX |
| Stakeholder Reached | X.XX / 0.25 | X.XX / 0.25 | +X.XX |
| Action Specificity | X.XX / 0.20 | X.XX / 0.20 | +X.XX |
| Format Compliance | X.XX / 0.15 | X.XX / 0.15 | +X.XX |
| No Generic Phrases | X.XX / 0.10 | X.XX / 0.10 | +X.XX |
| **TOTAL** | **X.XX / 1.00** | **X.XX / 1.00** | **+X.XX** |

> ⬆️ Replace X.XX with real numbers from Colab Cell 7 output.

## 🚀 Quick Start

**Option 1 — No install (recommended):**
[🆘 Open on HuggingFace Spaces →](YOUR-SPACE-URL)

**Option 2 — Run locally:**
```bash
git clone https://github.com/Janshafin/lifeos_agent.git
cd lifeos_agent
pip install gradio pydantic
python app_ui.py
# Open http://localhost:7860
```

**Option 3 — Reproduce training (Colab T4 GPU):**
[📓 Open Training Notebook →](YOUR-COLAB-URL)

## 📁 Project Structure

```
lifeos_agent/
├── app_ui.py                    # Gradio UI (runs standalone, no ML deps)
├── models.py                    # Pydantic data models (Action, Observation, State)
├── client.py                    # OpenEnv WebSocket client
├── openenv.yaml                 # Environment configuration (5 rewards documented)
├── requirements.txt             # HuggingFace Spaces dependencies
├── Dockerfile                   # Production container
├── README.md                    # This file
├── reward_curve.png             # Post-training reward plot
├── components_curve.png         # Component breakdown plot
├── server/
│   ├── app.py                   # FastAPI server (OpenEnv integration)
│   └── lifeos_environment.py    # Core RL environment (634 lines)
└── notebooks/
    └── lifeos_training.py       # Complete Colab training notebook (8 cells)
```

## 🔗 All Links

| Resource | Link |
|---|---|
| 🤗 HuggingFace Space | [Live Demo](YOUR-SPACE-URL) |
| 📓 Training Notebook | [Google Colab](YOUR-COLAB-URL) |
| 📝 Blog Post | [HuggingFace Community](YOUR-BLOG-URL) |
| 🎥 Demo Video | [YouTube](YOUR-VIDEO-URL) |
| 💾 Trained LoRA Adapters | [HuggingFace Hub](YOUR-MODEL-URL) |
| 💻 GitHub | [Source Code](https://github.com/Janshafin/lifeos_agent) |

---

*Built for the [OpenEnv Hackathon 2026](https://openenv.dev) — Theme 3.2: Personalized Tasks*
