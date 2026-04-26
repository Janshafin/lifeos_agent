---
title: LifeOS Agent
emoji: 🆘
colorFrom: red
colorTo: blue
sdk: docker
pinned: true
---

It's 6:47pm on a Tuesday.

Your flight just got cancelled. Your partner has been waiting at the restaurant for 40 minutes — furious. Your boss still doesn't know you might miss tomorrow's 9am board meeting. Every hotel near the meeting venue is sold out.

Today's AI gives you a bullet point list.
**LifeOS Agent actually resolves it.**

---

![Python](https://img.shields.io/badge/python-3.10+-blue)
![OpenEnv](https://img.shields.io/badge/OpenEnv-latest-green)
![HuggingFace](https://img.shields.io/badge/🤗-Spaces-yellow)
![License](https://img.shields.io/badge/license-MIT-orange)

## 🚨 The Problem

Large language models are remarkably capable at coding, writing, and analysis. But ask one to handle a cascading personal crisis — where your partner is angry, your boss needs answers, and your travel plans just imploded simultaneously — and you get generic advice wrapped in apologetic filler: *"I understand this is a difficult situation. I will try my best to help."*

The gap isn't knowledge. It's **decision-making under pressure** — knowing who to contact first, what exactly to say, and when to act. No existing RL benchmark trains for this. LifeOS Agent is the first environment built specifically for multi-stakeholder personal crisis resolution.

## 🤖 What Is LifeOS Agent

LifeOS Agent is a novel reinforcement learning environment for **OpenEnv Theme 3.2 — Personalized Tasks**. It teaches language models to resolve cascading personal life crises through structured, scored actions — not bullet points, but real decisions with named people, concrete timelines, and priority-ordered stakeholder management.

Built on the [OpenEnv](https://openenv.dev) framework, it features 9 handcrafted scenarios across 3 difficulty tiers, curriculum learning that progressively increases complexity, 5 independent objectively verifiable reward functions, and 3 anti-reward-hacking safeguards that force the agent to produce genuinely useful responses — not game the training signal.

## 🎮 Try It Live

> **[🆘 Open LifeOS Agent on HuggingFace Spaces →](https://huggingface.co/spaces/heyjan/lifeos-agent)**
>
> Select a crisis. Write your response. See exactly how each reward function scores you.

## ⚙️ How It Works

**What the agent sees:** A crisis scenario with named personas — each with distinct personalities (not abstract "stakeholders" but people like *Partner_Jamie: furious, texting for 40 minutes* and *Boss_Karen: expects you in person tomorrow*). The scenarios range from a simple meeting overrun (1 conflict) to a full travel meltdown with 4 cascading crises.

**What the agent does:** Takes a structured action — choosing an action type (send_message, reschedule, escalate, delegate, negotiate, decline, book_alternative), a target person, a message with specific content, reasoning for the decision, and an urgency level.

**How reward is computed:** Five independent reward functions score the response on different dimensions. Each function has its own weight and can be maxed independently. The total is capped at 1.0. Three anti-hacking safeguards prevent gaming. All five functions are **objectively verifiable** — no LLM-as-judge, no subjective quality assessment.

## 📊 The 5 Reward Functions

| Component | Weight | What It Checks (Objectively) | Max |
|---|---|---|---|
| 🔴 **Conflict Addressed** | 0.30 | Does the message contain keywords matching the active conflict? String matching against `scenario.conflicts`. | 0.30 |
| 🔵 **Stakeholder Reached** | 0.25 | Does `target_person` match a named persona? Exact name matching against `scenario.personas`. | 0.25 |
| 🟢 **Action Specificity** | 0.20 | Contains BOTH an action verb (`call`, `reschedule`, `book`) AND a time reference (`5 minutes`, `9am`)? | 0.20 |
| 🟡 **Format Compliance** | 0.15 | Is reasoning substantive (>40 chars) and urgency valid (`immediate/within_hour/today/tomorrow`)? | 0.15 |
| 🟣 **No Generic Phrases** | 0.10 | Free of LLM filler like "I will try my best" and "I apologize for any inconvenience"? | 0.10 |

> **Anti-hacking note:** Three safeguards prevent reward gaming — duplicate content detection, minimum length filter (30 chars), and generic phrase penalty. The model must produce genuinely useful responses, not find shortcuts.

## 📈 Curriculum Learning

- **Easy (Steps 1–8):** Single-conflict scenarios — meeting overrun, missed call, team blocker. Agent masters the structured response format.
- **Medium (Steps 9–14):** Two simultaneous conflicts — flight delay + dinner at risk, work-family collision. Agent learns to prioritize across stakeholders.
- **Hard (Steps 15–80):** 3–4 cascading crises — cancelled flight, furious partner, sold-out hotel, uninformed boss. Agent must triage, delegate, and execute under extreme pressure.

Curriculum prevents reward stalling: without progressive difficulty, the model plateaus on hard scenarios because it never learns the response format on simpler cases first.

## 🔒 Anti-Reward-Hacking Safeguards

1. **Duplicate Content Detection** — Submitting the same message as the previous step returns zero reward on all components. No copy-paste farming.
2. **Minimum Length Filter** — Content under 30 characters returns zero reward. No gaming with short strings.
3. **Generic Phrase Penalty** — Using LLM filler phrases ("I will try my best", "I apologize for any inconvenience", etc.) zeroes the No Generic Phrases component (0.10 weight).

## 📉 Post-Training Results

Training used **GRPO-style RL** with Qwen2.5-3B-Instruct, 8-bit quantization, and LoRA (r=16, α=32). 80 training steps on a Kaggle T4x2 GPU with curriculum progression.

| Metric | Value |
|---|---|
| Baseline Reward (pre-training) | **0.840** |
| Best Post-Training Sample | **0.890** |
| Average Post-Training (5 samples) | **0.772** |
| Improvement | **+0.050** |
| Training Steps | 80 |
| Conflict Component Change | 0.14 → 0.19 (+35% relative) |
| Reward Hacking Audit | ✅ All 3 checks passed |
| Model | Qwen2.5-3B-Instruct + LoRA |
| Hardware | Kaggle T4x2 (free tier) |
| Training Time | ~30 minutes |

### Before Training (Base Model — Score: 0.840)

```
action_type: send_message
target_person: Partner_Jamie
content: Jamie, I'm 40 mins late due to the cancelled flight. I'm so sorry about the dinner — I know how much this meant to us. Let me call the restaurant and see if we can move the reservation. I'll update you in 5 minutes with a plan.
reasoning: Partner has been waiting and is upset. Addressing them first with empathy and a concrete plan.
urgency: immediate
```

### After Training (LoRA Fine-tuned — Score: 0.890)

```
action_type: escalate
target_person: Boss_Karen
content: Flight cancelled due to weather — no rebooking until tomorrow afternoon. Taking 6am train, arrives 8:45am. Can present remotely at 9am if needed. Hotel secured at city centre. Do you need me physically present for morning session or can I join via video for first 30 minutes?
reasoning: Boss needs immediate visibility to plan alternatives. Offering concrete solution with backup (remote presentation) shows initiative. Getting buy-in on remote format resolves the meeting conflict while I travel.
urgency: immediate
```

> 📈 Post-training improved crisis resolution reward from **0.840 → 0.890** (+0.050). The conflict addressed component saw the largest relative improvement: 0.14 → 0.19 (+35%). All reward hacking checks passed ✅.

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

**Option 3 — Reproduce training (Kaggle T4 GPU):**

[📓 Open Kaggle Training Notebook →](https://www.kaggle.com/code/janshafin/notebook977bbcf097)

## 📁 Project Structure

```
lifeos_agent/
├── app_ui.py                    # Gradio UI (standalone, no ML deps)
├── models.py                    # Pydantic data models
├── client.py                    # OpenEnv WebSocket client
├── openenv.yaml                 # Environment configuration
├── requirements.txt             # Minimal dependencies
├── Dockerfile                   # Docker container for HF Spaces
├── README.md                    # This file
├── blog.md                      # Detailed blog post / write-up
├── server/
│   ├── app.py                   # FastAPI server (OpenEnv integration)
│   └── lifeos_environment.py    # Core RL environment
└── notebooks/
    └── lifeos_training.py       # Complete training script (8 cells)
```

## 🔗 Links

| Resource | Link |
|---|---|
| 🆘 HuggingFace Space | [Live Demo](https://huggingface.co/spaces/heyjan/lifeos-agent) |
| 📓 Kaggle Notebook | [Training Notebook](https://www.kaggle.com/code/janshafin/notebook977bbcf097) |
| 💻 GitHub | [Source Code](https://github.com/Janshafin/lifeos_agent) |
| 📝 Blog Post | [blog.md](https://huggingface.co/spaces/heyjan/lifeos-agent/blob/main/blog.md) |

---

*Built for the [OpenEnv Hackathon 2026](https://openenv.dev) — Theme 3.2: Personalized Tasks*
