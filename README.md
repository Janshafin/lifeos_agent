<div align="center">

# 🧠 LifeOS Agent

### Teaching LLMs to handle real life crises — not just answer questions about them

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue?style=for-the-badge)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-BSD--3-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-yellow?style=for-the-badge)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge)](https://docker.com)

</div>

---

**It's 6:47pm on a Tuesday.** Your flight just got cancelled. Your partner is sitting alone at a restaurant across town. You have a 9am meeting tomorrow in another city — your boss doesn't know you might miss it. Every hotel near the venue is sold out. Your phone buzzes: *"Where are you?"*

What do you do first? Who do you contact? What do you say — *exactly*?

**Current LLMs give you a bullet-point list.** LifeOS Agent trains them to actually *handle* it.

---

## 🔍 The Problem

Ask any LLM for help with a personal crisis and you'll get:

> *"I understand this is stressful. Here are some steps you might consider..."*

Generic. Passive. Useless in the moment. Today's LLMs **describe** solutions — they don't **execute** them. They can't prioritize competing stakeholders, craft time-sensitive messages, or make hard trade-offs between your boss, your partner, and an airline agent simultaneously.

LifeOS Agent is an **OpenEnv reinforcement learning environment** that trains language models to go from *"here are some suggestions"* to *"I've drafted a message to your partner, rebooked your flight to the red-eye, and emailed your boss a backup plan — which should I send first?"*

---

## 🚀 What Makes LifeOS Different

### Curriculum Learning
The agent doesn't start with impossible scenarios. It **earns its way up**:

| Stage | Conflicts | Example |
|-------|-----------|---------|
| 🟢 **Easy** | 1 | Meeting overrun — who do you message first? |
| 🟡 **Medium** | 2 | Flight delayed + dinner reservation at risk |
| 🔴 **Hard** | 3–4 | Flight cancelled + partner waiting + hotel sold out + boss unaware |

### 5 Independent Reward Functions

Unlike single-score rewards that invite hacking, LifeOS decomposes reward into **five interpretable components**:

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| `conflict_addressed` | **30%** | Does the action reference an actual active conflict? |
| `stakeholder_reached` | **25%** | Is the target person a real persona in the scenario? |
| `action_specificity` | **20%** | Does the message contain a time reference AND an action verb? |
| `format_compliance` | **15%** | Is reasoning substantive (>30 chars) with valid urgency? |
| `no_escalation` | **10%** | Are generic filler phrases absent? |

Each component is computed **independently** — you can see exactly where the agent is improving and where it's gaming the system.

---

## ⚙️ How It Works

**Observe.** The agent receives a crisis scenario with active conflicts, persona descriptions, and time pressure. A flight cancellation might present four simultaneous conflicts: rebooking travel, notifying your partner, finding a hotel, and informing your boss — each with a different persona who responds differently to your actions.

**Act.** The agent selects an action type (send_message, reschedule, delegate, etc.), targets a specific person, crafts the actual message content, explains its reasoning, and declares urgency. This isn't multiple-choice — the agent must *generate* real communication.

**Learn.** Five reward functions score the action independently. The agent learns that mentioning "reschedule to the 11pm red-eye" scores higher than "I'll look into travel options" — because specificity and conflict-addressing are rewarded separately from format compliance.

---

## 📈 Results

<!-- Replace with your actual reward_curve.png after training -->
![Training Progress](reward_curve.png)

| Metric | Before Training | After Training | Change |
|--------|----------------|----------------|--------|
| **Total Reward** | `YOUR_START` | `YOUR_END` | `+YOUR_IMPROVEMENT` |
| Conflict Addressed | — | — | — |
| Stakeholder Reached | — | — | — |
| Action Specificity | — | — | — |
| Format Compliance | — | — | — |
| No Escalation | — | — | — |

> **Fill in your exact numbers from Cell 7 of the training notebook after running it.**

---

## 🛡️ Anti-Reward-Hacking Safeguards

RL agents are creative optimizers — they *will* find shortcuts. LifeOS includes three safeguards:

1. **Duplicate Detection** — If the agent repeats the exact same content as its previous step, all five reward components return **0.0**. No credit for copy-paste.

2. **Generic Phrase Penalty** — Phrases like *"I will try my best"* and *"I apologize for any inconvenience"* trigger the `no_escalation` component to return **0.0**. The agent must produce specific, actionable responses.

3. **Minimum Content Length** — Responses under 30 characters are automatically scored at **0.0** across all components. No gaming through minimal output.

---

## 🏃 Quick Start

### Docker (recommended)

```bash
# Build the environment
openenv build

# Run on port 8001
docker run -p 8001:8000 openenv-lifeos-agent:latest

# Validate (in another terminal)
openenv validate --verbose --url http://localhost:8001
```

### Local Development (faster iteration)

```bash
# Install dependencies
pip install uv
cd /path/to/lifeos_agent

# Run directly
uv run --project . server --port 8001
```

### Python Client

```python
from client import create_env
from models import LifeOSAction

env = create_env("http://localhost:8001").sync()
with env:
    result = env.reset(seed=42)
    print(result.observation.scenario_description)

    action = LifeOSAction(
        action_type="send_message",
        target_person="Partner_Jamie",
        content="My flight was cancelled. I'm rebooking the 11pm red-eye now. I'll be at the restaurant by 7:30 — please order for us.",
        reasoning="Partner is waiting and worried. Immediate, specific communication reduces anxiety and shows I have a plan.",
        urgency="immediate",
    )
    result = env.step(action)
    print(f"Reward: {result.reward}")
```

### Train in Colab

Open `notebooks/lifeos_training.py` and paste each cell into a new Colab notebook, or:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

---

## 📁 Project Structure

```
lifeos_agent/
├── models.py                         # Pydantic models: Action, Observation, State
├── client.py                         # OpenEnv WebSocket client
├── openenv.yaml                      # Environment metadata
├── server/
│   ├── app.py                        # FastAPI application
│   ├── lifeos_environment.py         # Core RL environment (9 scenarios, 5 rewards)
│   └── Dockerfile                    # Production container
└── notebooks/
    └── lifeos_training.py            # Colab training notebook (8 cells)
```

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 🤗 HuggingFace Space | *Coming soon* |
| 📓 Colab Notebook | *Coming soon* |
| 📝 Blog Post | *Coming soon* |
| 🎥 Demo Video | *Coming soon* |

---

## 📄 License

BSD-3-Clause — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the OpenEnv Hackathon** · *Because your AI assistant should do more than apologize for the inconvenience.*

</div>
