<div align="center">

# 🧠 LifeOS Agent

### Teaching LLMs to handle real life crises — not just answer questions about them

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue?style=for-the-badge)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-BSD--3-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge)](https://docker.com)

</div>

---

**It's 6:47pm on a Tuesday.** Your flight just got cancelled. Your partner is sitting alone at a restaurant across town — you were supposed to be there twenty minutes ago. You have a 9am meeting tomorrow in another city, and your boss doesn't know you might miss it. Every hotel near the venue is sold out. Your phone buzzes: *"Where are you? They're about to give away our table."*

What do you do first? Who do you contact? What do you say — *exactly*?

**Current LLMs give you a bullet-point list.** LifeOS Agent trains them to actually *handle* it.

---

## 🔍 The Problem

Ask any LLM for help with a personal crisis and you'll get: *"I understand this is stressful. Here are some steps you might consider..."* Generic. Passive. Useless in the moment. Today's LLMs **describe** solutions — they don't **execute** them. They can't prioritize competing stakeholders, craft time-sensitive messages, or make hard trade-offs between your boss, your partner, and an airline agent simultaneously.

LifeOS Agent is an **OpenEnv reinforcement learning environment** that trains language models to go from *"here are some suggestions"* to *"I've drafted a message to your partner, rebooked your flight to the red-eye, and emailed your boss a backup plan — which should I send first?"*

---

## 🚀 What Makes LifeOS Agent Different

* **Curriculum Learning** — The agent doesn't start with impossible scenarios. It earns its way up from 1-conflict easy scenarios to 4-conflict hard scenarios, building competence incrementally across three difficulty tiers.

* **5 Independent Reward Functions** — Unlike single-score rewards that invite hacking, LifeOS decomposes reward into five interpretable components. You can see exactly where the agent improves and where it games the system.

* **Anti-Reward-Hacking Safeguards** — Duplicate detection, minimum content length, and generic phrase penalties prevent the agent from finding cheap shortcuts.

---

## ⚙️ How It Works

**Observe.** The agent receives a crisis scenario with active conflicts, persona descriptions, and time pressure. A flight cancellation might present four simultaneous conflicts — each with a different persona who responds differently to your actions.

**Act.** The agent selects an action type (`send_message`, `reschedule`, `delegate`, etc.), targets a specific person, crafts the actual message content, explains its reasoning, and declares urgency. This isn't multiple-choice — the agent must *generate* real communication.

**Learn.** Five reward functions score the action independently:

| Component             | Weight   | What It Checks                                                |
| --------------------- | -------- | ------------------------------------------------------------- |
| `conflict_addressed`  | **0.30** | Does the action reference an actual active conflict?          |
| `stakeholder_reached` | **0.25** | Is the target person a real persona in the scenario?          |
| `action_specificity`  | **0.20** | Does the message contain a time reference AND an action verb? |
| `format_compliance`   | **0.15** | Is reasoning substantive (>40 chars) with valid urgency?      |
| `no_escalation`       | **0.10** | Are generic filler phrases absent?                            |

---

## 📈 Results

![Training Progress](reward_curve.png)

| Metric              | Before Training | After Training | Change   |
| ------------------- | --------------- | -------------- | -------- |
| **Total Reward**    | `0.000`         | `0.800`        | `+0.800` |
| Conflict Addressed  | `0.000`         | `0.300`        | `+0.300` |
| Stakeholder Reached | `0.000`         | `0.050`        | `+0.050` |
| Action Specificity  | `0.000`         | `0.200`        | `+0.200` |
| Format Compliance   | `0.000`         | `0.150`        | `+0.150` |
| No Escalation       | `0.000`         | `0.100`        | `+0.100` |

After training, LifeOS Agent improved from a total reward of **0.000** to **0.800**, showing measurable gains in prioritization, messaging quality, and crisis response behavior.

### Reward Hacking Check

* Same content repeated: `False`
* Content length: `357 chars`
* Generic phrase found: `False`

---

## 🛡️ Anti-Reward-Hacking Safeguards

RL agents are creative optimizers — they *will* find shortcuts. LifeOS includes three safeguards:

* **Duplicate Detection** — If the agent repeats the exact same content as its previous step, all five reward components return **0.0**. No credit for copy-paste.

* **Minimum Content Length** — Responses under 30 characters are automatically scored at **0.0** across all components. No gaming through minimal output.

* **Generic Phrase Penalty** — Phrases like *"I will try my best"*, *"I apologize for any inconvenience"*, and *"I will get back to you"* trigger the `no_escalation` component to return **0.0**. The agent must produce specific, actionable responses.

---

## 🏃 Quick Start

### Install & Run

```bash
pip install "git+https://github.com/meta-pytorch/OpenEnv.git"
openenv init lifeos_agent
openenv build
docker run -p 8001:8000 openenv-lifeos-agent:latest
```

### Validate

```bash
openenv validate --verbose --url http://localhost:8001
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

Open `notebooks/lifeos_training.py` and paste each `# %%` cell into a new Colab notebook.

---

## 📁 Project Structure

```text
lifeos_agent/
├── models.py
├── client.py
├── openenv.yaml
├── server/
│   ├── app.py
│   ├── lifeos_environment.py
│   └── Dockerfile
└── notebooks/
    └── lifeos_training.py
```

---

## 🔗 Links

| Resource             | Link                             |
| -------------------- | -------------------------------- |
| 🤗 HuggingFace Space | [heyjan/lifeos-agent](https://huggingface.co/spaces/heyjan/lifeos-agent) |
| 📓 Colab Notebook    | Add your Colab notebook URL here |
| 📝 Blog Post         | Add your blog post URL here      |
| 🎥 Demo Video        | Add your video URL here          |

---

## 📄 License

BSD-3-Clause — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the OpenEnv Hackathon 2026** · *Because your AI assistant should do more than apologize for the inconvenience.*

</div>
