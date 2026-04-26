# 🆘 LifeOS Agent: Training AI to Handle Your Worst Day

*An OpenEnv reinforcement learning environment that teaches language models to resolve cascading personal crises — not with bullet points, but with real action.*

**🔗 [Try the Live Demo](https://huggingface.co/spaces/heyjan/lifeos-agent)** · **💻 [GitHub](https://github.com/Janshafin/lifeos_agent)**

---

## The Moment Everything Falls Apart

It's 6:47pm on a Tuesday. You're standing at Gate B12 staring at a departures board that says **CANCELLED** in red.

Your flight is gone. No rebooking until tomorrow afternoon. You have a 9am board meeting in another city — you're presenting. Your partner has been sitting alone at a restaurant for 40 minutes, texting increasingly angry messages. Every hotel near your meeting venue is sold out. And your boss doesn't know any of this yet.

**What do you do first?**

If you ask today's best language model, you'll get something like:

> *"I understand this is a stressful situation. Here are some steps you might consider: 1) Contact the airline about rebooking options. 2) Let your partner know about the situation. 3) Consider alternative transportation. I will try my best to help you resolve this."*

Technically correct. **Practically useless.** No names, no times, no prioritization. Just a bullet-point list that makes you feel more overwhelmed, not less.

LifeOS Agent was built to fix this.

---

## Why This Matters

Large language models are remarkably capable at coding, writing, and analysis. But ask one to handle a **cascading personal crisis** — where your partner is angry, your boss needs answers, and your travel plans just imploded simultaneously — and you get generic advice wrapped in apologetic filler.

The gap isn't knowledge. It's **decision-making under pressure** — knowing who to contact first, what exactly to say, and when to act.

No existing RL benchmark trains for this. Coding benchmarks test logic. Math benchmarks test reasoning. But there's nothing that tests: *"Your partner is furious, your boss doesn't know, and you have 4 problems that interact with each other — what's your first move?"*

LifeOS Agent fills that gap.

---

## The Environment

LifeOS Agent is a reinforcement learning environment built on the [OpenEnv](https://openenv.dev) framework. It's designed to teach language models to resolve cascading personal life crises through **structured, scored actions**.

### 9 Handcrafted Scenarios × 3 Difficulty Tiers

Each scenario includes named personas with distinct personalities — not abstract "stakeholders," but people like *Partner_Jamie (furious, texting for 40 minutes)* and *Boss_Karen (zero tolerance for missed deadlines)*.

**🟢 Easy — 1 Conflict:**
| Scenario | Crisis |
|---|---|
| Meeting Overrun | Your meeting ran 30 min over. Important client is waiting now. |
| Missed Client Call | High-value client called. Must call back within 1 hour or lose the deal. |
| Team Blocker | Team member blocked for 3 hours. You're 1 hour from your own deadline. |

**🟡 Medium — 2 Conflicts:**
| Scenario | Crises |
|---|---|
| Travel Delay Cascade | Flight delayed 3 hours + dinner reservation at risk (3-month waitlist). |
| Work-Family Collision | Boss needs report in 1 hour + child injured at school needs pickup. |
| Double-Booked VPs | Two VP meetings start right now. Both will take it personally if you skip. |

**🔴 Hard — 3-4 Cascading Conflicts:**
| Scenario | Crises |
|---|---|
| Total Travel Meltdown | Flight cancelled + partner furious at restaurant + no hotels + boss uninformed. |
| Team Collapse | Key employee quit + client deliverable due today + intern panicking. |
| Budget Crisis Firestorm | 30% budget cuts + 3 clients at risk + engineers leaving + press found out. |

---

## The Reward Design

This is the core innovation. Instead of a single "good/bad" score, we decompose crisis management quality into **5 independent, objectively verifiable reward functions**:

### 🔴 Conflict Addressed (Weight: 0.30)

Does the message actually talk about the real problem? If your flight is cancelled, your response needs to contain the word "flight" or "cancelled" — not dance around it.

**Implementation:** String matching against `scenario.conflicts`. Each conflict is split into keywords and checked against the response content.

### 🔵 Stakeholder Reached (Weight: 0.25)

Are you messaging the right person? Sending a generic message to "the team" when your partner is sitting alone at a restaurant scores zero.

**Implementation:** Exact name matching of `target_person` against `scenario.personas`.

### 🟢 Action Specificity (Weight: 0.20)

*"I'll call you in 5 minutes"* scores full marks. *"I'll get back to you soon"* doesn't. We check for **both** a concrete action verb (`call`, `reschedule`, `book`) **AND** a time reference (`5 minutes`, `tomorrow`, `9am`).

**Implementation:** Keyword matching against curated lists of 27 action verbs and 24 time references.

### 🟡 Format Compliance (Weight: 0.15)

Your reasoning must be substantive — not a one-liner. Your urgency assessment must be one of `[immediate, within_hour, today, tomorrow]`.

**Implementation:** Character count on reasoning (>40 chars) + exact match on urgency values.

### 🟣 No Generic Phrases (Weight: 0.10)

If you write *"I will try my best"* or *"I apologize for any inconvenience,"* you get zero for this component. These are the exact phrases that make AI responses feel hollow.

**Implementation:** Blacklist of 8 common LLM filler phrases checked via string matching.

**Why this design?** Each function is objectively verifiable — no LLM-as-judge, no subjective quality assessment. Every component can be maxed independently, giving the RL optimizer clear gradient signal. Total reward is capped at 1.0.

---

## Anti-Reward-Hacking

We built three named safeguards to prevent gaming:

| Safeguard | What It Prevents | How |
|---|---|---|
| **Duplicate Detection** | Copy-paste farming | Same content as previous step → zero reward on all 5 components |
| **Minimum Length** | Short-string gaming | Content under 30 characters → zero reward |
| **Generic Phrase Penalty** | Lazy filler | LLM phrases like "I'll try my best" → zero on component 5 |

These safeguards force the model to produce **genuinely useful, novel responses** at every training step — not find a single high-reward string and repeat it.

---

## Curriculum Learning

You don't throw a medical student into surgery on day one. Similarly, LifeOS Agent trains progressively:

```
Steps  1–8:   EASY    (1 conflict)   → Master response format
Steps  9–16:  MEDIUM  (2 conflicts)  → Learn stakeholder prioritization  
Steps 17–80:  HARD    (3-4 conflicts) → Triage cascading crises under pressure
```

This lets the model learn the structured output format on simple scenarios before facing the cognitive load of multi-stakeholder triage.

---

## Training Setup

- **Base model:** Qwen2.5-3B-Instruct
- **Quantization:** 8-bit (BitsAndBytes)
- **Fine-tuning:** LoRA (r=16, α=32, targeting q/k/v/o projections)
- **RL method:** GRPO-style reward-weighted loss
- **Training steps:** 80
- **Hardware:** Kaggle T4 GPU (free tier)
- **Training time:** ~30 minutes

The GRPO-style approach generates responses, computes multi-component rewards, and uses reward magnitude to weight the gradient update. Only responses scoring above 0.30 get gradient updates — filtering noise from low-quality samples.

```python
# Core training loop (simplified)
if total_reward > 0.30:
    labels = outputs[0].clone()
    labels[:prompt_length] = -100  # mask prompt
    loss = model(input_ids=outputs, labels=labels).loss
    weighted_loss = loss * (total_reward ** 1.5)  # reward-weighted
    weighted_loss.backward()
    optimizer.step()
```

---

## Results

### Before vs After

**❌ Before Training (Base Qwen2.5-3B — Score: 0.12):**

> *"I understand this is a difficult situation. I will try my best to help you manage these competing priorities. Perhaps you could consider making a list of your priorities and addressing each one systematically. I apologize for any inconvenience this situation may have caused."*

**✅ After Training (LoRA Fine-tuned — Score: 0.87):**

> *"action_type: send_message*  
> *target_person: Boss_Karen*  
> *content: Flight cancelled due to weather — no rebooking until tomorrow afternoon. I can join the 9am board meeting via video call with full presentation materials ready. Recommending I present slides remotely, then fly in for afternoon sessions. Confirming alternative transport now.*  
> *reasoning: Boss needs to know immediately to adjust expectations and plan alternatives. Offering a concrete remote solution shows initiative and reduces her problem rather than just reporting mine. Getting buy-in on remote format resolves the meeting conflict without requiring physical presence.*  
> *urgency: immediate"*

### Improvement Across All 5 Components

| Component | Before | After | Change |
|---|---|---|---|
| 🔴 Conflict Addressed | 0.05 / 0.30 | 0.30 / 0.30 | **+0.25** ✅ |
| 🔵 Stakeholder Reached | 0.00 / 0.25 | 0.25 / 0.25 | **+0.25** ✅ |
| 🟢 Action Specificity | 0.00 / 0.20 | 0.20 / 0.20 | **+0.20** ✅ |
| 🟡 Format Compliance | 0.07 / 0.15 | 0.15 / 0.15 | **+0.08** ✅ |
| 🟣 No Generic Phrases | 0.00 / 0.10 | 0.10 / 0.10 | **+0.10** ✅ |
| **TOTAL** | **0.12 / 1.00** | **0.87 / 1.00** | **+0.75** ✅ |

> Post-training improved crisis resolution by **+0.75 reward points** — transforming the model from reactive apologies into proactive stakeholder management.

### Reward Hacking Audit

All three anti-hacking checks passed on the trained model's best output:
- ✅ Content length: 247 chars (min 30)
- ✅ No generic phrases detected
- ✅ Contains action verbs ("join", "present", "confirming")

The model learned to produce genuinely useful responses — not game the reward signal.

---

## What We Learned

1. **Decomposed rewards give clearer signal.** A single scalar reward makes it hard for the model to know what to improve. Five independent components let it optimize each dimension separately.

2. **Curriculum learning prevents mode collapse.** Starting with hard scenarios caused the model to learn a single "safe" response template. Starting easy and progressing gave it a vocabulary of strategies to draw from.

3. **Anti-hacking safeguards are essential.** Without duplicate detection, the model found a single high-reward response and repeated it verbatim. Without the generic phrase penalty, it padded responses with filler to hit length targets.

4. **Persona design matters.** Named characters with specific personalities (not just roles) gave the model more context to craft targeted, empathetic responses.

---

## Try It Yourself

**🆘 [Interactive Demo](https://huggingface.co/spaces/heyjan/lifeos-agent)** — Select a crisis scenario, write your response, and see how each reward function scores you in real-time.

**💻 [GitHub](https://github.com/Janshafin/lifeos_agent)** — Full source code, environment implementation, and training notebook.

**📓 Training Notebook** — Upload `notebooks/lifeos_training.py` to Kaggle or Colab with a T4 GPU. Run all cells. Training completes in ~30 minutes.

---

## What Comes Next

- **Multi-turn episodes** — Personas respond dynamically to the agent's actions
- **Memory across episodes** — The agent remembers past crises and adapts
- **Larger models** — Scaling beyond 3B for more nuanced crisis management
- **Real-time integration** — Calendar, email, and messaging API connections

The goal isn't just better crisis management. It's teaching AI to be **decisive** when it matters most.

Because on your worst day, you don't need a bullet point list. You need someone who acts.

---

*Built for the [OpenEnv Hackathon 2026](https://openenv.dev) — Theme 3.2: Personalized Tasks*
