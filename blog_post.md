# LifeOS Agent: Training AI to Handle Your Worst Day

*An OpenEnv reinforcement learning environment that teaches language models to resolve cascading personal crises — not with bullet points, but with real action.*

---

## The Moment Everything Falls Apart

It's 6:47pm on a Tuesday. You're standing at Gate B12 staring at a departures board that says CANCELLED in red.

Your flight is gone. No rebooking until tomorrow afternoon. You have a 9am board meeting in another city — you're presenting. Your partner has been sitting alone at a restaurant across town for 40 minutes, texting increasingly angry messages. Every hotel near your meeting venue is sold out. And your boss doesn't know any of this yet.

What do you do first?

If you ask a large language model, you'll get something like: *"I understand this is a stressful situation. Here are some steps you might consider: 1) Contact the airline about rebooking options. 2) Let your partner know about the situation. 3) Consider alternative transportation. I will try my best to help you resolve this."*

Technically correct. Practically useless. No names, no times, no prioritization. Just a list that makes you feel more overwhelmed, not less.

**LifeOS Agent was built to fix this.**

## What Makes This Different

LifeOS Agent is not an assistant. It's a training environment — built on the OpenEnv framework — that teaches language models to actually *resolve* crises, not just describe them.

The insight is simple: real crisis management isn't about knowing what to do. It's about knowing what to do *first*, *who* to tell, *what exactly* to say, and *when* to do it. Current LLMs are terrible at this because they're trained to be helpful in general, not decisive under pressure.

We created 9 handcrafted crisis scenarios across three difficulty tiers. Easy scenarios have one conflict. Medium scenarios have two. Hard scenarios have three or four cascading crises that interact with each other — fixing one might break another.

Each scenario includes named personas with distinct personalities. Your partner who's been waiting for 40 minutes isn't just "upset" — she's *furious*, she's been texting for 40 minutes, and she's considering leaving. Your boss isn't just "expecting a report" — she has zero tolerance for missed deadlines and will question your reliability.

## The Reward Design

Instead of a single "good/bad" score, we decompose crisis management quality into five independent reward functions:

**Conflict Addressed (0.30):** Did you actually talk about the real problem? If your flight is cancelled, your message needs to say "flight" or "cancelled" — not dance around it.

**Stakeholder Reached (0.25):** Are you messaging the right person? Sending a message to "the team" when your partner is sitting alone at a restaurant scores zero.

**Action Specificity (0.20):** "I'll call you in 5 minutes" scores full marks. "I'll get back to you soon" doesn't. We check for concrete action verbs AND time references.

**Format Compliance (0.15):** Your reasoning must be substantive — not a one-liner. Your urgency assessment must match the valid options.

**No Generic Phrases (0.10):** If you write "I will try my best" or "I apologize for any inconvenience," you get zero for this component. These are the exact phrases that make AI responses feel hollow.

## Curriculum Learning

You don't throw a medical student into surgery on day one. Similarly, our agent starts with easy single-conflict scenarios, graduates to medium two-conflict scenarios after 8 episodes, and finally faces the hardest cascading crises after 16 episodes.

This progressive difficulty lets the model master the response format and basic crisis communication before tackling multi-stakeholder triage.

## Results

After 60 training steps with LoRA fine-tuning on Qwen2.5-3B-Instruct (8-bit quantized), the model's responses transform from generic listicles into specific, actionable crisis management.

The untrained model scores ~0.17. The trained model consistently achieves 0.8+.

More importantly, the *quality* of responses changes completely. The trained model names specific people, proposes concrete timelines, addresses multiple conflicts in priority order, and avoids every generic filler phrase we penalize.

## Try It Yourself

**[Interactive Demo →](https://huggingface.co/spaces/heyjan/lifeos-agent)** — Select a crisis, write your response, see how each reward function scores you.

**[Training Notebook →](#)** — Train the model yourself on a free Colab T4 GPU in 30 minutes.

**[GitHub →](https://github.com/Janshafin/lifeos_agent)** — Full source code, environment, and documentation.

## What Comes Next

We're exploring multi-turn episodes where personas respond dynamically, memory across episodes so the agent learns from past crises, and scaling to larger models. The goal isn't just better crisis management — it's teaching AI to be *decisive* when it matters most.

Because on your worst day, you don't need a bullet point list. You need someone who acts.

---

*Built for the OpenEnv Hackathon 2026 by the LifeOS Team.*
