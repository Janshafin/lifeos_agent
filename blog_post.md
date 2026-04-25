# LifeOS Agent: Teaching LLMs to Handle Real Life Crises

**It's 6:47pm on a Tuesday.** Your flight just got cancelled.

Your partner is sitting alone at a restaurant across town — you were supposed to be there twenty minutes ago. You have a 9am meeting tomorrow in another city, and your boss is expecting you in person. You open your hotel app: *sold out, sold out, sold out.* Your phone buzzes. It's your partner: *"Where are you? They're about to give away our table."*

You need to rebook a flight, message your partner something that doesn't start a fight, find a hotel, and email your boss a backup plan — all in the next fifteen minutes. You don't need "suggestions." You need someone to *handle* it.

This is the problem we built LifeOS Agent to solve.

## Why LLMs fail at this today

Ask GPT-4 or Claude to help you with this exact situation and you'll get a well-structured, empathetic, completely unhelpful response. Something like: *"I understand this must be very stressful. Here are some steps you might consider..."* followed by a generic numbered list.

The failure mode is predictable: current LLMs **describe** solutions instead of **executing** them. They can't prioritize between your angry partner and your strict boss. They won't draft the actual message to send. They treat every stakeholder with equal, vague politeness — which is exactly the wrong move when you have four competing crises and ten minutes.

Real crisis management requires trade-offs, specificity, and the ability to craft different messages for different people with different personalities. It requires knowing that your partner needs reassurance *now* while your boss needs a professional email with a backup plan *within the hour*.

## What we built

LifeOS Agent is an OpenEnv reinforcement learning environment with **9 crisis scenarios** spanning three difficulty tiers. Easy scenarios have one conflict (a meeting overrun). Hard scenarios have four simultaneous conflicts (the flight cancellation nightmare above). The agent progresses through a **curriculum** — it earns its way from easy to hard, building competence incrementally.

The key innovation is our reward architecture. Instead of a single opaque score, we decompose reward into **five independent functions**, each targeting a specific capability:

- **Conflict Addressed (30%)** — Did you actually reference the problem, or just speak in generalities?
- **Stakeholder Reached (25%)** — Did you contact a real person in the scenario, or talk to nobody?
- **Action Specificity (20%)** — Did you include a concrete time and action verb ("reschedule to the 11pm red-eye"), or vague hand-waving?
- **Format Compliance (15%)** — Did you explain your reasoning substantively?
- **No Escalation (10%)** — Did you avoid the generic filler phrases that LLMs default to?

Each component is tracked independently, so we can see *exactly* where the model is improving — and where it might be gaming the system.

## Results

We trained Qwen2.5-3B (4-bit quantized via Unsloth) for 60 steps with curriculum learning. The reward curve tells the story: the agent starts by producing generic responses that score well only on format compliance. By the end of training, it's addressing specific conflicts, targeting real stakeholders, and including concrete time references in its messages.

The before/after comparison is striking. Given the flight cancellation scenario, the untrained model produces: *"I understand this is a difficult situation. I recommend you contact the airline and inform your partner."* After training: *"I need to contact the airline now to reschedule to the 11pm red-eye, then immediately message my partner with a specific arrival time to reduce their anxiety."*

That's the difference between an AI that *talks about* crisis management and one that *does* it.

## Why this matters

We're at an inflection point for personal AI assistants. The next generation won't just answer questions — they'll manage your life when things go sideways. But training them requires environments that capture the *messiness* of real human situations: competing priorities, emotional stakeholders, time pressure, and the need for diplomatic specificity.

LifeOS Agent is a step toward that future. It's open-source, runs anywhere Docker does, and integrates with the OpenEnv ecosystem for standardized RL training.

The crises are simulated. The skills are real.

---

*Try LifeOS Agent on [HuggingFace Spaces](https://huggingface.co/spaces/YOUR_SPACE) or explore the [source code on GitHub](https://github.com/Janshafin/lifeos_agent).*
