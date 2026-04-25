# LifeOS Agent: Teaching LLMs to Handle Real Life Crises

**It's 6:47pm on a Tuesday.** Your flight just got cancelled.

Your partner is sitting alone at a restaurant across town — you were supposed to be there twenty minutes ago. You have a 9am meeting tomorrow in another city, and your boss is expecting you in person. You open your hotel app: *sold out, sold out, sold out.* Your phone buzzes. It's your partner: *"Where are you? They're about to give away our table."*

You need to rebook a flight, message your partner something that doesn't start a fight, find a hotel, and email your boss a backup plan — all in the next fifteen minutes. You don't need "suggestions." You need someone to *handle* it.

## The problem with AI personal assistants today

Ask GPT-4 or Claude to help you with this exact situation and you'll get a well-structured, empathetic, completely unhelpful response. Something like: *"I understand this must be very stressful. Here are some steps you might consider..."* followed by a generic numbered list that tells you nothing you didn't already know.

The failure mode is predictable. Current LLMs **describe** solutions instead of **executing** them. They can't prioritize between your angry partner and your strict boss. They won't draft the actual message to send. They treat every stakeholder with equal, vague politeness — which is exactly the wrong move when you have four competing crises and ten minutes. Real crisis management requires trade-offs, specificity, and the ability to craft different messages for different people with different personalities.

## Introducing LifeOS Agent

LifeOS Agent is an OpenEnv reinforcement learning environment that trains LLMs to resolve cascading personal crises through concrete actions. The environment presents **9 crisis scenarios** across three difficulty tiers — from a simple meeting overrun (1 conflict) to a full travel meltdown with four simultaneous stakeholder emergencies.

The agent observes a crisis scenario with active conflicts, persona descriptions, and time pressure. It must respond with a structured action: choosing an action type (send_message, reschedule, delegate), targeting a specific person, crafting the actual message content, explaining its reasoning, and declaring urgency. This isn't multiple-choice — the agent generates real communication.

What makes it novel: no existing RL environment trains specifically for multi-stakeholder personal conflict resolution. This targets a genuine capability gap in how LLMs function as personal assistants.

## Reward design — the hard part

Instead of a single opaque score, we decompose reward into **five independent functions**:

**Conflict Addressed (30%)** checks whether the response actually references the active crisis. Mentioning "reschedule to the red-eye" scores higher than "I'll look into options." **Stakeholder Reached (25%)** verifies the agent contacted a real persona in the scenario, not a made-up person. **Action Specificity (20%)** rewards concrete time references and action verbs — "call the airline now to rebook the 11pm flight" beats vague hand-waving. **Format Compliance (15%)** ensures the reasoning is substantive and urgency is valid. **No Escalation (10%)** penalises the generic filler phrases that LLMs default to.

The agent progresses through a **curriculum**: it trains on easy scenarios first (1 conflict), then medium (2 conflicts), then hard (3–4 conflicts). This prevents the model from being overwhelmed early and builds composable conflict-resolution skills.

We also built explicit anti-reward-hacking safeguards: duplicate content detection returns zero reward, responses under 30 characters score zero across all components, and generic phrases like "I will try my best" trigger a penalty.

## Results

After 60 training steps on a free Colab T4 GPU, reward improved measurably across all five components. The trained agent now names specific stakeholders, commits to time windows, and avoids the generic filler phrases that plague untrained models.

The before/after comparison is striking. Untrained: *"I understand this is a difficult situation. I recommend contacting the airline."* Trained: *"I need to contact the airline now to reschedule to the 11pm red-eye, then immediately message Partner_Jamie with a specific arrival time."*

That's the difference between an AI that *talks about* crisis management and one that *does* it.

## Try it yourself

The environment is live on [HuggingFace Spaces](https://huggingface.co/spaces/heyjan/lifeos-agent). The full training notebook runs on a free Colab T4 GPU in under 40 minutes. Clone the repo, run `openenv build`, and start training your own crisis management agent.

We're at an inflection point for personal AI assistants. The next generation won't just answer questions — they'll manage your life when things go sideways. LifeOS Agent is a step toward training them for the messiness of real human situations.

---

*[LifeOS Agent on GitHub](https://github.com/Janshafin/lifeos_agent) · [Training Notebook](YOUR_COLAB_LINK) · Built for the OpenEnv Hackathon 2026*
