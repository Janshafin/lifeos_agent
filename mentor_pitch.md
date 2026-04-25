# LifeOS Agent — Mentor Pitch Answers

---

## 1. What problem are you solving?

Current LLMs fail at real personal crisis management. When you're facing a cancelled flight, an angry partner, a sold-out hotel, and a boss who doesn't know — all at once — today's AI gives you a polite bulleted list of suggestions. It describes what you *could* do instead of *doing* it. LifeOS Agent trains models to take specific, prioritised actions across multiple stakeholders under time pressure. We're closing the gap between "here are some ideas" and "I've drafted the message, rebooked the flight, and emailed your boss — which should I send first?"

---

## 2. What have you built so far?

We've built a complete OpenEnv-compliant RL training environment with 9 crisis scenarios across 3 difficulty tiers, 5 independent reward functions, curriculum learning that progresses from easy to hard, and 3 anti-reward-hacking safeguards. The environment runs in Docker, serves via FastAPI with WebSocket support, and we've trained Qwen2.5-3B-Instruct using Unsloth 4-bit quantization with LoRA on a free Colab T4 GPU. We have measurable reward improvement across all five components, a complete training notebook, and deployment on HuggingFace Spaces.

---

## 3. Why did you choose this problem?

Everyone has experienced a cascading personal crisis — a travel meltdown, a scheduling collision, a family emergency during a work deadline. These are universal, high-stakes, emotionally charged situations where people actually *need* AI help. Yet no existing RL environment trains for this capability. Multi-stakeholder personal conflict resolution is an unsolved, high-impact problem that sits at the intersection of reasoning, communication, and emotional intelligence — exactly where LLMs need to improve most.

---

## 4. How does your reward function work?

We use five independent reward functions, each targeting a distinct capability. Conflict Addressed (30%) checks if the response references an active crisis. Stakeholder Reached (25%) verifies the agent targeted a real persona. Action Specificity (20%) rewards concrete time references and action verbs. Format Compliance (15%) ensures substantive reasoning and valid urgency. No Escalation (10%) penalises generic filler phrases. Each component is computed independently and tracked separately, so we can diagnose exactly where the model is improving versus gaming the system. The total is capped at 1.0.

---

## 5. How do you prevent reward hacking?

Three explicit safeguards. First, duplicate detection — if the agent repeats the exact same content as its previous step, all five components return zero. No credit for copy-paste. Second, minimum content length — responses under 30 characters score zero across the board, preventing gaming through minimal output. Third, generic phrase penalty — we maintain a list of six common LLM filler phrases like "I will try my best" and "I apologize for any inconvenience." If any appear, the no_escalation component returns zero. These three rules together ensure the agent can't find cheap shortcuts.

---

## 6. What would you do with more time?

Three things. First, we'd add dynamic persona responses — right now personas give simple acknowledgements, but with an LLM-powered persona simulator, the agent would face realistic pushback and negotiation. Second, we'd implement multi-step planning where the agent must sequence actions across 3-5 steps to fully resolve a crisis, not just take one good action. Third, we'd scale to 50+ scenarios using procedural generation — combining conflict types, persona personalities, and time constraints to create an effectively infinite training distribution.

---

## 7. Why will this win?

Three reasons. First, **novelty** — no one has built an RL environment for personal crisis management. This is a genuinely new problem domain with clear real-world impact. Second, **technical rigour** — we have 5 independent reward functions, curriculum learning, anti-hacking safeguards, and measurable training improvement with labelled reward curves. We're not just training a model — we're demonstrating *how* it learns. Third, **storytelling** — everyone has been in a situation where their flight got cancelled and their phone won't stop buzzing. This problem resonates. Judges will remember it.
