# LifeOS Agent — 90-Second YouTube Demo Script

---

## [0:00–0:20] HOOK — Read the crisis aloud (fast, vivid)

"It's 6:47pm on a Tuesday. Your flight just got cancelled.

Your partner is sitting alone at a restaurant waiting for you. You have a 9am meeting tomorrow in another city — your boss doesn't know you might miss it. Every hotel near the venue is sold out. Your phone buzzes: 'Where are you?'

Four crises. Ten minutes. What does your AI assistant say? 'I understand this is stressful. Here are some steps you might consider...' Useless. We built something better."

---

## [0:20–0:45] SHOW THE ENVIRONMENT — Screen recording walkthrough

"This is LifeOS Agent — an RL training environment built on OpenEnv.

*[Show terminal: docker container starting on port 8001]*

It serves 9 crisis scenarios across three difficulty tiers. The agent sees a crisis, identifies the stakeholders, and takes structured actions — choosing who to contact, what to say, and how urgent it is.

*[Show a sample interaction: the agent receives the flight cancellation scenario and responds with ACTION_TYPE, TARGET_PERSON, CONTENT, REASONING, URGENCY]*

Every response is scored by five independent reward functions — not one single score. Conflict addressed, stakeholder reached, action specificity, format compliance, and a penalty for generic filler phrases."

---

## [0:45–1:05] SHOW THE REWARD CURVE — Describe what the chart shows

"Here's what 60 training steps look like.

*[Show reward_curve.png — top plot]*

The green zone is easy scenarios — one conflict. Orange is medium — two conflicts. Red is hard — three or four simultaneous crises.

You can see the reward climbing as the model learns to address conflicts by name, target real stakeholders, and include specific times in its messages.

*[Show bottom plot]*

And here are all five reward components tracked independently. Conflict addressing improves fastest. Specificity takes longer to learn. That's exactly what you'd expect."

---

## [1:05–1:30] BEFORE/AFTER — Make the difference obvious

"Before training, the model says:

*'I understand this is a difficult situation. I recommend you contact the airline and inform your partner.'*

After training:

*'I need to contact the airline now to reschedule to the 11pm red-eye, then immediately message Partner_Jamie: My flight was cancelled. I'm rebooting on the red-eye. I'll arrive by 6am — don't wait up.'*

That's the difference. Not suggestions. Actions.

**LifeOS Agent. Available on HuggingFace. Link in the description.**"

---

## PRODUCTION NOTES

- **Total read time at normal pace**: ~87 seconds
- **Screen recordings needed**: (1) Docker container starting, (2) sample agent interaction in terminal, (3) reward_curve.png full screen, (4) before/after text comparison side by side
- **Tone**: Confident, fast, no filler. Let the results speak.
- **Music**: Low-energy electronic, fade under voice at 0:00, fade out at 1:30
- **Thumbnail text**: "Your AI Can't Handle This" with a split screen: generic advice vs. specific action
