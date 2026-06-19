# Event-Driven Conversational Opinion Dynamics Design

Date: 2026-06-19

## Status

Approved direction: move beyond the current toy network-herding pilot toward an event-driven, conversational, persona-based opinion dynamics experiment.

The previous OpenAI mini pilot was useful as an infrastructure check: it verified real LLM calls, audit logs, cost accounting, replay artifacts, and sweep execution. It is not a publishable social simulation result. Showing an LLM a list such as `A, A, A, B, B` and observing that it often chooses `A` is expected conformity behavior, not a realistic model of human social interaction.

This design defines the next experiment: agents should behave like situated people who encounter news, rumors, personal stories, and messages from people they trust or distrust. The system should measure how private opinions, public speech, salience, emotion, and memory evolve under social exposure.

## Core Research Question

Can LLM agents with stable human-like profiles, memories, relationships, and heterogeneous media exposure produce interpretable opinion dynamics after staged real-world-like events and conversations?

More specifically:

> When people with different interests, trust levels, relationships, and media diets encounter a sequence of public events and private conversations, do we observe persuasion, resistance, silence, public-private opinion gaps, emotional escalation, or polarization?

## Why This Is Different From the Toy Pilot

The old `network_herding` pilot asked:

> Given neighbor actions, does an LLM choose A or B?

The new experiment asks:

> Given a life situation, trusted and distrusted relationships, imperfect information, public conversation, private memory, and new events over time, what does a person privately believe, what do they publicly say, and what changes their mind?

The new experiment must not tell the agent it is simulating a network experiment. The agent should be placed in a realistic role and asked to stay in character. Experiment-control fields may exist in the JSON output, but they should not dominate the role instruction.

## Design Alternatives

### Alternative A: Add Structured Event Shocks to `network_herding`

This would keep binary A/B opinions and add fields such as event stance, salience, and credibility.

Pros:

- small implementation change;
- easy to compare against current metrics;
- cheap to run.

Cons:

- still mostly a numeric opinion toy;
- does not produce natural conversation;
- does not model public-private differences;
- weak human-behavior validity.

Decision: reject as the main next experiment. Keep it only as a deterministic baseline later.

### Alternative B: Event-Driven Conversational Opinion Dynamics

Agents have profiles, relationships, memory, media exposure, private stance, public stance, and natural-language messages. Staged events enter the world over several days. Agents may speak, remain silent, privately change their mind, resist persuasion, ask questions, or selectively share information.

Pros:

- directly addresses the user's critique;
- creates auditable human-like traces;
- supports event injection;
- supports private/public opinion gaps;
- still small enough to implement and run cheaply.

Cons:

- more expensive per step than binary decisions;
- harder to score automatically;
- requires careful prompt and schema design.

Decision: recommended next experiment.

### Alternative C: Full Generative Society

Build a broader Smallville-like society with daily schedules, locations, institutions, long-term memory, and open-ended social life.

Pros:

- closer to the long-term vision;
- richer emergent behavior.

Cons:

- too large for the next step;
- hard to evaluate;
- likely to become a demo before becoming a research instrument.

Decision: defer. The next experiment should be narrower and scientifically inspectable.

## Recommended Experiment

Scenario name:

```text
event_driven_opinion_dynamics
```

First study:

```text
Congestion pricing in a fictional mid-sized city
```

The topic should be realistic but fictional. It should affect agents differently, support reasonable arguments on both sides, and allow personal experience to compete with official information.

Example issue:

> The city is considering a weekday congestion charge for cars entering the downtown core. The policy is framed as a way to reduce traffic and improve air quality, but opponents argue that it burdens workers and small businesses.

This issue is useful because it naturally creates cross-cutting pressures:

- commuters want shorter travel times but dislike fees;
- small businesses fear losing customers;
- public-health workers care about air quality;
- delivery drivers and taxi drivers face direct income risks;
- climate-minded residents favor the policy;
- institutionally distrustful residents may suspect hidden motives.

## Agent Model

Each agent has a stable profile and dynamic state.

### Static Profile

Fields:

- `agent_id`;
- `name`;
- `age`;
- `occupation`;
- `household_context`;
- `neighborhood`;
- `core_values`;
- `material_interests`;
- `political_trust`;
- `media_habits`;
- `communication_style`;
- `susceptibilities`;
- `initial_private_stance`;
- `initial_public_stance`;
- `initial_confidence`;
- `initial_salience`.

Example:

```json
{
  "agent_id": "jisoo",
  "name": "Jisoo Park",
  "age": 37,
  "occupation": "hospital nurse",
  "household_context": "single parent with a teenage son",
  "neighborhood": "east side commuter district",
  "core_values": ["practicality", "fairness", "public health"],
  "material_interests": ["commute time", "parking cost", "hospital staffing"],
  "political_trust": 0.45,
  "media_habits": ["local news", "neighborhood group chat"],
  "communication_style": "careful, concrete, conflict-avoidant",
  "susceptibilities": ["personal stories from coworkers", "child-related costs"],
  "initial_private_stance": -0.1,
  "initial_public_stance": 0.0,
  "initial_confidence": 0.35,
  "initial_salience": 0.45
}
```

### Dynamic State

Fields:

- `private_stance`: continuous support for the policy, from `-1.0` to `1.0`;
- `public_stance`: what the agent is willing to express publicly, from `-1.0` to `1.0`;
- `confidence`: certainty in current stance, from `0.0` to `1.0`;
- `salience`: how personally important the issue feels, from `0.0` to `1.0`;
- `emotion`: short categorical label such as `calm`, `annoyed`, `worried`, `angry`, `hopeful`, `conflicted`;
- `trust_by_source`: source-specific trust values;
- `memory_summary`: compact natural-language memory;
- `last_private_reasoning`: hidden research artifact, not shown to other agents;
- `messages_posted`: public or private messages emitted this step.

The simulator should persist dynamic state after every step.

## Relationship Model

Relationships are not only graph edges. Each edge should carry a relationship type and trust score.

Fields:

- `source_agent_id`;
- `target_agent_id`;
- `relationship_type`: `friend`, `coworker`, `family`, `neighbor`, `weak_tie`;
- `trust`: `0.0` to `1.0`;
- `conversation_frequency`: `low`, `medium`, `high`;
- `conflict_sensitivity`: `0.0` to `1.0`.

The relationship graph controls which messages an agent sees and how much personal influence a message plausibly has.

## Event Model

Events are staged over simulated days. They are not just numeric shocks. They are pieces of information that enter the agents' media environment or social channels.

Event fields:

- `event_id`;
- `day`;
- `title`;
- `source`;
- `source_type`: `official`, `news`, `personal_story`, `viral_clip`, `fact_check`, `private_message`;
- `content`;
- `policy_stance`: continuous value from `-1.0` to `1.0`;
- `credibility`;
- `emotional_intensity`;
- `affected_interests`;
- `audience_filter`.

Example schedule:

| day | event |
| ---: | --- |
| 0 | Baseline private diary and initial attitude survey |
| 1 | City announces congestion-pricing proposal |
| 2 | Taxi driver interview claims the fee threatens livelihoods |
| 3 | Public-health report predicts lower asthma-related ER visits |
| 4 | Friend shares a personal commute story in group chat |
| 5 | Viral clip alleges the city is hiding revenue motives |
| 6 | Local newspaper publishes a fact-check |
| 7 | Final group discussion and private survey |

The important design point: not every agent sees every event. Exposure depends on media habits, relationships, and channel membership.

## Channels

First implementation should support three channel types:

1. `news_feed`
   - one-way exposure from event sources;
   - controlled by media habits and audience filters.

2. `group_chat`
   - many-to-many conversation;
   - agents can post short natural-language messages;
   - all channel members see the messages.

3. `private_dm`
   - one-to-one message;
   - controlled by relationship graph and agent decision.

Do not add public social media feeds yet. Public feeds introduce recommendation logic, virality, and scale problems that should come after the first event-driven study works.

## Agent Prompt Design

The agent should not be told:

```text
You are simulating a social network experiment.
```

The role prompt should instead place the agent in a human situation:

```text
You are Jisoo Park, a 37-year-old hospital nurse in a mid-sized city.
You are a single parent and commute by car on weekdays.
You care about fairness, public health, commute time, and your son's school schedule.
Stay in character. Do not mention being an AI, a model, a simulation, or an experiment.
```

The user message should include only what the person plausibly experiences:

```text
Today is Day 4.

Recent memory:
You were unsure about the congestion charge. You disliked the idea of another fee,
but you also noticed traffic has made hospital shifts harder.

You saw:
- Local news headline: ...
- Neighborhood group chat message from Minho: ...
- Private DM from Clara: ...

Decide what you privately think now, whether you say anything, and what you say.
```

The model should still return structured JSON for logging:

```json
{
  "private_stance": -0.25,
  "public_stance": 0.0,
  "confidence": 0.52,
  "salience": 0.77,
  "emotion": "worried",
  "private_reasoning": "The fee feels unfair, but the hospital commute story matters.",
  "messages": [
    {
      "channel": "group_chat",
      "recipient": null,
      "text": "I get the air quality argument, but I worry this lands hardest on people who cannot change their commute."
    }
  ],
  "memory_update": "Jisoo is more worried about fairness after hearing coworkers discuss commute costs."
}
```

The JSON schema is an audit interface. The natural-language role should remain human-facing.

## Experimental Conditions

Run a small set of conditions, not a large seed sweep.

Recommended first batch:

1. `news_only`
   - agents see staged events;
   - no agent-to-agent messages.

2. `news_plus_group_chat`
   - agents see staged events;
   - group chat messages are visible to channel members.

3. `news_plus_group_chat_private_public`
   - same as group chat;
   - explicitly track private stance and public stance separately.

4. `news_plus_group_chat_memory`
   - same as private/public;
   - agents carry compact memory summaries across days.

This isolates what each component adds. The first publishable direction is not "the agents talked." It is whether conversation, memory, and public/private split produce different opinion dynamics than news-only exposure.

## Scale

The first experiment should be deliberately small:

- agents: 8 to 12;
- simulated days: 7;
- LLM decisions: roughly one per active agent per day;
- seeds: 1 or 2;
- model: cheap OpenAI-compatible model first, then OpenAI mini comparison if needed.

This is not a statistical sweep. It is a trace-validity pilot: can we produce and audit believable human-like dynamics?

## Metrics

Primary metrics:

- mean private stance by day;
- mean public stance by day;
- private-public gap by day;
- stance variance and polarization;
- salience by day;
- confidence by day;
- emotion distribution;
- number of public messages;
- number of private messages;
- agent-level stance change after each event;
- source exposure counts;
- influence trace from message exposure to later stance change.

Qualitative audit metrics:

- whether messages sound like situated people rather than survey respondents;
- whether agents preserve stable identity over days;
- whether agents sometimes remain silent;
- whether agents resist evidence when it conflicts with trust or interests;
- whether personal stories affect salience more than official statistics.

## Hypotheses

Initial hypotheses:

1. Personal stories increase salience more than neutral statistical reports.
2. Messages from high-trust ties produce larger stance movement than low-trust news exposure.
3. Public stance is more moderate than private stance in mixed-opinion group chats.
4. Fact-checks reduce belief in false claims for high-trust-in-institutions agents, but may fail or backfire for low-trust agents.
5. Bridge agents reduce group-level polarization by carrying arguments across social clusters.

These hypotheses are stronger than "agents converge" because they are about mechanisms that can be inspected in traces.

## Artifacts

Each run should write:

- `config.json`;
- `agents.json`;
- `relationships.json`;
- `events.json`;
- `exposures.jsonl`;
- `messages.jsonl`;
- `agent_states.jsonl`;
- `llm_decisions.jsonl`;
- `metrics.json`;
- `summary.md`.

The audit trail must allow reconstruction of:

- what each agent knew at each step;
- what each agent said;
- what each agent privately believed;
- which events and messages preceded stance changes;
- which model call produced each state update.

## Implementation Boundaries

Add a new scenario rather than stretching `network_herding`.

New scenario:

```text
event_driven_opinion_dynamics
```

The existing LLM provider, cost accounting, prompt replay, and sweep infrastructure should be reused. The binary network-herding state model should not be reused as the main state object, because it is too narrow.

Likely new modules:

- `event_models.py`: agent profiles, relationships, events, exposures, messages, dynamic state;
- `event_config.py`: scenario-specific config parsing and validation, unless the existing config parser can stay clean;
- `event_runner.py`: day-by-day execution loop;
- `event_policy.py`: LLM persona policy and structured output parser;
- `event_replay.py`: artifacts for agents, relationships, events, exposures, messages, and states;
- `event_metrics.py`: stance, salience, public-private gap, message volume, and exposure metrics.

If the existing `config.py` becomes too large, split scenario-specific config types into focused modules while preserving the public CLI behavior.

## Evaluation Standard

This first event-driven experiment is successful if:

- agents read as distinct people with stable interests;
- agent outputs include plausible silence, uncertainty, and conflict avoidance;
- private and public stance can diverge;
- staged events produce interpretable changes in stance, salience, or emotion;
- group chat messages influence later private reasoning or public speech;
- every claim in the analysis can be traced to concrete exposures, messages, and model outputs;
- total cost is predictable before the run and measured after the run.

It is not successful merely because the final average opinion changes.

## Non-Goals

The first version will not include:

- real-time web news ingestion;
- real polling prediction;
- large seed sweeps;
- recommendation algorithms;
- image/video understanding;
- autonomous daily schedules;
- location simulation;
- economic transactions;
- model fine-tuning;
- claims of external validity.

Those can come after the trace-validity pilot works.

## Next Step

Write an implementation plan for a minimal but complete version of `event_driven_opinion_dynamics`.

The first implementation should produce one runnable local experiment:

```bash
python -m society_simulation run experiments/event_driven_congestion_pricing.json
```

The run should use a mock LLM policy by default for tests and support an OpenAI-compatible policy for the paid pilot. The paid pilot should stay under a small explicit budget by limiting agents, days, output tokens, and number of conditions.
