# Social Memory Layer Design

Date: 2026-06-20

## Status

Approved direction: the project should not become a clone of Park et al.'s
Generative Agents sandbox. Memory streams, retrieval, and reflection are
necessary infrastructure, but they are no longer sufficient as a research
contribution by themselves.

The next research step is to turn memory into an experimental mechanism for
event-grounded public-opinion dynamics. The system should test how people-like
agents remember, selectively retrieve, reinterpret, and publicly express views
after news events, personal stories, rumors, fact checks, and social messages.

## Core Research Question

Can an event-grounded social memory architecture improve the behavioral validity
of LLM agent societies for public-opinion dynamics, beyond merely making agents
sound believable?

More specifically:

> When agents encounter sequential public events and social conversation, does
> memory retrieval shaped by recency, salience, source trust, emotion, identity
> relevance, and relationship context produce more realistic patterns of
> persuasion, resistance, silence, polarization, and public-private opinion gaps?

## Critical Reading of Park et al.

Park et al. introduced the right architectural primitives for long-lived LLM
agents: memory stream, retrieval, reflection, planning, and reaction. The paper's
strength is that it shows why stateless personas are weak and why agents need
long-term experience to remain coherent.

For this project, the limitation is the evaluation target. Park et al. mainly
evaluate believability in an interactive sandbox. Believability is important,
but it is not the same as empirical validity. A behavior can read as plausible
while failing to match human opinion change, social desirability pressure,
motivated reasoning, or poll movement.

The paper also exposes risks that matter directly here:

- retrieval failure can make agents miss relevant past information;
- incomplete retrieval can create strange certainty about partial facts;
- LLM priors can embellish memories;
- instruction tuning can make agents overly polite, cooperative, and moderate;
- short sandbox timescales do not establish robust social prediction.

Our contribution should therefore be:

> Shift generative agents from believable NPC simulation toward auditable,
> event-grounded, empirically checkable opinion-dynamics experiments.

## Design Alternatives

### Alternative A: Smallville-Style General Society

Build schedules, locations, daily routines, open-ended interaction, and social
life before returning to opinion experiments.

Pros:

- closest to the iconic Generative Agents demo;
- rich qualitative behavior;
- useful for future product or game-like interfaces.

Cons:

- too broad for the current research question;
- hard to evaluate against human opinion data;
- expensive and likely to drift into demo-building;
- does not directly address public opinion or crowd psychology.

Decision: reject for the next step.

### Alternative B: Minimal Park-Style Memory Stream

Add a generic memory table with recency, relevance, and importance retrieval,
then pass retrieved memories into the existing event prompt.

Pros:

- small and clean implementation;
- directly follows prior art;
- likely improves continuity.

Cons:

- weak novelty;
- treats memory as generic context rather than a social mechanism;
- does not model motivated recall, source trust, identity threat, or public
  expression pressure.

Decision: useful baseline, but not the main contribution.

### Alternative C: Event-Grounded Social Memory Layer

Add memory objects, retrieval, and reflection, but make their scoring and
evaluation specific to opinion dynamics. Memories carry social and political
metadata: source trust, emotional intensity, stance signal, identity relevance,
relationship context, public/private origin, and whether the memory came from an
event, message, reflection, or self-commitment.

Pros:

- builds on Park et al. without merely copying it;
- directly supports human-like opinion dynamics;
- creates ablations that can be evaluated scientifically;
- connects to public/private stance, social pressure, and media events.

Cons:

- more design complexity;
- requires careful metrics;
- can become overfit if we add too many psychological knobs too early.

Decision: recommended.

## Scope for the First Build

The first build should add the social memory layer to
`event_driven_opinion_dynamics` without changing the CLI surface or requiring
paid LLM calls.

It should support deterministic mock tests first. LLM prompts can consume the
new retrieved-memory context after the model and replay layer are reliable.

In scope:

- memory object model;
- deterministic memory store;
- retrieval scoring with transparent components;
- event/message/decision memory creation;
- replay artifacts for memories and retrieval traces;
- metrics for memory count, retrieved memory count, and retrieval score
  components;
- prompt integration that adds retrieved memories without exposing experiment
  mechanics to agents;
- ablation-ready config flag to run with or without memory retrieval.

Out of scope for the first build:

- embedding API calls;
- vector database;
- autonomous daily schedules;
- real news ingestion;
- human benchmark collection;
- new paid LLM experiment.

## Data Model

### `SocialMemory`

Each memory is an immutable record attached to one agent.

Fields:

- `memory_id`: stable deterministic id;
- `agent_id`;
- `day`;
- `kind`: `event_exposure`, `social_message`, `self_reasoning`,
  `self_message`, `reflection`;
- `text`: natural-language memory content;
- `source_id`: event id, message id, reflection id, or decision id;
- `source_type`: event/message/self/reflection source label;
- `channel`: `news_feed`, group chat id, private DM id, or `internal`;
- `related_agent_ids`;
- `related_event_ids`;
- `stance_signal`: continuous value from `-1.0` to `1.0`;
- `emotional_intensity`: `0.0` to `1.0`;
- `source_trust`: `0.0` to `1.0`;
- `identity_relevance`: `0.0` to `1.0`;
- `importance`: `0.0` to `1.0`;
- `private`: boolean indicating whether this memory should remain internal.

### `MemoryQuery`

The query represents the current decision context.

Fields:

- `agent_id`;
- `day`;
- `text`;
- `related_agent_ids`;
- `related_event_ids`;
- `stance_hint`;
- `affected_interests`;

### `RetrievedMemory`

Returned by retrieval and written to replay.

Fields:

- `memory`;
- `score`;
- `recency_score`;
- `relevance_score`;
- `importance_score`;
- `trust_score`;
- `emotion_score`;
- `identity_score`;

## Retrieval Scoring

The first implementation should avoid embeddings. Use transparent metadata and
token overlap so every retrieval decision is auditable and cheap.

Score:

```text
score =
  0.20 * recency_score +
  0.25 * relevance_score +
  0.20 * importance_score +
  0.10 * trust_score +
  0.10 * emotion_score +
  0.15 * identity_score
```

Definitions:

- `recency_score`: exponential decay by day distance;
- `relevance_score`: normalized token overlap plus related event/agent overlap;
- `importance_score`: stored memory importance;
- `trust_score`: stored source trust;
- `emotion_score`: stored emotional intensity;
- `identity_score`: stored identity relevance.

This is intentionally simple. The research value is not that this is the final
psychological model; the value is that the mechanism is inspectable and can be
ablated.

## Memory Creation Rules

The runner should create memories at four points.

1. **Event exposure**
   - One memory per exposed event per agent.
   - Trust comes from event credibility, adjusted later by agent trust profile.
   - Identity relevance is higher when event `affected_interests` overlap with
     the agent's material interests or core values.

2. **Social message exposure**
   - One memory per message seen by an agent.
   - Trust comes from relationship trust if sender is known, otherwise neutral.
   - Identity relevance is based on text overlap with agent interests and values.

3. **Agent decision**
   - Store private reasoning as a private self memory.
   - Store emitted messages as self-message memories.

4. **Reflection**
   - Not required in the first code task.
   - The model should leave a clean interface for adding reflection later.

## Prompt Integration

The LLM prompt should receive retrieved memories as ordinary resident context,
not as an experimental instruction.

Use wording such as:

```text
Things you currently remember:
- ...
```

Avoid wording such as:

```text
The simulator retrieved these memories using recency/relevance scores.
```

The prompt should keep the existing output schema:

- `private_stance`;
- `public_stance`;
- `confidence`;
- `salience`;
- `emotion`;
- `private_reasoning`;
- `messages`;
- `memory_update`.

The memory layer changes the context, not the output contract.

## Metrics and Replay

Replay should add:

- `memories.jsonl`;
- `retrievals.jsonl`;
- memory summary fields in `metrics.json`.

Metrics:

- `memory_count`;
- `retrieval_count`;
- `mean_retrieved_memories_per_decision`;
- `mean_retrieval_score`;
- `retrieval_kind_counts`;
- `private_memory_count`;
- `public_memory_count`.

These are operational metrics. They do not claim predictive validity.

## Ablations

The first implementation should make these configurations possible:

1. `memory_retrieval.enabled = false`
   - current event-driven behavior;
   - no retrieved memories in prompt.

2. `memory_retrieval.enabled = true`
   - memories are stored and retrieved;
   - prompt includes retrieved memories.

Later experiments can add:

- no trust term;
- no emotion term;
- no identity term;
- no social message memories;
- reflection on/off.

## Error Handling

The memory layer must fail closed:

- invalid memory scores outside `0.0` to `1.0` raise `ValueError`;
- invalid stance signals outside `-1.0` to `1.0` raise `ValueError`;
- missing agent ids raise during config validation or runner setup;
- replay writing must not persist secrets;
- retrieval with no memories returns an empty tuple, not an error.

## Testing Strategy

Tests should be deterministic and should not call LLM APIs.

Required tests:

- memory model validation;
- memory query and retrieval scoring;
- event exposure creates memory objects;
- message exposure uses relationship trust;
- decision memories are private when they contain private reasoning;
- runner works with retrieval disabled;
- runner writes `memories.jsonl` and `retrievals.jsonl` when enabled;
- prompt includes retrieved memories only when enabled;
- full pytest suite remains green.

## Research Interpretation

This build will not by itself prove that the simulator predicts human society.
It creates the missing substrate for the next meaningful experiment:

> Given the same event sequence and population, compare opinion dynamics under
> no memory, generic memory, and social memory retrieval. Then compare those
> shifts against human panel responses or real polling traces.

That is where the publishable contribution should emerge.
