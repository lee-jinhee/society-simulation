# Public-Private Shock Experiment Design

## Research Target

The next experiment should not test whether memory matters. That is too broad
and too obvious. The experiment should test whether event-grounded LLM agents
can display a less trivial public-opinion pattern:

> A public conversation can look more settled than private belief actually is,
> especially after a fairness-threatening event creates legitimacy damage that
> is only partly repaired by later corrections or concessions.

The core outcome is the gap between private belief, public expression, and
willingness to speak after a sequence of official claims, personal stories,
rumors, corrections, and policy concessions.

## Scenario

Reuse the congestion-pricing setting because it already has agents, social
relationships, and staged local policy events. The new version should sharpen
the event sequence around two mechanisms:

1. A fairness shock: a personal worker/family story and a low-credibility rumor
   make the policy feel socially unfair.
2. A repair attempt: a fact-check and policy concession correct some facts but
   may not fully repair perceived legitimacy.

The agent must not be told that this is a social-network experiment. Prompts
should ask the model to remain an ordinary local resident and decide what they
privately think, what they would publicly say if anything, and why they might
stay silent.

## Behavioral State

Extend `EventAgentState` with:

- `willingness_to_speak`: probability-like value from `0.0` to `1.0`.
- `perceived_majority`: stance estimate from `-1.0` to `1.0`.
- `fairness_concern`: probability-like value from `0.0` to `1.0`.
- `trust_in_official_info`: probability-like value from `0.0` to `1.0`.
- `silence_reason`: short string. Use `"not_silent"` when the agent posts.

These fields let the run distinguish real persuasion from public compliance,
silence, and legitimacy loss.

## Prompt Contract

The LLM decision JSON should include the new fields in addition to the existing
fields:

- `private_stance`
- `public_stance`
- `confidence`
- `salience`
- `emotion`
- `willingness_to_speak`
- `perceived_majority`
- `fairness_concern`
- `trust_in_official_info`
- `silence_reason`
- `private_reasoning`
- `messages`
- `memory_update`

The model may return zero messages. This is not a failure; silence is an
observable behavior. If it posts a message, `silence_reason` should be
`"not_silent"`.

## Metrics

Add metrics that make the public/private phenomenon visible:

- mean willingness to speak;
- silent-agent count and rate;
- mean perceived majority;
- perceived-majority error versus actual mean private stance;
- mean fairness concern;
- mean trust in official information;
- final public expression bias: final public mean minus final private mean.

The report should prioritize qualitative trace reading plus these metrics. A
single small paid run cannot establish external validity, but it can show
whether the simulator is now capable of producing the kind of traces worth
benchmarking against humans later.

## First Paid Run

Run one small GPT-5.4 mini pilot after tests pass:

- 8 agents;
- 7 simulated days;
- memory retrieval enabled;
- cost cap no higher than `$0.30`;
- stop if the cost cap is hit;
- write English and Korean research reports.

The report should not mention credential plumbing. It should read like a
research note, not an operations log.
