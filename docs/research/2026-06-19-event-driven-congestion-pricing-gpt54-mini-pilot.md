# Event-Grounded Conversational Opinion Dynamics with GPT-5.4 Mini

Date: 2026-06-19

## Abstract

We ran an event-driven conversational opinion dynamics pilot with eight persona agents, seven simulated days, staged congestion-pricing events, and previous-day group-chat exposure. Each agent used GPT-5.4 mini to update private stance, public stance, confidence, salience, emotion, memory, and one natural-language message per day. The run completed all 56 planned LLM calls with no parse errors, no invalid generated messages, and an estimated total cost of `$0.082902`. Compared with the deterministic mock policy, the LLM agents moved toward average support for the policy while preserving heterogeneous dissent among cost-sensitive personas. This pilot is not evidence that the system predicts real public opinion, but it shows that the new event-driven simulator can execute an auditable, budget-bounded LLM society run with richer dynamics than the previous toy herding setup.

## Research Question

The purpose of this pilot was to test whether event-grounded LLM personas can produce coherent crowd-level opinion dynamics from staged public information and conversational feedback.

The concrete questions were:

1. Can the event-driven runner complete a paid LLM society run without schema failure or token explosion?
2. Do LLM personas react to staged events differently from the deterministic mock keyword policy?
3. Does the replay trail preserve enough information to audit individual decisions, aggregate dynamics, and cost?

## Method

The scenario is a congestion-pricing local policy debate.

- Agents: 8 persona profiles.
- Duration: 7 update days plus day 0 initial states.
- Events: staged official announcements, personal stories, opposition messages, fact checks, and late coalition framing.
- Interaction channel: neighborhood group chat.
- Decisions: each agent updates private stance, public stance, confidence, salience, emotion, memory, and at most one message per day.
- Model: GPT-5.4 mini.
- Maximum calls: `8 agents * 7 days = 56`.
- Cost control: per-run estimated cost cap of `$0.50`.

Pricing used the OpenAI model page for GPT-5.4 mini: `$0.75` per 1M input tokens and `$4.50` per 1M output tokens.

## Operational Note

The first paid attempt used `max_completion_tokens=180`. It stopped after one call because the model produced a valid-looking JSON object that was truncated by the token limit. That failed attempt cost `$0.0011235` and wrote a partial replay with one audit row.

We then hardened the prompt and runner context:

- the model receives allowed channels;
- the model receives allowed recipients;
- messages are constrained to at most one short post;
- private reasoning, memory, and message text are bounded;
- event IDs and source IDs are explicitly disallowed as recipients.

The second paid run used `max_completion_tokens=700` and completed successfully.

## Results

### Completion and Cost

| quantity | value |
| --- | ---: |
| planned LLM calls | 56 |
| completed LLM calls | 56 |
| parse or validation errors | 0 |
| prompt tokens | 59,614 |
| completion tokens | 8,487 |
| estimated input cost | `$0.0447105` |
| estimated output cost | `$0.0381915` |
| estimated total cost | `$0.082902` |
| max completion tokens observed | 171 |

The run stayed far below the `$0.50` cap and far below the operator stop threshold of `$3.00`.

### Aggregate Opinion Dynamics

| metric | GPT-5.4 mini | deterministic mock |
| --- | ---: | ---: |
| final private stance mean | 0.3625 | -0.5375 |
| final public stance mean | 0.35125 | -0.24375 |
| final private-public gap | 0.05875 | 0.29375 |
| message count | 56 | 56 |

The deterministic mock moved strongly negative because its keyword rule reacted heavily to cost and livelihood terms. GPT-5.4 mini instead integrated the full policy framing: transit discounts, congestion reduction, exemptions, hospital access, and distributional concerns. The macro outcome was therefore more supportive and less privately/publicly split than the mock run.

### Final Agent States

| agent | private | public | confidence | salience | emotion |
| --- | ---: | ---: | ---: | ---: | --- |
| jisoo | 0.55 | 0.48 | 0.88 | 0.99 | cautiously supportive |
| minho | -0.18 | -0.05 | 0.90 | 0.99 | concerned |
| amara | 0.78 | 0.74 | 0.92 | 0.99 | thoughtful |
| carlos | 0.02 | 0.00 | 0.90 | 0.98 | cautiously optimistic |
| mei | 0.61 | 0.54 | 0.90 | 0.98 | thoughtful |
| nora | 0.50 | 0.44 | 0.88 | 0.99 | concerned but more open |
| owen | -0.08 | -0.02 | 0.90 | 0.99 | skeptical but cautiously open |
| sara | 0.70 | 0.68 | 0.90 | 0.99 | thoughtful |

The final distribution is not unanimous. The most cost-sensitive personas remain negative or near neutral, while public-health and planning-oriented personas become supportive.

### Message Validity

All 56 generated messages used the configured group-chat channel. No private direct messages were emitted. No generated message failed runner validation.

Example first-day message:

> If the downtown charge really funds transit discounts, I’m open to it. I’d want clear exemptions and proof it helps hospital workers and patients.

## Interpretation

The most important finding is not that congestion pricing becomes popular. A single synthetic scenario cannot support that claim. The stronger result is operational and mechanistic:

1. Event-grounded LLM personas can complete a multi-day social run with auditable decisions and bounded cost.
2. The LLM policy does not behave like the keyword mock baseline.
3. The model preserves persona heterogeneity instead of collapsing into full consensus.
4. Public and private stances stayed close, suggesting this prompt does not yet create strong social-desirability pressure.

This is a more credible starting point than majority-count herding, because agents react to semantically rich events and produce natural-language social traces.

## Limitations

This is still a pilot, not a publishable final experiment.

Major limitations:

- one model;
- one seed;
- one policy domain;
- one city-like scenario;
- no empirical survey calibration;
- no comparison against human panel responses;
- no prompt ablation;
- no model comparison against Chinese low-cost models;
- no explicit social pressure manipulation;
- no external news ingestion.

The run demonstrates feasibility and reveals useful mechanisms, but it does not validate predictive accuracy.

## Next Experiments

The next research step should be a controlled mechanism study:

1. Run the same scenario across multiple models, including cheaper Chinese OpenAI-compatible models.
2. Add prompt variants that manipulate public pressure, conformity, and reputational concern.
3. Add a calibrated human-response benchmark for a small set of event/persona prompts.
4. Run counterfactual event orders: official-first, opposition-first, personal-story-first, fact-check-first.
5. Measure whether LLM societies amplify, dampen, or polarize around shocks.

## Reproducibility

Configuration:

- `experiments/event_driven_congestion_pricing_gpt54_mini_pilot.json`

Artifacts:

- `runs/event_driven_congestion_pricing_gpt54_mini_pilot_v2/config.json`
- `runs/event_driven_congestion_pricing_gpt54_mini_pilot_v2/metrics.json`
- `runs/event_driven_congestion_pricing_gpt54_mini_pilot_v2/llm_decisions.jsonl`
- `runs/event_driven_congestion_pricing_gpt54_mini_pilot_v2/messages.jsonl`
- `runs/event_driven_congestion_pricing_gpt54_mini_pilot_v2/agent_states.jsonl`
