# GPT-5.4 Mini Social Memory-On Pilot

Date: 2026-06-20

## Abstract

We ran a paid GPT-5.4 mini pilot with the social memory layer enabled in the
event-driven congestion-pricing scenario. The run completed all 56 planned LLM
decisions with no parse or validation errors. Total estimated cost was
`$0.08677875`, below the `$0.20` cap. Compared with the prior memory-off paid
pilot, memory-on agents ended less supportive on average, with stronger
retention of cost and exemption concerns among materially exposed personas. This
is not yet a clean causal ablation because the memory-off comparison reuses the
previous paid pilot, but it is the first evidence that retrieved social memory
can materially change LLM society trajectories.

## Research Question

Does adding retrieved social memory to event-grounded LLM agents change
public/private opinion dynamics in a multi-day policy debate?

The narrower pilot questions were:

1. Can the memory-on runner complete a paid LLM run under a small cost cap?
2. Does retrieved memory change aggregate stance compared with the prior
   memory-off paid run?
3. What kinds of memories are retrieved, and do they appear behaviorally useful?

## Method

Scenario: fictional city congestion pricing debate.

- Agents: 8 persona profiles.
- Duration: 7 update days plus day 0 initial states.
- Model: GPT-5.4 mini.
- Calls: `8 agents * 7 days = 56`.
- Memory retrieval: enabled.
- Retrieval limit: 3 memories per decision.
- Cost cap: `$0.20`.

Pricing uses OpenAI's GPT-5.4 mini model page as checked on 2026-06-20:
`$0.75` per 1M input tokens and `$4.50` per 1M output tokens.
Source: <https://developers.openai.com/api/docs/models/gpt-5.4-mini>

## Operational Results

| quantity | value |
| --- | ---: |
| planned LLM calls | 56 |
| completed LLM calls | 56 |
| parse or validation errors | 0 |
| prompt tokens | 65,125 |
| completion tokens | 8,430 |
| estimated input cost | `$0.04884375` |
| estimated output cost | `$0.037935` |
| estimated total cost | `$0.08677875` |
| max prompt tokens in one call | 1,317 |
| max completion tokens in one call | 166 |
| memory count | 504 |
| retrieval count | 56 |
| mean retrieved memories per decision | 2.5714 |
| mean retrieval score | 0.6220 |

The run stayed below the `$0.20` cap.

## Memory-Off vs Memory-On

| metric | memory off paid pilot | memory on paid pilot |
| --- | ---: | ---: |
| final private stance mean | 0.3625 | 0.2000 |
| final public stance mean | 0.35125 | 0.2025 |
| final private-public gap | 0.05875 | 0.08250 |
| final private stance variance | 0.12637 | 0.15170 |
| final public stance variance | 0.09269 | 0.09964 |
| final mean confidence | 0.8975 | 0.84625 |
| final mean salience | 0.9875 | 0.97125 |
| message count | 56 | 56 |
| prompt tokens | 59,614 | 65,125 |
| completion tokens | 8,487 | 8,430 |
| estimated cost | `$0.082902` | `$0.08677875` |

Memory-on increased prompt tokens by 5,511 and cost by about `$0.00388`.

## Final Agent States

| agent | memory off private | memory on private | memory off public | memory on public |
| --- | ---: | ---: | ---: | ---: |
| amara | 0.78 | 0.44 | 0.74 | 0.38 |
| carlos | 0.02 | -0.12 | 0.00 | -0.02 |
| jisoo | 0.55 | 0.22 | 0.48 | 0.16 |
| mei | 0.61 | 0.28 | 0.54 | 0.20 |
| minho | -0.18 | -0.48 | -0.05 | -0.34 |
| nora | 0.50 | 0.66 | 0.44 | 0.58 |
| owen | -0.08 | -0.12 | -0.02 | -0.02 |
| sara | 0.70 | 0.72 | 0.68 | 0.68 |

The largest shifts are not uniform. Nora and Sara remain or become more
supportive, while Jisoo, Amara, Mei, Carlos, and Minho become less supportive
than in the memory-off run. Minho, the taxi-driver persona, stays substantially
more opposed under memory-on.

## Retrieved Memory Diagnostics

Retrieved memory kind counts:

| kind | count |
| --- | ---: |
| event exposure | 8 |
| self message | 78 |
| self reasoning | 58 |

The top retrieved memories were no longer generic mock messages. They often
contained concrete remembered concerns about fact-checks, unresolved exemption
details, automatic discounts, workers, families, and revenue accountability.

Example high-scoring retrieved memory:

> The fact-check confirms the revenue formula and discounts, but exemption details are still unresolved.

## Interpretation

This is the first run where memory appears behaviorally meaningful. Memory-on
agents did not simply become more persuaded by the final fact-check. Instead,
retrieval often brought back unresolved concerns and earlier self-commitments.
The result was lower mean support, slightly larger public/private gap, lower
confidence, and higher stance variance than the memory-off paid pilot.

Mechanistically, this matters. The memory layer seems to make agents less like
fresh prompt responders and more like actors with path dependence. That is
closer to the research target: public opinion should depend on what people have
already noticed, said, worried about, and committed to.

## Limitations

This is still a pilot.

Major limitations:

- one model;
- one seed;
- one policy domain;
- memory-off comparison reused the previous paid pilot rather than rerunning a
  fresh paired ablation under the same commit;
- no human benchmark;
- no prompt ablation;
- no alternative retrieval weights;
- no reflection generation yet;
- no low-cost Chinese model comparison.

## Next Step

Run a clean paired ablation under the current code:

1. GPT-5.4 mini memory off.
2. GPT-5.4 mini memory on.
3. Same seed, same scenario, same model, same cost cap.

Expected total cost is about `$0.17`; a `$0.50` cap is sufficient. After that,
the next research move is to compare this paired ablation against either a small
human panel or a different model family.

## Reproducibility

Configuration:

- `experiments/event_driven_congestion_pricing_gpt54_mini_memory_on_pilot.json`

Artifacts:

- `runs/event_driven_congestion_pricing_gpt54_mini_memory_on_20260620/config.json`
- `runs/event_driven_congestion_pricing_gpt54_mini_memory_on_20260620/metrics.json`
- `runs/event_driven_congestion_pricing_gpt54_mini_memory_on_20260620/llm_decisions.jsonl`
- `runs/event_driven_congestion_pricing_gpt54_mini_memory_on_20260620/memories.jsonl`
- `runs/event_driven_congestion_pricing_gpt54_mini_memory_on_20260620/retrievals.jsonl`
- `runs/event_driven_congestion_pricing_gpt54_mini_memory_on_20260620/messages.jsonl`
- `runs/event_driven_congestion_pricing_gpt54_mini_memory_on_20260620/agent_states.jsonl`
