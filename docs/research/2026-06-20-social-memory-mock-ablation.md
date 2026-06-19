# Social Memory Mock Ablation

Date: 2026-06-20

## Purpose

This no-cost ablation tested whether the new social memory layer produces
auditable memory and retrieval traces in the event-driven congestion-pricing
scenario.

This was not intended to test whether memory changes opinion dynamics. The
current deterministic mock persona policy accepts retrieved memories in the same
interface as the LLM policy, but its decision rule does not semantically use
those memories.

## Method

We ran the same `event_driven_congestion_pricing` scenario twice:

1. `memory_retrieval.enabled = false`
2. `memory_retrieval.enabled = true`, `limit = 3`

Both runs used the deterministic mock persona policy, so no external model calls
or API costs were incurred.

Artifacts:

- `runs/event_driven_congestion_pricing_mock_memory_off_20260620`
- `runs/event_driven_congestion_pricing_mock_memory_on_20260620`

## Results

| metric | memory off | memory on |
| --- | ---: | ---: |
| final private stance mean | -0.5375 | -0.5375 |
| final public stance mean | -0.24375 | -0.24375 |
| final private-public gap | 0.29375 | 0.29375 |
| message count | 56 | 56 |
| memory count | 0 | 504 |
| retrieval count | 0 | 56 |
| mean retrieved memories per decision | 0.0 | 2.5714 |
| mean retrieval score | 0.0 | 0.6076 |
| private memory count | 0 | 56 |
| public memory count | 0 | 448 |

Memory kind counts in the memory-on run:

| memory kind | count |
| --- | ---: |
| event exposure | 56 |
| self message | 56 |
| self reasoning | 56 |
| social message | 336 |

Retrieved memory kind counts:

| retrieved kind | count |
| --- | ---: |
| event exposure | 8 |
| self message | 88 |
| self reasoning | 48 |

Day 1 has no prior memories to retrieve. From day 2 onward, each of the eight
agents receives up to three retrieved memories per decision.

## Interpretation

The memory layer is operational: it stores memories, retrieves them, writes
`memories.jsonl` and `retrievals.jsonl`, and records memory metrics.

The mock ablation also exposes a limitation. The highest-scoring retrieved
memories are often generic mock self-messages such as:

> I am weighing the benefits and costs before deciding where I land.

This is useful as a diagnostic. It means the memory infrastructure works, but the
deterministic mock persona is too shallow to evaluate social memory as a
behavioral mechanism. The next meaningful test requires an LLM policy that can
semantically use recalled memories.

## Cost Estimate for the Next LLM Ablation

The mock prompt-token estimate increased from 48,201 tokens to 50,969 tokens
when retrieval was enabled, a 5.74% increase.

Using the previous GPT-5.4 mini paid run as the memory-off reference:

- previous memory-off LLM run: 59,614 prompt tokens, 8,487 completion tokens,
  estimated cost `$0.082902`;
- estimated memory-on run: about 63,037 prompt tokens and similar completion
  tokens, estimated cost `$0.08547`;
- fresh memory off/on pair estimate: about `$0.16837`.

Pricing uses OpenAI's GPT-5.4 mini model page as checked on 2026-06-20: `$0.75`
per 1M input tokens and `$4.50` per 1M output tokens.
Source: <https://developers.openai.com/api/docs/models/gpt-5.4-mini>

## Next Step

Run a paid GPT-5.4 mini memory-on pilot with a conservative cost cap. The first
paid memory test can reuse the prior memory-off pilot as a reference, but a
cleaner ablation should rerun both memory-off and memory-on under the current
code.

Recommended cap:

- memory-on only: `$0.20`
- fresh off/on pair: `$0.50`

The publishable research question is not whether memory produces more traces.
The question is whether social memory retrieval changes public/private stance,
message content, resistance, and social convergence in ways that are more
consistent with human opinion dynamics.
