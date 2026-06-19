# LLM Decision Audit Trail Design

Date: 2026-06-19

## Status

Approved direction: build the LLM decision audit trail before adding richer model sweeps.

This design adds per-decision trace artifacts for mock and real LLM network policies. The existing `llm_usage` metrics answer "how many calls and how much estimated cost"; this audit trail answers "what did each simulated agent see, what did the model return, and what action did the simulator use?"

## Goal

Write one JSONL record for every successful LLM-backed agent decision in a network run.

The artifact should make LLM experiments inspectable enough to support:

- debugging malformed or surprising model behavior;
- replaying how individual agents responded to local social context;
- estimating cost at the call level;
- comparing mock LLM runs with real provider runs;
- building later analysis around prompt sensitivity and crowd dynamics.

## Non-Goals

The first audit trail will not include:

- failed-call retry logs;
- streaming response traces;
- provider request headers;
- API keys or environment variable values;
- tokenization using provider-specific tokenizers;
- prompt redaction or privacy filtering;
- a separate database-backed trace store.

Those can be added later if the project needs them. The first version should be a deterministic file artifact beside the existing replay bundle.

## Artifact

LLM-enabled network runs write:

```text
llm_decisions.jsonl
```

Each line is one JSON object. Non-LLM policies do not write this file.

Required fields:

- `agent_id`: integer simulated agent id.
- `round_index`: integer update round.
- `provider`: provider family, such as `mock` or `openai_compatible`.
- `model`: model id used by the policy.
- `policy_type`: simulator update policy type, such as `mock_llm` or `llm`.
- `prompt`: exact prompt text submitted to the mock provider or the joined chat messages for a real provider.
- `raw_response`: provider response payload. For mock runs this is a small object containing the mock response content; for real provider runs this is the returned JSON object.
- `parsed_action`: validated simulator action, `A` or `B`.
- `parsed_belief_probability`: validated belief probability in `[0, 1]`.
- `confidence`: simulator confidence derived from the parsed belief.
- `prompt_tokens`: usage prompt tokens when the provider reports them, otherwise the existing deterministic estimate.
- `completion_tokens`: usage completion tokens when the provider reports them, otherwise the existing deterministic estimate.
- `input_cost_usd`: call-level estimated input cost.
- `output_cost_usd`: call-level estimated output cost.
- `total_cost_usd`: sum of call-level input and output cost.
- `latency_ms`: elapsed wall-clock milliseconds for the successful decision path.

Real chat-provider records may also include:

- `messages`: the chat messages sent to the provider.

## Security

The audit trail must never write:

- API keys;
- authorization headers;
- local environment variable names or values;
- provider-specific secret config.

`raw_response` is still potentially sensitive because provider responses can echo prompt content. The artifact should be treated as experiment data, not a public log.

## Integration

Policies expose an optional `audit_records()` method. The network runner discovers this method with duck typing, just as it already discovers `usage_summary()`.

`NetworkReplayWriter.write(...)` accepts optional `llm_decisions`. If records are present, it writes `llm_decisions.jsonl`; otherwise it preserves the current artifact set.

This keeps no-LLM policies unchanged and avoids threading LLM-specific concepts through every policy interface.

## Testing

Unit tests should cover:

- mock LLM policies record one audit row per decision;
- OpenAI-compatible policies record raw provider responses, parsed output, cost, and tokens using fake transport only;
- returned audit records are defensive copies;
- replay writer writes `llm_decisions.jsonl` when records are supplied;
- non-LLM runs do not write the file;
- mock LLM network runs write one row per agent per update round.
