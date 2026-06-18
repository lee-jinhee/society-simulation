# OpenAI-Compatible LLM Provider Design

## Goal

Add a real LLM provider path for network herding experiments while keeping cost and credentials explicit. This phase should make the simulator able to call OpenAI-compatible chat completion APIs, including OpenAI and lower-cost compatible providers, without embedding vendor-specific SDKs or secrets.

## Scope

This phase adds `update_policy.type: "llm"` with `provider: "openai_compatible"`. It does not run paid experiments by default, does not choose a default paid model, and does not store API keys in config or artifacts. Tests use fake transports only.

## API Choice

Use the Chat Completions-compatible `/chat/completions` shape for this phase. OpenAI now recommends the Responses API for new OpenAI-only projects, but the Chat Completions shape is the practical portability layer because many non-OpenAI providers expose OpenAI-compatible chat endpoints.

The policy sends a compact system message and a user message derived from `NetworkObservation`. The assistant is asked to return JSON:

```json
{"action":"A","belief_probability":0.75}
```

The simulator validates the returned action and belief before accepting the decision.

## Config

`NetworkUpdatePolicyConfig` accepts the following for real LLM runs:

- `type: "llm"`
- `provider: "openai_compatible"`; optional, defaults to `"openai_compatible"`
- `model`; required
- `base_url`; optional, defaults to `"https://api.openai.com/v1"`
- `api_key_env`; optional, defaults to `"OPENAI_API_KEY"`
- `temperature`; optional, defaults to `0.0`
- `max_completion_tokens`; optional, defaults to `32`
- `token_limit_parameter`; optional, defaults to `"max_completion_tokens"`, can be `"max_tokens"` for older compatible vendors
- `timeout_seconds`; optional, defaults to `30.0`
- `input_cost_per_1m_tokens`; optional, defaults to `0.0`
- `output_cost_per_1m_tokens`; optional, defaults to `0.0`
- `max_estimated_cost_usd`; optional budget cap

API keys are loaded from `api_key_env` at policy construction time. Missing keys fail the run with a clear error.

## Usage And Cost

The policy records:

- calls
- prompt tokens
- completion tokens
- input/output/total estimated cost
- provider
- model

If the provider returns `usage.prompt_tokens` and `usage.completion_tokens`, those values are used. If usage is absent, the existing deterministic token estimator is used as a fallback. Costs are computed from user-supplied per-1M-token prices, not hardcoded model price tables.

If `max_estimated_cost_usd` is set, the policy checks the prompt-side projected cost before a call and the full estimated cost after a call. This cannot perfectly prevent one unexpectedly expensive response, but it stops additional calls and makes budget failure explicit.

## Runtime Errors

The provider path raises `ValueError` for:

- missing API key env var
- unsupported provider
- malformed HTTP response
- missing `choices[0].message.content`
- non-JSON model content
- invalid `action`
- invalid `belief_probability`
- exceeded cost cap

CLI already reports `ValueError` run failures as clean experiment errors.

## Testing

Tests cover:

- request payload/header construction through a fake transport
- response parsing and decision validation
- provider usage accounting with provider-reported usage
- fallback token accounting when usage is missing
- missing API key error
- malformed model output error
- budget cap error
- config validation and factory construction
- example config validity

## Non-Goals

- No real paid API call in automated tests.
- No SDK dependency.
- No streaming.
- No Responses API support in this phase.
- No current pricing table.
- No sweep-scale paid LLM workflow yet.
