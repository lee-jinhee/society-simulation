# LLM Policy Baseline Design

## Goal

Add a first LLM-shaped policy path for network herding experiments without making paid API calls. The result should let us run a tiny "LLM-style" social simulation, estimate token/cost exposure, and keep the existing no-LLM baselines intact.

## Scope

This phase implements a deterministic `mock_llm` network update policy. It does not call OpenAI, Chinese model APIs, or any network service. Real providers will be added after the mock path proves the policy interface, replay artifacts, metrics, and cost accounting work.

## Approach

`mock_llm` fits into the existing `NetworkUpdatePolicyConfig` and `build_network_update_policy()` flow. The policy builds a prompt from each agent's current state and observed neighbor state, sends it to a provider interface, validates the returned action and belief, and accumulates estimated usage. The mock provider is deterministic and cheap: it can emulate neighbor-majority, current-action, or contrarian behavior while exercising the same request/response and accounting path that a real provider will use.

## Components

### `llm_policy.py`

Owns the provider-shaped API:

- `LLMPricing` stores input/output price per one million tokens.
- `LLMUsage` stores calls, prompt tokens, completion tokens, and estimated USD cost.
- `estimate_tokens(text)` uses a simple deterministic approximation, `ceil(len(text) / 4)`.
- `MockLLMProvider` returns a validated action and belief from a `NetworkObservation`.
- `MockLLMPolicy` implements `decide(observation)` and exposes `usage_summary()`.

### Config

`NetworkUpdatePolicyConfig` accepts:

- `type: "mock_llm"`;
- optional `provider`, default `"mock"`;
- optional `model`, default `"mock-neighbor-majority"`;
- optional `response_style`, default `"neighbor_majority"`;
- optional `input_cost_per_1m_tokens`, default `0.0`;
- optional `output_cost_per_1m_tokens`, default `0.0`.

Only mock providers are accepted in this phase. Real provider names should fail cleanly until an implementation exists.

### Runner Metrics

`run_network_herding()` adds `llm_usage` to `metrics` when the selected policy exposes usage. This keeps replay artifacts and sweep summaries compatible: existing summary fields remain unchanged, while full run metrics preserve cost accounting for LLM-style experiments.

### CLI

For single `run` commands, if `metrics["llm_usage"]` exists, print:

- `llm_calls=...`
- `llm_prompt_tokens=...`
- `llm_completion_tokens=...`
- `llm_estimated_cost_usd=...`

Existing `run`, `sweep`, and `analyze` output for non-LLM configs must not change.

## Testing

Tests should cover:

- config parsing, validation, serialization, and rejection of unsupported real providers;
- mock provider decisions for neighbor-majority/current/contrarian styles;
- usage and cost accumulation;
- runner metrics containing `llm_usage` for `mock_llm`;
- CLI output for a tiny mock LLM config;
- full-suite regression.

## Non-Goals

- No real API clients.
- No API keys or environment variables.
- No model-specific current pricing table.
- No prompt engineering research yet.
- No sweep-scale LLM runs.
