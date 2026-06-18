# OpenAI-Compatible LLM Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cost-aware OpenAI-compatible LLM provider path for network herding simulations.

**Architecture:** Extend `llm_policy.py` with a standard-library HTTP transport and `OpenAICompatibleLLMPolicy`. Extend `NetworkUpdatePolicyConfig` and `build_network_update_policy()` so configs can select `type: "llm"` with provider, endpoint, key env var, model, token limit, timeout, pricing, and budget cap. Preserve `mock_llm` and no-LLM behavior.

**Tech Stack:** Python 3.11+, standard library `urllib.request`, `json`, `os`; `pytest` with fake transports for all provider tests.

---

## File Structure

- Modify `src/society_simulation/llm_policy.py`
  - Add reusable prompt helpers, response parsing, HTTP transport, OpenAI-compatible client, and real LLM policy.
- Modify `src/society_simulation/config.py`
  - Add real LLM provider config fields and validation.
- Modify `src/society_simulation/network_policies.py`
  - Return `OpenAICompatibleLLMPolicy` for `update_policy.type == "llm"`.
- Modify `tests/test_llm_policy.py`
  - Add fake-transport unit tests for request shape, parsing, usage, fallback usage, missing keys, malformed responses, and budget caps.
- Modify `tests/test_network_config.py`
  - Add parsing, round-trip, validation, and irrelevant-field tests for real LLM config.
- Modify `tests/test_network_policies.py`
  - Add factory test with env key set.
- Modify `tests/test_cli.py`
  - Add example config validity test.
- Add `examples/network_herding_openai_compatible.json`
  - Runnable template config that requires an env key and explicit model.
- Modify `README.md`
  - Document the real provider path, key setup, cost controls, and no automatic paid calls.

---

### Task 1: Provider Core

**Files:**
- Modify: `src/society_simulation/llm_policy.py`
- Modify: `tests/test_llm_policy.py`

- [ ] **Step 1: Write failing fake-transport tests**

Add tests that instantiate `OpenAICompatibleLLMPolicy` with `api_key="test-key"` and a fake transport:

```python
def test_openai_compatible_policy_sends_chat_completion_request_and_tracks_provider_usage():
    captured = {}

    def transport(url, headers, payload, timeout_seconds):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        captured["timeout_seconds"] = timeout_seconds
        return {
            "choices": [{"message": {"content": "{\"action\":\"A\",\"belief_probability\":0.75}"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }

    policy = OpenAICompatibleLLMPolicy(
        model="cheap-chat",
        api_key="test-key",
        base_url="https://example.test/v1",
        input_cost_per_1m_tokens=1.0,
        output_cost_per_1m_tokens=2.0,
        transport=transport,
    )

    decision = policy.decide(observation(("A", "B"), (1.0, 0.0)))

    assert decision.action == "A"
    assert decision.belief_probability == pytest.approx(0.75)
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "cheap-chat"
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert captured["payload"]["messages"][1]["role"] == "user"
    assert captured["payload"]["max_completion_tokens"] == 32
    usage = policy.usage_summary()
    assert usage["provider"] == "openai_compatible"
    assert usage["calls"] == 1
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 7
    assert usage["total_cost_usd"] == pytest.approx(25 / 1_000_000)
```

Also add tests for missing key, malformed JSON content, fallback token estimates when `usage` is absent, and budget cap failure.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_llm_policy.py -v
```

Expected: import failure or missing class failure for `OpenAICompatibleLLMPolicy`.

- [ ] **Step 3: Implement provider core**

Add:

- `JSONTransport` type alias
- `_urllib_json_transport(url, headers, payload, timeout_seconds)`
- `OpenAICompatibleClient`
- `_parse_llm_decision_content(content)`
- `OpenAICompatibleLLMPolicy`

- [ ] **Step 4: Run tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/test_llm_policy.py -v
```

Commit:

```bash
git add src/society_simulation/llm_policy.py tests/test_llm_policy.py
git commit -m "feat: add openai compatible llm provider"
```

---

### Task 2: Config And Factory Wiring

**Files:**
- Modify: `src/society_simulation/config.py`
- Modify: `src/society_simulation/network_policies.py`
- Modify: `tests/test_network_config.py`
- Modify: `tests/test_network_policies.py`

- [ ] **Step 1: Write failing config and factory tests**

Add tests that:

- `type: "llm"` with `model` parses and validates
- `to_dict()` preserves real LLM fields
- missing `model` fails
- unsupported `provider` fails
- invalid `temperature`, `timeout_seconds`, `max_completion_tokens`, `token_limit_parameter`, and `max_estimated_cost_usd` fail
- no-LLM policies reject real LLM fields
- factory builds `OpenAICompatibleLLMPolicy` when env key is set

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_network_config.py tests/test_network_policies.py -v
```

Expected: failures because `type: "llm"` is unsupported.

- [ ] **Step 3: Implement config and factory**

Extend `NetworkUpdatePolicyConfig` with:

- `base_url`
- `api_key_env`
- `temperature`
- `max_completion_tokens`
- `token_limit_parameter`
- `timeout_seconds`
- `max_estimated_cost_usd`

Wire `build_network_update_policy()` to instantiate `OpenAICompatibleLLMPolicy`.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/test_network_config.py tests/test_network_policies.py tests/test_llm_policy.py -v
```

Commit:

```bash
git add src/society_simulation/config.py src/society_simulation/network_policies.py tests/test_network_config.py tests/test_network_policies.py
git commit -m "feat: wire real llm policy config"
```

---

### Task 3: Example And Docs

**Files:**
- Add: `examples/network_herding_openai_compatible.json`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing example validity test**

Add a test that loads `examples/network_herding_openai_compatible.json` and validates `update_policy.type == "llm"`.

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cli.py::test_example_openai_compatible_network_config_exists_and_is_valid -v
```

Expected: file missing.

- [ ] **Step 3: Add example and README**

Document:

- set env key, for example `export SOCIETY_SIM_LLM_API_KEY=...`
- adjust `base_url`, `model`, and price fields
- use low `num_agents` and `rounds` first
- no paid call is made unless user runs a `type: "llm"` config with a valid key

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cli.py tests/test_network_config.py tests/test_network_policies.py tests/test_llm_policy.py -v
```

Commit:

```bash
git add examples/network_herding_openai_compatible.json README.md tests/test_cli.py
git commit -m "docs: document openai compatible llm runs"
```

---

### Task 4: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run full suite**

```bash
./.venv/bin/python -m pytest -v
```

- [ ] **Step 2: Run no-cost mock smoke**

```bash
./.venv/bin/python -m society_simulation run examples/network_herding_mock_llm.json
```

- [ ] **Step 3: Check git status**

```bash
git status --short
```

Expected: no tracked changes.
