# LLM Policy Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic mock LLM policy path with usage and cost accounting, without making paid API calls.

**Architecture:** Extend the existing network update policy factory with `mock_llm`. Keep provider-shaped request/response and usage accounting in a focused `llm_policy.py` module. Preserve existing non-LLM config, runner, CLI, sweep, and analyzer behavior.

**Tech Stack:** Python 3.11+, standard library only at runtime, `pytest` for tests.

---

## File Structure

- Create `src/society_simulation/llm_policy.py`
  - Token estimation, pricing, mock provider, and mock LLM policy.
- Modify `src/society_simulation/config.py`
  - Parse and validate `mock_llm` policy fields.
- Modify `src/society_simulation/network_policies.py`
  - Add `MockLLMPolicy` to the update policy union and factory.
- Modify `src/society_simulation/network_runner.py`
  - Attach `llm_usage` metrics when a policy exposes usage.
- Modify `src/society_simulation/cli.py`
  - Print LLM usage lines only when present.
- Create `examples/network_herding_mock_llm.json`
  - Tiny local mock LLM example.
- Create `tests/test_llm_policy.py`
  - Unit tests for decisions, validation, and cost accounting.
- Modify `tests/test_network_config.py`
  - Config parse/validation/round-trip tests.
- Modify `tests/test_network_policies.py`
  - Factory support tests.
- Modify `tests/test_network_runner.py`
  - Runner metrics test.
- Modify `tests/test_cli.py`
  - CLI output test for mock LLM usage.
- Modify `README.md`
  - Document the mock LLM smoke path and no-cost status.

---

### Task 1: Mock LLM Policy Core

**Files:**
- Create: `src/society_simulation/llm_policy.py`
- Create: `tests/test_llm_policy.py`

- [ ] **Step 1: Write failing tests**

Create tests for:

```python
def test_estimate_tokens_uses_deterministic_character_approximation() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
```

```python
def test_mock_llm_policy_decides_neighbor_majority_and_tracks_usage() -> None:
    policy = MockLLMPolicy(response_style="neighbor_majority")
    decision = policy.decide(observation(("A", "A", "B"), (1.0, 1.0, 0.0)))
    assert decision.action == "A"
    assert decision.belief_probability == pytest.approx(2 / 3)
    usage = policy.usage_summary()
    assert usage["calls"] == 1
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_cost_usd"] == 0.0
```

```python
def test_mock_llm_policy_estimates_cost() -> None:
    policy = MockLLMPolicy(
        response_style="current",
        input_cost_per_1m_tokens=1.0,
        output_cost_per_1m_tokens=2.0,
    )
    policy.decide(observation((), (), current_action="B", current_belief=0.2))
    usage = policy.usage_summary()
    assert usage["total_cost_usd"] > 0
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_llm_policy.py -v
```

Expected: import failure because `society_simulation.llm_policy` does not exist.

- [ ] **Step 3: Implement `llm_policy.py`**

Implement:

- `estimate_tokens(text: str) -> int`
- `LLMPricing`
- `LLMUsage`
- `MockLLMPolicy`
- decision validation using existing `NetworkObservation` and `NetworkDecision`

- [ ] **Step 4: Run tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/test_llm_policy.py -v
```

Commit:

```bash
git add src/society_simulation/llm_policy.py tests/test_llm_policy.py
git commit -m "feat: add mock llm policy core"
```

---

### Task 2: Config and Policy Factory Integration

**Files:**
- Modify: `src/society_simulation/config.py`
- Modify: `src/society_simulation/network_policies.py`
- Modify: `tests/test_network_config.py`
- Modify: `tests/test_network_policies.py`

- [ ] **Step 1: Write failing tests**

Add tests that:

- `mock_llm` config parses and round-trips;
- unsupported real providers fail;
- irrelevant threshold/degroot fields are rejected for `mock_llm`;
- `build_network_update_policy()` returns `MockLLMPolicy`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_network_config.py tests/test_network_policies.py -v
```

Expected: failures because `mock_llm` is unsupported.

- [ ] **Step 3: Implement integration**

Extend `NetworkUpdatePolicyConfig` with mock LLM fields and validation. Import and return `MockLLMPolicy` in `build_network_update_policy()`.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/test_network_config.py tests/test_network_policies.py tests/test_llm_policy.py -v
```

Commit:

```bash
git add src/society_simulation/config.py src/society_simulation/network_policies.py tests/test_network_config.py tests/test_network_policies.py
git commit -m "feat: wire mock llm policy config"
```

---

### Task 3: Runner Metrics, CLI, Example, and Docs

**Files:**
- Modify: `src/society_simulation/network_runner.py`
- Modify: `src/society_simulation/cli.py`
- Create: `examples/network_herding_mock_llm.json`
- Modify: `tests/test_network_runner.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing tests**

Add tests that:

- running a `mock_llm` network config stores `llm_usage` metrics;
- CLI prints `llm_calls`, `llm_prompt_tokens`, `llm_completion_tokens`, and `llm_estimated_cost_usd`;
- existing non-LLM CLI outputs remain unchanged.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_network_runner.py tests/test_cli.py -v
```

Expected: mock LLM metrics/CLI tests fail.

- [ ] **Step 3: Implement runner and CLI output**

If a policy exposes `usage_summary()`, add it to final metrics. In CLI, print LLM usage lines only when `metrics["llm_usage"]` exists.

- [ ] **Step 4: Add example and README**

Create `examples/network_herding_mock_llm.json` with a tiny 5-agent, 3-round mock LLM run. README should state that this path costs `0` API dollars because it uses the deterministic mock provider.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/test_network_runner.py tests/test_cli.py tests/test_llm_policy.py -v
./.venv/bin/python -m society_simulation run examples/network_herding_mock_llm.json
```

Commit:

```bash
git add src/society_simulation/network_runner.py src/society_simulation/cli.py examples/network_herding_mock_llm.json tests/test_network_runner.py tests/test_cli.py README.md
git commit -m "feat: expose mock llm usage workflow"
```

---

### Task 4: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused tests**

```bash
./.venv/bin/python -m pytest tests/test_llm_policy.py tests/test_network_config.py tests/test_network_policies.py tests/test_network_runner.py tests/test_cli.py -v
```

- [ ] **Step 2: Run full suite**

```bash
./.venv/bin/python -m pytest -v
```

- [ ] **Step 3: Run mock LLM example**

```bash
./.venv/bin/python -m society_simulation run examples/network_herding_mock_llm.json
```

Expected output includes `llm_calls=`, `llm_prompt_tokens=`, `llm_completion_tokens=`, and `llm_estimated_cost_usd=`.

- [ ] **Step 4: Confirm git status**

```bash
git status --short
```

Expected: no tracked files modified; ignored `runs/` artifacts may exist.
