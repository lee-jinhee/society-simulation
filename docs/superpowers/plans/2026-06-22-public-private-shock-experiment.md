# Public-Private Shock Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a small event-shock experiment that measures divergence between private belief, public expression, and willingness to speak.

**Architecture:** Extend the existing event-driven opinion simulator rather than creating a new runner. Add state fields for speak willingness, perceived majority, fairness concern, official trust, and silence reason; thread them through parser, prompt, metrics, replay, and a new experiment config.

**Tech Stack:** Python 3.11, dataclasses, pytest, JSON experiment configs, existing OpenAI-compatible policy client.

---

## File Structure

- Modify `src/society_simulation/event_models.py` to extend `EventAgentState`.
- Modify `src/society_simulation/event_policy.py` to parse/prompt/audit the new fields and make silence explicit.
- Modify `src/society_simulation/event_metrics.py` to compute public/private/silence metrics.
- Modify `src/society_simulation/cli.py` and `src/society_simulation/event_replay.py` to surface key metrics.
- Add `experiments/public_private_shock_gpt54_mini_pilot.json`.
- Add or update tests in `tests/test_event_models.py`, `tests/test_event_policy.py`, and `tests/test_event_metrics.py`.
- After the run, add English and Korean reports under `docs/research/`.

### Task 1: Add Social-Expression State Fields

**Files:**
- Modify: `tests/test_event_models.py`
- Modify: `src/society_simulation/event_models.py`

- [ ] **Step 1: Write the failing tests**

Add expectations to `_valid_agent_state_kwargs()`:

```python
"willingness_to_speak": 0.42,
"perceived_majority": -0.1,
"fairness_concern": 0.73,
"trust_in_official_info": 0.38,
"silence_reason": "Waiting to see whether exemptions are real.",
```

Extend the invalid field parametrization:

```python
("willingness_to_speak", -0.1, "willingness_to_speak must be a number between 0 and 1"),
("perceived_majority", 1.1, "perceived_majority must be a number between -1 and 1"),
("fairness_concern", 1.1, "fairness_concern must be a number between 0 and 1"),
("trust_in_official_info", -0.1, "trust_in_official_info must be a number between 0 and 1"),
```

Add `silence_reason` to the empty-string validation.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --extra dev pytest tests/test_event_models.py -q
```

Expected: failures showing `EventAgentState` does not accept or validate the new fields.

- [ ] **Step 3: Implement the state fields**

Add the fields to `EventAgentState`, validate them in `__post_init__`, and set initial defaults in `EventAgentProfile.initial_state()`:

```python
willingness_to_speak=0.5,
perceived_majority=0.0,
fairness_concern=0.3,
trust_in_official_info=self.political_trust,
silence_reason="not_silent",
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run --extra dev pytest tests/test_event_models.py -q
```

Expected: all event model tests pass.

### Task 2: Extend LLM Decision Contract

**Files:**
- Modify: `tests/test_event_policy.py`
- Modify: `src/society_simulation/event_policy.py`

- [ ] **Step 1: Write the failing tests**

Update `llm_decision_content()` and parse tests so every LLM response includes:

```python
"willingness_to_speak": 0.25,
"perceived_majority": -0.15,
"fairness_concern": 0.8,
"trust_in_official_info": 0.35,
"silence_reason": "I am not sure the group wants to hear another complaint.",
```

Add assertions that parsed state and audit records include these fields. Add a
prompt assertion that the policy text contains `zero messages is allowed`,
`willingness_to_speak`, `perceived_majority`, `fairness_concern`,
`trust_in_official_info`, and `silence_reason`.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --extra dev pytest tests/test_event_policy.py -q
```

Expected: failures from missing required fields and prompt text.

- [ ] **Step 3: Implement parser, prompt, mock policy, and audit fields**

Update `_REQUIRED_DECISION_FIELDS`, `parse_event_decision_content()`,
`MockPersonaPolicy._response_content()`, `_event_user_message()`, and
`_event_audit_record()`.

Use deterministic mock defaults:

```python
"willingness_to_speak": 0.7 if messages else 0.2,
"perceived_majority": public_stance,
"fairness_concern": 0.6 if delta < 0 else 0.35,
"trust_in_official_info": profile.political_trust,
"silence_reason": "not_silent" if messages else "Not enough new information to post.",
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run --extra dev pytest tests/test_event_policy.py -q
```

Expected: all event policy tests pass.

### Task 3: Add Public/Private/Silence Metrics

**Files:**
- Modify: `tests/test_event_metrics.py`
- Modify: `src/society_simulation/event_metrics.py`
- Modify: `src/society_simulation/cli.py`
- Modify: `src/society_simulation/event_replay.py`

- [ ] **Step 1: Write the failing metrics tests**

Extend the `state()` helper with the new fields and add assertions for:

```python
mean_willingness_to_speak
silent_agent_count
silent_agent_rate
mean_perceived_majority
perceived_majority_error
mean_fairness_concern
mean_trust_in_official_info
public_expression_bias
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --extra dev pytest tests/test_event_metrics.py -q
```

Expected: missing metric keys.

- [ ] **Step 3: Implement metrics and summaries**

Compute the new fields per day in `compute_event_timeseries()`, expose final
values in `compute_event_metrics()`, print the most important values in the CLI,
and include them in replay `summary.md`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run --extra dev pytest tests/test_event_metrics.py -q
```

Expected: all event metric tests pass.

### Task 4: Add Public-Private Shock Experiment Config

**Files:**
- Add: `experiments/public_private_shock_gpt54_mini_pilot.json`
- Test manually with: `uv run --extra dev society-sim run experiments/public_private_shock_gpt54_mini_pilot.json`

- [ ] **Step 1: Create the experiment config**

Copy the eight personas and relationship graph from the congestion-pricing pilot.
Use a sharper event sequence:

1. official announcement;
2. worker/family hardship story;
3. health/traffic benefit report;
4. viral fairness rumor;
5. official correction;
6. policy concession on exemptions and discounts;
7. public neighborhood hearing prompt.

Use memory retrieval enabled with limit `3`, GPT-5.4 mini, temperature `0.0`,
`max_completion_tokens` around `900`, and cost cap `0.30`.

- [ ] **Step 2: Validate the config with mock first**

Temporarily inspect by loading the config through existing tests or CLI parsing.
Do not run the paid model until the test suite passes.

### Task 5: Verify, Run Paid Pilot, and Write Reports

**Files:**
- Add: `docs/research/2026-06-22-public-private-shock-gpt54-mini-pilot.md`
- Add: `docs/research/2026-06-22-public-private-shock-gpt54-mini-pilot.ko.md`

- [ ] **Step 1: Run full tests**

Run:

```bash
uv run --extra dev pytest
```

Expected: all tests pass.

- [ ] **Step 2: Run the paid pilot**

Run with the existing local environment loaded without printing secrets:

```bash
set -a; source /home/jhlee/repo/.env; set +a; uv run --extra dev society-sim run experiments/public_private_shock_gpt54_mini_pilot.json
```

Expected: run completes below the configured cost cap or stops before exceeding it.

- [ ] **Step 3: Analyze metrics and traces**

Read `metrics.json`, `agent_states.jsonl`, `messages.jsonl`, and
`llm_decisions.jsonl` from the run output. Extract the public/private gap,
silent-agent rate, perceived-majority error, fairness concern, official trust,
and a few concrete examples of private reasoning versus public speech.

- [ ] **Step 4: Write English and Korean reports**

Write reports as research notes. Do not mention credential variables or API-key
plumbing. Include objective, method, cost, results, interpretation, limitations,
and next experiment.

- [ ] **Step 5: Final verification**

Run:

```bash
uv run --extra dev pytest
```

Expected: all tests pass after reports and configs are added.
