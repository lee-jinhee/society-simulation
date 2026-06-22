# Speech Decision Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class speech decision so agents can publicly post, privately message, read silently, or avoid the discussion.

**Architecture:** Extend `EventAgentState` with `speech_action`, require it in LLM/mock decisions, validate message consistency in the event runner, and expose action-count metrics. Keep the existing runner and replay structure; this is a schema-and-validation layer, not a new simulator.

**Tech Stack:** Python 3.11, dataclasses, pytest, JSON configs, existing event-driven opinion runner.

---

## File Structure

- Modify `src/society_simulation/event_models.py` for `speech_action` validation and initial state.
- Modify `src/society_simulation/event_policy.py` for parsing, prompt, mock response, and audit fields.
- Modify `src/society_simulation/event_runner.py` for message/action consistency validation.
- Modify `src/society_simulation/event_metrics.py`, `src/society_simulation/cli.py`, and `src/society_simulation/event_replay.py` for metrics and summaries.
- Modify tests in `tests/test_event_models.py`, `tests/test_event_policy.py`, `tests/test_event_runner.py`, and `tests/test_event_metrics.py`.
- Update `experiments/public_private_shock_gpt54_mini_pilot.json` only if the prompt contract requires larger completion budget.

### Task 1: State and Parser Contract

- [ ] Add failing model tests expecting `speech_action` in `EventAgentState`, rejecting unsupported values.
- [ ] Run `../../.venv/bin/python -m pytest tests/test_event_models.py -q` and verify failure.
- [ ] Add `speech_action` to `EventAgentState`, validate it against the allowed values, and default initial state to `read_only`.
- [ ] Run `../../.venv/bin/python -m pytest tests/test_event_models.py -q` and verify pass.
- [ ] Add failing policy tests requiring `speech_action` in LLM JSON, audit records, and prompt text.
- [ ] Run `../../.venv/bin/python -m pytest tests/test_event_policy.py -q` and verify failure.
- [ ] Add `speech_action` to `_REQUIRED_DECISION_FIELDS`, parser, mock response, prompt, and audit record.
- [ ] Run `../../.venv/bin/python -m pytest tests/test_event_policy.py -q` and verify pass.

### Task 2: Runner Consistency Validation

- [ ] Add failing runner tests:
  - `public_post` with zero messages fails.
  - `public_post` with a private recipient fails.
  - `private_message` with public recipient fails.
  - `read_only` with a message fails.
  - `avoid_discussion` with a message fails.
- [ ] Run `../../.venv/bin/python -m pytest tests/test_event_runner.py -q` and verify failure.
- [ ] Implement `_validate_speech_action_messages()` and call it after generic message/state validation.
- [ ] Run `../../.venv/bin/python -m pytest tests/test_event_runner.py -q` and verify pass.

### Task 3: Speech Action Metrics

- [ ] Add failing metrics tests for `speech_action_counts`, final action counts, and final action rates.
- [ ] Run `../../.venv/bin/python -m pytest tests/test_event_metrics.py -q` and verify failure.
- [ ] Implement metrics in `event_metrics.py` and surface key values in CLI/replay summary.
- [ ] Run `../../.venv/bin/python -m pytest tests/test_event_metrics.py tests/test_cli.py tests/test_event_replay.py -q` and verify pass.

### Task 4: Full Verification and Documentation

- [ ] Run `../../.venv/bin/python -m pytest`.
- [ ] Update the public-private shock experiment config only if needed for the prompt contract.
- [ ] Add a short research note or amend the existing report's next-step section if useful.
- [ ] Run `../../.venv/bin/python -m pytest` again.
- [ ] Commit, merge to main, push.
