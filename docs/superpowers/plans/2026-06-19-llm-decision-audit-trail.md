# LLM Decision Audit Trail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for code changes. Use superpowers:verification-before-completion before claiming completion or merging.

**Goal:** Add a per-decision `llm_decisions.jsonl` artifact for LLM-backed network policies.

**Architecture:** LLM policies collect successful decision audit rows. The network runner pulls those rows through an optional `audit_records()` method and passes them to the replay writer. The replay writer writes a JSONL artifact only when records are present.

**Tech Stack:** Python 3.11+, standard library only at runtime, `pytest` for tests.

---

## File Structure

- Modify `src/society_simulation/llm_policy.py`
  - Add call-level audit record collection for `MockLLMPolicy`.
  - Add call-level audit record collection for `OpenAICompatibleLLMPolicy`.
  - Return defensive copies from `audit_records()`.

- Modify `src/society_simulation/network_runner.py`
  - Collect optional policy audit records after a run.
  - Pass them to `NetworkReplayWriter.write(...)`.

- Modify `src/society_simulation/network_replay.py`
  - Accept optional `llm_decisions`.
  - Validate the required audit fields.
  - Write `llm_decisions.jsonl` only when records are present.

- Modify `tests/test_llm_policy.py`
  - Cover mock and OpenAI-compatible policy audit rows.

- Modify `tests/test_network_replay.py`
  - Cover optional audit artifact writing and missing-field validation.

- Modify `tests/test_network_runner.py`
  - Cover mock LLM audit artifact line count and non-LLM absence.

- Modify `README.md`
  - Document the new audit artifact.

---

## Steps

- [x] Step 1: Add failing policy tests for mock and OpenAI-compatible audit rows.
- [x] Step 2: Add failing replay writer and runner tests for `llm_decisions.jsonl`.
- [x] Step 3: Implement policy audit record collection and defensive copies.
- [x] Step 4: Implement runner handoff and replay writer artifact output.
- [x] Step 5: Document the artifact in `README.md`.
- [x] Step 6: Run focused tests, then the full suite.
- [ ] Step 7: Commit, merge to `main`, and push.
