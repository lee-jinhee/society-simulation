# Visible Endorsement Cascade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a mock-complete Instagram experiment where the same seed post can be shown with different visible like counts, so endorsement-cascade behavior can be measured without paid LLM calls.

**Architecture:** `InstagramSocialDynamicsConfig` gains optional `seed_posts`. The seed generator injects those posts into the initial world, the feed carries visible post metadata, and the mock policy can react to `visible_like_count` instead of only abstract feed score. A new sweep config varies seed-post like count and reuses the existing analyzer.

**Tech Stack:** Python dataclasses, existing social-media runner, pytest, JSON sweep configs.

---

### Task 1: Config and Seed Post Injection

**Files:**
- Modify: `src/society_simulation/social_media_config.py`
- Modify: `src/society_simulation/social_media_seed.py`
- Modify: `src/society_simulation/sweep_config.py`
- Test: `tests/test_social_media_config.py`
- Test: `tests/test_social_media_seed.py`

- [ ] **Step 1: Write failing config tests**

Add tests that default `seed_posts` to `[]`, accept a valid seed post, reject invalid author IDs and negative like counts.

- [ ] **Step 2: Run tests to verify RED**

Run: `PYTHONPATH=src /home/jhlee/repo/society-simulation/.venv/bin/python -m pytest tests/test_social_media_config.py tests/test_social_media_seed.py -q`

Expected: failures because `seed_posts` is not exposed on the config and not injected into the world.

- [ ] **Step 3: Implement config and seed injection**

Add a frozen `seed_posts` tuple to `InstagramSocialDynamicsConfig`, validate post fields, include it in `to_dict()`, allow it in sweep configs, and append `SocialMediaPost(..., seed_post=True)` objects in `build_initial_world()`.

- [ ] **Step 4: Run tests to verify GREEN**

Run the same targeted tests and expect them to pass.

### Task 2: Visible Feed Metadata

**Files:**
- Modify: `src/society_simulation/social_media_models.py`
- Modify: `src/society_simulation/social_media_feed.py`
- Modify: `src/society_simulation/social_media_policy.py`
- Test: `tests/test_social_media_feed.py`
- Test: `tests/test_social_media_policy.py`

- [ ] **Step 1: Write failing feed/prompt tests**

Add a feed test asserting `FeedItem.visible_like_count` matches the source post, and a prompt test asserting visible likes appear in the natural user-facing feed description.

- [ ] **Step 2: Run tests to verify RED**

Run: `PYTHONPATH=src /home/jhlee/repo/society-simulation/.venv/bin/python -m pytest tests/test_social_media_feed.py tests/test_social_media_policy.py -q`

Expected: failures because `FeedItem` lacks visible metadata.

- [ ] **Step 3: Implement feed metadata**

Extend `FeedItem` with trailing default fields: `visible_like_count`, `topic`, `text`, and `author_handle`. Populate them in `build_feed()` and include them in `build_social_media_prompt()`.

- [ ] **Step 4: Run tests to verify GREEN**

Run the same targeted tests and expect them to pass.

### Task 3: Endorsement-Sensitive Mock Policy

**Files:**
- Modify: `src/society_simulation/social_media_policy.py`
- Test: `tests/test_social_media_policy.py`

- [ ] **Step 1: Write failing behavior tests**

Add tests showing high visible likes trigger `like_post` for `endorsement_sensitive`, while a low-like, low-score item does not.

- [ ] **Step 2: Run tests to verify RED**

Run: `PYTHONPATH=src /home/jhlee/repo/society-simulation/.venv/bin/python -m pytest tests/test_social_media_policy.py -q`

Expected: low/high visible endorsement is not distinguished yet.

- [ ] **Step 3: Implement minimal endorsement rule**

Use a simple threshold based on visible likes and persona traits: conformity/status lower the threshold, skepticism raises it. Keep existing high-score behavior so old tests remain valid.

- [ ] **Step 4: Run tests to verify GREEN**

Run the same targeted tests and expect them to pass.

### Task 4: Experiment Config and Smoke Run

**Files:**
- Create: `experiments/instagram_visible_endorsement_sweep.json`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing experiment validation test**

Add a CLI/config test asserting the new sweep config parses and materializes low/moderate/high endorsement runs.

- [ ] **Step 2: Run tests to verify RED**

Run: `PYTHONPATH=src /home/jhlee/repo/society-simulation/.venv/bin/python -m pytest tests/test_cli.py -q`

Expected: missing experiment file or unknown `seed_posts`.

- [ ] **Step 3: Add the sweep config**

Create a sweep with factors `seed` and `visible_likes`, where the `visible_likes` factor overrides the same seed post with `like_count` values such as `0`, `20`, and `80`.

- [ ] **Step 4: Run smoke experiment and analyzer**

Run:

```bash
PYTHONPATH=src /home/jhlee/repo/society-simulation/.venv/bin/python -m society_simulation sweep experiments/instagram_visible_endorsement_sweep.json
PYTHONPATH=src /home/jhlee/repo/society-simulation/.venv/bin/python -m society_simulation analyze runs/sweeps/instagram_visible_endorsement_sweep
```

Expected: all mock runs complete and `analysis/report.md` contains social metrics grouped by `visible_likes`.

### Task 5: Verification and Integration

**Files:**
- Commit all touched files after verification.

- [ ] **Step 1: Run full tests**

Run: `PYTHONPATH=src /home/jhlee/repo/society-simulation/.venv/bin/python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Inspect report**

Read `runs/sweeps/instagram_visible_endorsement_sweep/analysis/report.md` and summarize whether high visible endorsement changed likes/actions relative to low endorsement.

- [ ] **Step 3: Commit, merge, and push**

Commit message: `feat: add visible endorsement social sweep`.

Merge to `main` with fast-forward, rerun full tests on main, and push `origin main`.
