# Instagram Ad Reach Implementation Plan

Date: 2026-06-23

Spec: `docs/superpowers/specs/2026-06-23-instagram-ad-reach-experiment-design.md`

## Goal

Implement the first auditable Instagram-like advertising intervention layer.
The milestone should let us create a stable synthetic Instagram world, inject
no-ad / organic-post / sponsored-ad conditions, record campaign delivery, and
compare paid reach against organic spillover.

This does **not** claim real Instagram prediction. It creates a controlled
counterfactual lab for relative mechanism tests before calibration.

## Execution Notes

- Worktree: `.worktrees/instagram-ad-reach`
- Branch: `feature/instagram-ad-reach`
- Baseline before edits: `528 passed`
- User already approved implementation after this plan, so execution continues
  without a second approval gate.
- TDD rule: write failing tests before implementation for each code task.

## Task 1: Add Campaign Config And Model Primitives

### Intent

Represent ads as explicit simulator state, not hidden prompt text. Campaigns,
ad posts, and sponsored feed cards must be visible in replay artifacts.

### Files

- `src/society_simulation/social_media_config.py`
- `src/society_simulation/social_media_runner.py`
- `tests/test_social_media_config.py`
- `tests/test_social_media_models.py`

### Required Data Model

Add a frozen dataclass:

```python
@dataclass(frozen=True)
class AdCampaignConfig:
    campaign_id: str
    advertiser_id: int
    ad_condition: Literal["no_ad", "organic_post", "sponsored_ad"]
    creative_id: str
    creative_text: str
    topic: str
    stance: float
    start_tick: int
    end_tick: int
    budget_impressions: int
    frequency_cap: int
    targeting: Literal["broad", "interest_targeted"]
    sponsored_like_count: int = 0
    targeting_topics: tuple[str, ...] = ()
```

Extend `InstagramSocialDynamicsConfig`:

```python
ad_campaigns: tuple[AdCampaignConfig, ...] = ()
```

Extend `SocialMediaPost` with trailing default fields:

```python
campaign_id: str | None = None
is_ad: bool = False
```

Extend `FeedItem` with trailing default fields:

```python
campaign_id: str | None = None
is_sponsored: bool = False
advertiser_id: int | None = None
ad_seen_count: int = 0
```

Extend `FeedSource` with:

```python
"sponsored"
```

### Validation Rules

- `campaign_id`, `creative_id`, `creative_text`, and `topic` are non-empty.
- `stance` is a finite number between -1 and 1.
- `advertiser_id` is inside `0 <= advertiser_id < num_users`.
- `topic` is included in `topics`.
- `start_tick >= 1`, `end_tick >= start_tick`, and `end_tick <= ticks`.
- `budget_impressions >= 0` for `no_ad`; `> 0` for `sponsored_ad`.
- `frequency_cap >= 0` for `no_ad`; `> 0` for `sponsored_ad`.
- `sponsored_like_count >= 0`.
- `targeting_topics`, when omitted, defaults to `(topic,)`.

### Tests First

Add tests that initially fail:

- default config has `ad_campaigns == ()`;
- valid sponsored campaign round-trips through config construction;
- invalid ad condition is rejected;
- invalid targeting is rejected;
- advertiser outside population is rejected;
- campaign topic outside configured topics is rejected;
- sponsored campaign with zero budget/frequency cap is rejected;
- `FeedItem` and `SocialMediaPost` preserve campaign fields with defaults.

### Acceptance

- Existing configs keep working.
- Campaign primitives serialize naturally through existing dataclass-to-dict
  paths.
- No behavior changes yet except future-post feed filtering from Task 2.

## Task 2: Build Auditable Ad Delivery Layer

### Intent

Sponsored ads should be delivered by an explicit, inspectable mechanism with
budget, frequency cap, campaign window, and targeting constraints.

### Files

- `src/society_simulation/social_media_ads.py`
- `src/society_simulation/social_media_runner.py`
- `tests/test_social_media_ads.py`

### Required API

Create:

```python
@dataclass(frozen=True)
class AdImpression:
    tick: int
    viewer_id: int
    campaign_id: str
    advertiser_id: int
    post_id: int
    targeting: str
    source_reason: str
    seen_count: int
    frequency_cap: int
    topic: str

@dataclass
class AdDeliveryState:
    remaining_budget_by_campaign: dict[str, int]
    seen_by_user_campaign: dict[tuple[int, str], int]
    impressions: list[AdImpression]
```

Create helper functions:

```python
def initialize_ad_delivery(
    world: InstagramWorld,
    campaigns: Sequence[AdCampaignConfig],
) -> tuple[InstagramWorld, AdDeliveryState]:
    ...

def insert_sponsored_ads(
    *,
    world: InstagramWorld,
    viewer_id: int,
    tick: int,
    feed_items: Sequence[FeedItem],
    feed_size: int,
    campaigns: Sequence[AdCampaignConfig],
    state: AdDeliveryState,
) -> tuple[FeedItem, ...]:
    ...
```

### Semantics

- `no_ad` creates no campaign post and no sponsored impression.
- `organic_post` creates a campaign post but it only appears through normal
  feed ranking.
- `sponsored_ad` creates a campaign post and can insert a sponsored feed item.
- Sponsored insertion is eligible only when `start_tick <= tick <= end_tick`.
- Delivery stops when remaining budget reaches zero.
- A user-campaign pair cannot exceed `frequency_cap`.
- `broad` targeting can deliver to any active viewer.
- `interest_targeted` targeting delivers only when viewer interests overlap
  `targeting_topics`.
- Sponsored card should use `source="sponsored"` and `reason` values:
  `sponsored_broad` or `sponsored_targeted`.
- Sponsored card rank is `0`; existing feed items are shifted and trimmed to
  `feed_size`.

### Future-Post Rule

Normal feed candidate selection must ignore posts with `created_tick > tick`.
This prevents organic campaign posts from appearing before the campaign window.

### Tests First

Add tests that initially fail:

- no-ad does not create ad posts or impressions;
- organic-post creates an ad post but no paid impressions;
- future campaign post is excluded before `start_tick`;
- sponsored campaign does not deliver before `start_tick`;
- sponsored campaign respects total budget;
- sponsored campaign respects per-user frequency cap;
- broad targeting can deliver to a non-interest-matching user;
- interest targeting skips non-matching users;
- sponsored insertion preserves `feed_size` and ranks the ad first.

### Acceptance

- `AdDeliveryState.impressions` explains every paid impression.
- The ad layer is deterministic for a fixed config and activation order.

## Task 3: Integrate Ads Into Runner And Replay

### Intent

The simulator should emit replay artifacts that let us audit campaign exposure
and reproduce every aggregate ad metric.

### Files

- `src/society_simulation/social_media_runner.py`
- `src/society_simulation/social_media_replay.py`
- `tests/test_social_media_runner.py`
- `tests/test_social_media_replay.py`

### Runner Changes

In `run_instagram_social_dynamics`:

1. Build initial world as today.
2. Initialize ad delivery with configured campaigns.
3. On each activation, build the normal feed.
4. Insert sponsored cards when eligible.
5. Pass campaign-aware feed items to the policy.
6. Record action and world update as today.
7. Pass ad impressions to metrics and replay writer.

### Replay Changes

Write a new artifact:

```text
ad_impressions.jsonl
```

Each row should include:

```json
{
  "tick": 9,
  "viewer_id": 17,
  "campaign_id": "maple_3rd_opening",
  "advertiser_id": 0,
  "post_id": 123,
  "targeting": "interest_targeted",
  "source_reason": "sponsored_targeted",
  "seen_count": 1,
  "frequency_cap": 2,
  "topic": "coffee"
}
```

### Tests First

Add tests that initially fail:

- a sponsored run writes `ad_impressions.jsonl`;
- no-ad run writes an empty or absent paid-impression artifact consistently;
- runner summary includes nonzero paid impressions for sponsored condition;
- budget cap bounds paid impressions in the runner;
- replay feed rows include campaign metadata for sponsored cards.

### Acceptance

- Existing social-media runs without campaigns remain compatible.
- New replay artifact is sufficient to trace paid delivery.

## Task 4: Compute Campaign Metrics

### Intent

Separate paid delivery, organic ad exposure, and engagement responses.

### Files

- `src/society_simulation/social_media_metrics.py`
- `tests/test_social_media_metrics.py`

### Metrics

Add campaign metric computation:

```python
paid_impression_count
unique_paid_reach
organic_ad_impression_count
unique_organic_ad_reach
unique_total_ad_reach
relevant_paid_reach
relevant_total_reach
mean_ad_frequency
max_ad_frequency
frequency_cap_hit_count
ad_like_count
advertiser_follow_count
ad_dm_count
ad_generated_post_count
ad_negative_action_count
paid_to_organic_spillover_rate
ad_delivery_exhausted_budget
ad_delivery_remaining_budget
burn_in_action_mean
burn_in_follow_churn
burn_in_exposure_diversity
```

### Definitions

- Paid impressions come from `AdImpression`.
- Organic ad impressions are feed items with `campaign_id` and
  `is_sponsored == False`.
- Relevant reach means viewer interests overlap campaign topic or targeting
  topics.
- `ad_like_count`: `like_post` actions on campaign posts.
- `advertiser_follow_count`: `follow_user` actions targeting campaign
  advertisers.
- `ad_dm_count`: `send_dm` actions targeting campaign advertisers.
- `ad_generated_post_count`: post-creation actions on campaign topics after
  campaign start.
- `ad_negative_action_count`: keep `0` until explicit negative action types are
  added; do not pretend no-op is a negative reaction.
- `paid_to_organic_spillover_rate = organic_ad_impression_count /
  paid_impression_count` when paid impressions are positive, else `0.0`.
- Burn-in ticks are ticks before the earliest non-no-ad campaign start.

### Tests First

Add tests that initially fail:

- paid reach and frequency are computed from ad impressions;
- organic exposure excludes sponsored feed items;
- total reach deduplicates paid and organic viewers;
- relevant reach uses viewer interests;
- ad likes and advertiser follows are counted from actions;
- no campaign returns zero-valued campaign metrics;
- burn-in diagnostics are computed from pre-campaign ticks.

### Acceptance

- Metrics are raw condition metrics; grouped incrementality is handled by the
  analyzer.
- No synthetic metric should silently imply real-world calibration.

## Task 5: Extend Sweep Artifacts And Analyzer

### Intent

Sweeps should preserve ad metrics and produce a readable condition comparison
that answers the first research question.

### Files

- `src/society_simulation/sweep_artifacts.py`
- `src/society_simulation/sweep_analysis.py`
- `tests/test_sweep_artifacts.py`
- `tests/test_sweep_analysis.py`
- `tests/test_sweep_analysis_artifacts.py`

### Required Changes

Add ad metric fields to artifact schemas:

```python
AD_METRIC_FIELDS = (
    "paid_impression_count",
    "unique_paid_reach",
    "organic_ad_impression_count",
    "unique_organic_ad_reach",
    "unique_total_ad_reach",
    "relevant_paid_reach",
    "relevant_total_reach",
    "mean_ad_frequency",
    "max_ad_frequency",
    "frequency_cap_hit_count",
    "ad_like_count",
    "advertiser_follow_count",
    "ad_dm_count",
    "ad_generated_post_count",
    "ad_negative_action_count",
    "paid_to_organic_spillover_rate",
    "ad_delivery_exhausted_budget",
    "ad_delivery_remaining_budget",
    "burn_in_action_mean",
    "burn_in_follow_churn",
    "burn_in_exposure_diversity",
)
```

Analyzer report should include:

- condition summary by `ad_condition`;
- targeting summary by `targeting`;
- creative summary by `creative_id`;
- feed-policy summary by `feed_policy`;
- simple incrementality against no-ad within comparable groups when available;
- explicit warning that results are synthetic and uncalibrated.

### Tests First

Add tests that initially fail:

- sweep run rows preserve campaign metric fields;
- group summary includes ad reach and engagement means;
- markdown report contains a campaign/ad section;
- incrementality table compares sponsored/organic to no-ad when matching
  grouping keys exist.

### Acceptance

- Analyzer output can support the first mock report without hand-computing CSVs.

## Task 6: Add First Mock Experiment And Report

### Intent

Run the first non-trivial but cheap experiment: a local coffee shop campaign in
an Instagram-like synthetic network.

### Files

- `experiments/instagram_local_coffee_ad_reach_sweep.json`
- `docs/research/2026-06-23-instagram-local-coffee-ad-reach-mock.md`
- `docs/research/2026-06-23-instagram-local-coffee-ad-reach-mock.ko.md`
- tests as needed for experiment config parsing

### Experiment Shape

Use mock policy first:

- 40 users;
- 16 total ticks;
- campaign start tick 9;
- campaign end tick 16;
- conditions: no-ad, organic-post, sponsored broad, sponsored interest-targeted;
- creatives: discount and social proof;
- feed policies: chronological and engagement-ranked;
- two seeds initially.

### Report Requirements

Write a concise paper-style report in English and Korean:

- Abstract;
- Setup;
- Claim boundary;
- Experimental design;
- Mechanical validity checks;
- Results table;
- Interpretation;
- Limitations;
- Next experiment.

The report must avoid pretending this is a publishable empirical result. It
should say whether the simulator now supports meaningful follow-up experiments.

### Acceptance

- Mock experiment completes with zero API cost.
- Report cites replay artifacts and summary outputs.
- The result explains what changed compared with prior toy experiments:
  explicit platform state, sponsored-vs-organic controls, campaign logs, and
  reach/spillover metrics.

## Final Verification

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q
PYTHONPATH=src ./.venv/bin/python -m society_simulation sweep \
  experiments/instagram_local_coffee_ad_reach_sweep.json
PYTHONPATH=src ./.venv/bin/python -m society_simulation analyze \
  runs/sweeps/instagram_local_coffee_ad_reach_sweep
```

Then inspect:

- `runs/sweeps/instagram_local_coffee_ad_reach_sweep/analysis/report.md`
- representative `ad_impressions.jsonl`
- representative replay `feed_impressions.jsonl`
- aggregate CSV rows for campaign metrics.

## Stop Conditions

Stop and report before merging if:

- sponsored delivery cannot be audited from replay artifacts;
- no-ad produces paid impressions;
- organic-post control produces paid impressions;
- tests pass only by weakening old social-media behavior;
- analyzer cannot distinguish paid reach from organic spillover.
