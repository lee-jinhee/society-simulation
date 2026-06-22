# Instagram Ad Reach Experiment Design

Date: 2026-06-23

## Status

Approved direction from user: the first interesting Instagram-like experiment
should study what happens when an advertisement enters a stable synthetic social
environment. The goal is not to claim exact prediction of real Instagram ad
performance. The goal is to build a counterfactual lab for studying ad reach,
organic spillover, frequency fatigue, and relative condition ranking before any
real-data calibration.

## Claim Boundary

This module must not claim that it predicts real Instagram campaign reach,
conversion, or ROAS. Real platform outcomes depend on private auction,
delivery, pacing, user-history, advertiser-quality, and optimization systems
that this simulator does not observe.

The credible claim is narrower:

> Given a stable synthetic Instagram-like world, controlled ad interventions,
> and explicit treatment/control conditions, the simulator can compare
> mechanisms and generate hypotheses about relative ad reach and engagement.

The first validation target is rank-order and mechanism consistency, not exact
numeric prediction.

Examples of valid claims:

- interest-targeted delivery creates higher relevant reach than broad delivery
  in this synthetic population;
- high frequency increases fatigue in privacy-sensitive users;
- engagement-ranked feeds create more organic spillover than chronological
  feeds;
- social-proof creative outperforms discount creative among high-conformity
  clusters.

Examples of invalid claims:

- the simulator predicts actual Meta reach for a budget;
- LLM personas represent real customers without calibration;
- mock policy outputs are human behavior evidence;
- simulator lift can replace randomized lift testing.

## Research Question

In a stable Instagram-like society, how much incremental reach and engagement
does a sponsored post create beyond no-ad and organic-post controls?

Subquestions:

- Does targeting improve relevant reach, or merely reduce total reach?
- Does social-proof creative increase organic spillover compared with a direct
  discount offer?
- Does engagement-ranked feed amplify paid exposure into organic engagement?
- Does repeated exposure create fatigue or negative reactions?
- Can simulator outputs identify which mechanisms deserve a real A/B test?

## First Scenario

Use a local business advertisement instead of a political or policy ad. This
keeps the first ad experiment commercially meaningful while avoiding needless
ideological confounds.

Advertiser:

- `maple_3rd_coffee`
- category: `local_business`
- target-adjacent topics: `coffee`, `local_events`, `food`, `commute`

Creative A, discount offer:

> Maple & 3rd Coffee opens this weekend. First 100 visitors get a free pastry
> with any drink.

Creative B, social proof offer:

> Neighbors are already sharing Maple & 3rd Coffee's opening weekend menu.
> Drop by for espresso, pastries, and a first look.

The ad should be represented as platform state, not as hidden prompt text. A
feed item should expose that it is sponsored, what campaign it belongs to, the
advertiser identity, visible likes, and text.

## Stable World Requirement

The simulator must separate pre-ad stabilization from post-ad response.

Run phases:

1. Seed synthetic users, follows, and historical posts.
2. Burn in the environment without ads for `burn_in_ticks`.
3. Check stability criteria.
4. Inject ad treatment or control content.
5. Observe post-ad dynamics for `measurement_ticks`.

Initial stability criteria:

- the final three burn-in ticks have non-zero but bounded platform actions;
- follow/unfollow churn is low during burn-in;
- exposure diversity does not collapse to zero;
- no single seed post dominates every feed before treatment;
- baseline engagement metrics are written to the replay.

Milestone 1 may compute stability diagnostics and report them without blocking
the run. Milestone 2 can fail or reroll worlds that do not meet the stability
criteria.

## Experimental Conditions

The first sweep should use a compact factorial design:

- `ad_condition`: `no_ad`, `organic_post`, `sponsored_ad`
- `targeting`: `broad`, `interest_targeted`
- `creative`: `discount_offer`, `social_proof_offer`
- `feed_policy`: `chronological_following`, `engagement_ranked`
- `seed`: at least two seeds for mock, at least three before treating a result
  as robust

For the initial mock experiment, some combinations are structurally redundant:
`targeting` and `creative` are ignored by `no_ad`. The sweep can still include
the full grid for simplicity, but reports should group by `ad_condition` first
and avoid over-interpreting empty contrasts.

Recommended mock scale:

- 40 users;
- 8 burn-in ticks;
- ad injection at tick 9;
- 8 measurement ticks;
- 2 seeds for fast iteration;
- 3 or more seeds before writing a stronger research note.

## Treatment Semantics

### No-Ad Control

No sponsored or equivalent organic advertiser post is injected. This estimates
baseline reach and engagement in the synthetic world.

### Organic-Post Control

The advertiser post appears as a normal post from the advertiser account. It can
spread only through existing graph/feed mechanics. This separates paid delivery
from content quality.

### Sponsored-Ad Treatment

The advertiser post is eligible for paid insertion into feeds. It is marked as
sponsored, constrained by budget, targeting, and frequency cap, and can also
generate organic spillover if users engage with it.

## Ad Delivery Model

The first ad delivery model should be deliberately simple and auditable.

Campaign config:

- `campaign_id`
- `advertiser_id`
- `ad_condition`
- `creative_id`
- `creative_text`
- `topic`
- `stance`
- `start_tick`
- `end_tick`
- `budget_impressions`
- `frequency_cap`
- `targeting`
- `sponsored_like_count`

Targeting modes:

- `broad`: any active user can receive the ad until budget and frequency cap are
  exhausted;
- `interest_targeted`: active users whose interests overlap the campaign topic
  or adjacent topics are prioritized.

The delivery system should write an auditable ad impression record every time a
sponsored feed item is inserted.

## Agent Behavior

Mock and LLM policies should receive the same visible ad-card facts:

- sponsored label;
- advertiser handle;
- campaign topic;
- text;
- visible likes;
- whether the user has seen this ad before;
- whether followed users have engaged with it;
- source reason such as `sponsored_targeted`, `sponsored_broad`, or
  `organic_spillover`.

The mock policy should treat ads as ordinary platform content with extra
friction:

- higher interest match increases like/follow probability;
- discount offer increases direct engagement;
- social-proof offer increases conformity-sensitive engagement;
- repeated exposure increases fatigue;
- privacy-sensitive users may DM a friend instead of publicly engaging;
- high skepticism reduces sponsored engagement unless social proof is strong.

LLM prompt text must not say that the user is in an advertising experiment. It
should say the user is seeing an Instagram-like feed item marked sponsored.

## Metrics

The ad experiment needs campaign-specific metrics in addition to generic social
metrics.

Reach:

- `paid_impression_count`
- `unique_paid_reach`
- `organic_ad_impression_count`
- `unique_organic_ad_reach`
- `unique_total_ad_reach`
- `relevant_paid_reach`
- `relevant_total_reach`

Frequency:

- `mean_ad_frequency`
- `max_ad_frequency`
- `frequency_cap_hit_count`

Engagement:

- `ad_like_count`
- `advertiser_follow_count`
- `ad_dm_count`
- `ad_generated_post_count`
- `ad_negative_action_count`

Incrementality:

- `incremental_reach_vs_no_ad`
- `incremental_engagement_vs_no_ad`
- `paid_to_organic_spillover_rate`
- `relevant_reach_lift_vs_organic`

Diagnostics:

- `burn_in_action_mean`
- `burn_in_follow_churn`
- `burn_in_exposure_diversity`
- `ad_delivery_exhausted_budget`
- `ad_delivery_remaining_budget`

The first implementation can compute raw condition metrics. Incremental lift can
be computed in the analyzer by comparing grouped means against the `no_ad`
condition within the same seed/feed-policy block.

## Analyzer Output

The analysis report should include:

1. Overview of runs, completed/failed count, and campaign name.
2. Stability diagnostics for burn-in.
3. Condition summary by `ad_condition`.
4. Targeting summary by `targeting`.
5. Creative summary by `creative`.
6. Feed-policy summary by `feed_policy`.
7. Incrementality table comparing treatment and controls.
8. Short warning section stating that outputs are synthetic and uncalibrated.

Primary table:

| condition | paid reach | organic reach | total reach | relevant reach | engagement | spillover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

Secondary table:

| creative | broad targeting | interest targeting | relative lift |
| --- | ---: | ---: | ---: |

## Validation Plan

Validation should proceed in stages.

### Stage 1: Mechanical Validity

Use mock policy.

The run is acceptable if:

- no-ad has zero paid impressions;
- organic-post control has zero paid impressions but non-zero organic exposure;
- sponsored treatment has paid impressions bounded by budget/frequency cap;
- interest targeting has higher relevant reach share than broad targeting;
- replay artifacts can explain each aggregate metric.

### Stage 2: Behavioral Plausibility

Use a small LLM pilot with strict cost cap.

The run is useful if:

- LLM action reasons refer to ordinary ad-card facts such as relevance, offer,
  visible social proof, repetition, or distrust;
- condition ranking is qualitatively stable across seeds;
- no prompt contains "advertising experiment" or equivalent hidden setup;
- invalid JSON/action rate remains low.

### Stage 3: Calibration

Do not claim predictive validity until calibrated against real or semi-real ad
benchmarks, such as known CTR ranges, lift-test directionality, or anonymized
campaign logs. Calibration should start with relative comparisons, not exact
reach.

## Cost Plan

Mock experiments cost 0 USD.

The first LLM pilot should be smaller than the mock sweep:

- 12 to 16 users;
- 4 burn-in ticks;
- 4 measurement ticks;
- representative conditions only: no-ad, organic-post, sponsored broad, and
  sponsored interest-targeted;
- one seed first;
- hard cost cap below 3 USD;
- stop if estimated cost or invalid-action rate spikes.

The LLM pilot should be treated as qualitative behavior inspection, not a final
effect estimate.

## Implementation Boundary

The first implementation should not build:

- real ad auction;
- CPM/CPC bidding;
- conversion tracking;
- external Meta API integration;
- image/video creatives;
- purchase prediction;
- UI.

It should build the minimum auditable ad-intervention layer required to run the
mock sweep and produce a campaign report.

## Success Criteria

The feature is ready for the first research report when:

- the simulator can run no-ad, organic-post, and sponsored-ad conditions;
- campaign delivery records are written to replay artifacts;
- campaign metrics separate paid reach from organic spillover;
- the analyzer can compare treatment against controls;
- mock results show interpretable directional differences;
- the report explicitly states claim boundaries and calibration gaps.
