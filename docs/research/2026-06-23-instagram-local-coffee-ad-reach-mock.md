# Local Coffee Advertisement Reach in a Synthetic Instagram-Like Society

Date: 2026-06-23

## Abstract

We implemented and tested an auditable advertising intervention layer for the
Instagram-like social simulator. In a 48-run mock sweep, a sponsored local coffee
shop campaign produced substantially higher total ad reach and engagement than
both no-ad and organic-post controls. The result is not evidence of real
Instagram predictive validity. It is evidence that the simulator can now express
paid delivery, organic spillover, frequency caps, creative variants, and
treatment/control comparisons in replayable platform state.

## Claim Boundary

This is a synthetic, uncalibrated mock-policy experiment. It does not predict
Meta reach, CTR, conversion, ROAS, or real human behavior. The valid claim is
narrower: the implementation can run controlled counterfactual ad interventions
and produce auditable metrics for relative mechanism screening.

## Setup

Scenario: `maple_3rd_opening`, a local coffee shop opening-weekend campaign.

Scale:

- 40 synthetic users;
- 16 ticks;
- ad start at tick 9;
- 2 seeds;
- 48 mock-policy runs;
- 0 USD API cost.

Factors:

- `ad_condition`: `no_ad`, `organic_post`, `sponsored_ad`;
- `creative_id`: `discount_offer`, `social_proof_offer`;
- `targeting`: `broad`, `interest_targeted`;
- `feed_policy`: `chronological_following`, `engagement_ranked`;
- `seed`: `20260623`, `20260624`.

Artifacts:

- sweep summary: `runs/sweeps/instagram_local_coffee_ad_reach_sweep/summary.csv`;
- analysis report: `runs/sweeps/instagram_local_coffee_ad_reach_sweep/analysis/report.md`;
- per-run replay includes `ad_impressions.jsonl` and campaign-aware
  `feed_impressions.jsonl`.

## Mechanical Validity Checks

The sweep completed 48/48 runs with no failed runs.

- no-ad produced zero paid impressions;
- organic-post control produced zero paid impressions but nonzero organic ad
  exposure;
- sponsored-ad treatment produced exactly 45 paid impressions per sponsored run,
  bounded by campaign budget;
- replay artifacts record paid impressions separately from organic feed
  exposure;
- analyzer output separates paid reach, total ad reach, relevant reach, and
  ad-specific engagement.

## Results

Condition-level means:

| condition | paid impressions | unique paid reach | organic ad impressions | total ad reach | relevant total reach | ad likes | advertiser follows | ad DMs | generated posts | spillover rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_ad | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| organic_post | 0.00 | 0.00 | 76.38 | 17.25 | 16.38 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sponsored_ad | 45.00 | 33.50 | 83.06 | 36.25 | 35.00 | 13.63 | 6.50 | 3.25 | 2.06 | 1.85 |

Sponsored-only contrasts:

| factor | value | total ad reach | relevant reach | organic ad impressions | ad likes | follows | DMs | generated posts | spillover rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| creative | discount_offer | 35.75 | 34.50 | 66.13 | 6.50 | 7.75 | 3.50 | 1.88 | 1.47 |
| creative | social_proof_offer | 36.75 | 35.50 | 100.00 | 20.75 | 5.25 | 3.00 | 2.25 | 2.22 |
| targeting | broad | 36.50 | 35.00 | 82.50 | 13.50 | 6.38 | 3.25 | 2.00 | 1.83 |
| targeting | interest_targeted | 36.00 | 35.00 | 83.63 | 13.75 | 6.63 | 3.25 | 2.13 | 1.86 |
| feed policy | chronological_following | 34.50 | 33.50 | 43.25 | 17.13 | 6.63 | 3.25 | 3.88 | 0.96 |
| feed policy | engagement_ranked | 38.00 | 36.50 | 122.88 | 10.13 | 6.38 | 3.25 | 0.25 | 2.73 |

## Interpretation

The sponsored condition is mechanically meaningful. It adds paid delivery,
reaches more unique users, and creates more downstream campaign engagement than
the organic-post control. The organic control is also informative: it can create
many feed impressions, but those impressions concentrate into lower unique reach
and weak campaign engagement.

The strongest creative signal is social proof. In sponsored-only runs, the
social-proof creative generated more organic ad impressions, total reach, and
likes than the direct discount creative. This follows from the mock policy: high
visible endorsement and neighbor-sharing language increase conformity-sensitive
public engagement.

The feed-policy contrast is more interesting than a simple "more ranking is
better" story. Engagement-ranked feeds produced much higher organic spillover
and total reach, but chronological feeds generated more direct ad likes and more
generated posts in the sponsored-only subset. This suggests the simulator can
surface tradeoffs between distribution and action type, although the current
mock policy is still stylized.

Targeting did not matter much. Broad and interest-targeted sponsored runs were
nearly identical. This is a diagnostic result, not a claim about advertising:
the current synthetic population and targeting topics are too broad to create a
sharp targeting contrast. The next version needs more distinct user segments,
scarcer budget, and less overlapping interests.

## Limitations

The result is not publishable behavioral evidence. The policy is deterministic
mock behavior, not human or LLM behavior. The population is synthetic and
uncalibrated. The delivery model has no auction, pacing, conversion objective,
quality model, image creative, or real platform feedback loop. The creative
effect is partly built into the mock policy, so it is a mechanism test, not an
empirical discovery.

## Conclusion

This milestone is useful because it moves the project beyond the earlier toy
herding experiments. Ads are now represented as platform state, paid delivery is
auditable, organic spillover is separated from paid impressions, and
treatment/control comparisons can be produced automatically.

The next research step should not be another large mock sweep. It should be a
smaller LLM pilot with this ad-card representation, focused on whether agents
give ordinary human-like reasons: relevance, social proof, repetition, distrust,
privacy, or local interest.
