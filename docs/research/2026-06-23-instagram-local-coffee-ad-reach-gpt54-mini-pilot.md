# GPT-5.4 Mini Local Coffee Advertisement Reach Pilot

Date: 2026-06-23

## Abstract

We ran a small paid LLM pilot of the Instagram-like advertisement simulator. The
goal was not to estimate real Instagram ad performance. The goal was to test
whether LLM agents, shown ordinary feed-card information rather than an explicit
"social experiment" frame, produce plausible local-ad reactions and auditable
reasons.

The run completed 44 GPT-5.4 mini social-media decisions across 8 synthetic
users and 8 ticks. The configured cost accounting used `$0.75` per 1M input
tokens and `$4.50` per 1M output tokens, with a `$0.20` run cap. Observed usage
was 17,932 prompt tokens and 2,822 completion tokens, for an estimated cost of
`$0.026148`.

The pilot produced a useful but clearly preliminary result. The simulator
expressed paid delivery, frequency capping, ad-specific likes, and organic
spillover. LLM reasons referred to relevance, local interest, repetition, and
sponsored-content skepticism. However, the agent policy was too like-heavy:
41/44 decisions were `like_post`, with no follows, DMs, or generated posts. This
is not publishable behavioral evidence yet. It is a diagnostic pilot showing
what must be fixed before a larger ad-effect experiment.

## Research Question

Can an Instagram-like LLM society represent a local advertisement intervention
as ordinary platform experience, and can it produce auditable qualitative
signals about ad reach and reaction?

The intended target phenomenon is not simple herding. The target is a
platform-mediated path:

> paid exposure -> visible engagement -> organic feed spillover -> later
> engagement or avoidance.

This matters because an ad effect in a social platform is not only a direct
click-like event. It can also change what later users see as socially endorsed
content.

## Experimental Setup

Config:

- `experiments/instagram_local_coffee_ad_reach_gpt54_mini_pilot.json`

Run directory:

- `runs/instagram_local_coffee_ad_reach_gpt54_mini_pilot_20260623`

Scenario:

- 8 synthetic users;
- 8 simulated ticks;
- engagement-ranked feed;
- 4 visible feed cards per active user;
- activation probability `1.0`;
- interests over coffee, local events, food, commute, and fitness;
- one sponsored local coffee campaign from tick 5 through tick 8.

Campaign:

- campaign: `maple_3rd_opening`;
- creative: `social_proof_offer`;
- text: "Neighbors are already sharing Maple & 3rd Coffee's opening weekend
  menu. Drop by for espresso, pastries, and a first look.";
- targeting: `interest_targeted`;
- target topics: coffee, food, commute;
- paid impression budget: 16;
- frequency cap: 2;
- initial sponsored like count: 70.

The LLM prompt did not tell agents that they were participating in a social
network experiment. It described an account, interests, posting style, current
state, recent memories, and a visible feed. Sponsored cards were visible as feed
metadata: `label=sponsored`, campaign id, source reason, visible like count, and
seen-before count.

## Engineering Changes During the Pilot

The first paid attempt failed before replay writing because an LLM response used
a non-integer optional field where the parser expected `target_user_id` to be an
integer. Investigation found a real interface bug: the prompt required integer
`target_user_id`, but feed lines only showed `@handle`, not `author_id`.

We fixed this before the successful run:

- feed prompt lines now include `author_id`;
- the action schema tells the model to use integer `author_id` for follow, DM,
  or unfollow actions;
- optional fields are normalized when the LLM returns common null strings such
  as `"none"`, `"null"`, or `"N/A"`;
- optional numeric fields accept numeric strings such as `"2"` and `"0.25"`.

This was not just a convenience patch. It made the action contract executable:
the model can only emit valid integer targets if the prompt actually exposes
integer targets.

## Results

### Cost and Reliability

| metric | value |
|---|---:|
| LLM calls | 44 |
| prompt tokens | 17,932 |
| completion tokens | 2,822 |
| input cost | `$0.013449` |
| output cost | `$0.012699` |
| total estimated cost | `$0.026148` |
| configured cap | `$0.20` |
| successful replay writes | 1 |

### Aggregate Platform Metrics

| metric | value |
|---|---:|
| users | 8 |
| feed impressions | 176 |
| non-noop actions | 41 |
| action counts | `{'like_post': 41, 'do_nothing': 3}` |
| paid impressions | 12 |
| unique paid reach | 7 |
| relevant paid reach | 7 |
| organic ad impressions | 4 |
| unique organic ad reach | 3 |
| unique total ad reach | 7 |
| ad likes | 5 |
| advertiser follows | 0 |
| ad DMs | 0 |
| generated ad-topic posts | 0 |
| mean ad frequency | 1.714286 |
| max ad frequency | 2 |
| frequency-cap hit count | 5 |
| remaining paid budget | 4 |

The campaign did not exhaust its budget. It delivered 12 of 16 possible paid
impressions because interest targeting and the finite active population limited
eligible delivery.

### Ad-Context Decisions

There were 16 LLM decisions in which the ad post appeared in the visible feed as
either sponsored content or organic campaign content. Among these decisions:

| action in ad-visible context | count |
|---|---:|
| `like_post` | 13 |
| `do_nothing` | 3 |

Only 5 of the 16 ad-visible decisions directly liked the ad post. The remaining
like actions often chose another organic post in the same feed, especially when
that post matched the user's account interests or style.

Examples of qualitative reasons:

- A skeptical coffee/local-events user skipped the ad because the feed felt
  "mostly sponsored" and the coffee opening promo felt repetitive.
- A food/local-events user liked the ad because the local coffee opening matched
  their neighborhood and food interests.
- A coffee/commute user liked the ad after it appeared organically, citing a
  warm local vibe and relevance.

The important sign is that LLM agents did not simply act on the top-ranked card
or blindly copy the sponsored signal. They sometimes ignored the sponsored ad in
favor of an ordinary post, and the `do_nothing` reasons explicitly mentioned
skepticism and repetition.

## Interpretation

The pilot supports a narrow claim: the simulator can expose a sponsored local
ad as ordinary platform state and record how LLM agents react to it. The paid
ad generated direct ad likes, and those likes increased the ad post's visible
engagement. Later, the same campaign post appeared as organic campaign content
through the feed ranking layer. This is the intended mechanism boundary:

> paid delivery can create social visibility that later affects organic
> distribution.

The more interesting behavioral observation is negative. The LLM policy is
currently too eager to like. A realistic Instagram user should sometimes scroll,
linger, ignore, follow, DM, save, or post later. In this run, no agent followed
the advertiser, sent a DM, or created a campaign-related post. That means the
action space exists, but the prompt and scenario do not yet create enough
friction or motive diversity.

## Limitations

This is a single small pilot with no control condition, no seed sweep, no human
calibration, and no external benchmark. It should not be interpreted as evidence
about real ad reach or real human response.

The current social-media LLM prompt still has weak memory. Although the config
enables `memory_retrieval`, the successful run's prompts contained no retrieved
recent memories. This makes behavior more feed-reactive than socially
continuous.

The environment has no image creative, no auction, no pacing strategy, no
conversion objective, no explicit saves, no comments, no dwell time, and no
realistic user inactivity prior. The high `like_post` rate is therefore a model
and prompt artifact, not a behavioral finding.

## Conclusion

This was the right next experiment after the mock ad sweep. It cost little,
exercised the real LLM path, and revealed both a mechanism and a failure mode.

The useful mechanism is paid-to-organic spillover: a sponsored ad can receive
engagement, then reappear as organic campaign content through the feed ranking
layer. The useful failure mode is excessive like bias: the model treats ordinary
Instagram participation as too frictionless.

The next experiment should not scale user count first. It should improve
behavioral realism first:

1. add actual recent-memory retrieval to the social-media LLM prompt;
2. introduce action friction, including explicit scroll/no-action priors;
3. add `save_post` or `comment` only if the simulator can measure them cleanly;
4. run a paired no-ad vs sponsored-ad LLM pilot with the same seed;
5. compare not only reach, but reaction type, reason categories, and spillover
   timing.

Only after those changes should we run a larger sweep.

## Artifacts

- Config: `experiments/instagram_local_coffee_ad_reach_gpt54_mini_pilot.json`
- Run directory: `runs/instagram_local_coffee_ad_reach_gpt54_mini_pilot_20260623`
- Metrics: `runs/instagram_local_coffee_ad_reach_gpt54_mini_pilot_20260623/metrics.json`
- Decisions: `runs/instagram_local_coffee_ad_reach_gpt54_mini_pilot_20260623/llm_decisions.jsonl`
- Ad impressions: `runs/instagram_local_coffee_ad_reach_gpt54_mini_pilot_20260623/ad_impressions.jsonl`
- Actions: `runs/instagram_local_coffee_ad_reach_gpt54_mini_pilot_20260623/actions.jsonl`
