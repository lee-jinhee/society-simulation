# Visible Endorsement Cascade Mock Pilot

## Summary

This mock pilot tests whether an Instagram-like simulator can represent a
minimal endorsement cascade: the same post, from the same author, with the same
text, produces different downstream engagement when only the visible like count
changes.

The result is intentionally modest but useful. With the deterministic mock
policy, visible endorsement now creates a clear monotonic response in likes:

| visible likes condition | mean action count | mean like count |
| --- | ---: | ---: |
| low, 0 initial likes | 0.0000 | 0.0000 |
| moderate, 40 initial likes | 9.0000 | 9.0000 |
| high, 80 initial likes | 32.0000 | 32.0000 |

This is not evidence about real human behavior. It is a system-validity result:
the simulator can now expose a realistic social signal to agents, let the agent
policy react to that signal, and aggregate the response through the sweep
analyzer.

## Experiment

The experiment is defined in:

- `experiments/instagram_visible_endorsement_sweep.json`

It runs six mock simulations:

- 2 seeds: `20260622`, `20260623`
- 3 visible endorsement levels: `low`, `moderate`, `high`

Each run injects one configured seed post into the initial Instagram-like world:

> The new bus lane finally makes downtown feel easier to reach.

Only the seed post's initial `like_count` changes:

- low: `0`
- moderate: `40`
- high: `80`

The update policy is deterministic `mock_social` with
`response_style=endorsement_sensitive`. The policy does not receive an
experiment label. It sees ordinary feed-card metadata, including visible likes,
topic, text, and author handle.

## Implementation Change

This pilot required four simulator changes:

1. `InstagramSocialDynamicsConfig` now supports optional `seed_posts`.
2. The seed generator injects configured seed posts into the initial world.
3. `FeedItem` now carries visible card metadata: like count, topic, text, and
   author handle.
4. `MockSocialMediaPolicy` uses visible endorsement as a social signal. In
   `endorsement_sensitive` mode, algorithmic feed score alone no longer causes
   a like; visible endorsement must cross a persona-dependent threshold.

We also changed `action_count` for social-media metrics to exclude
`do_nothing`, so the report reflects actual platform actions rather than all
activation decisions.

## Result

Analyzer output:

- `runs/sweeps/instagram_visible_endorsement_sweep/analysis/report.md`

Key aggregate pattern:

- low visible likes produced no downstream likes;
- moderate visible likes produced a small cascade;
- high visible likes produced a large cascade;
- exposure diversity changed slightly but was not the primary manipulated
  outcome;
- no graph rewiring occurred because this mock policy still does not use
  follow/unfollow in response to endorsement.

## Interpretation

This is a better experiment than the earlier toy herding setup because the
agent is not given a social-science label or a neighbor vote table. It sees a
normal platform affordance: a post card with visible likes. The observed
aggregate behavior comes from the interaction between platform state and the
agent policy.

The current result is still not publishable as a human-behavior finding. The
policy is deterministic and hand-authored. Its value is that it validates the
experimental plumbing needed for a paid LLM pilot:

- controlled social stimulus;
- ordinary social-media presentation;
- auditable action traces;
- sweep-level aggregate comparison;
- no-cost debugging before LLM calls.

## Next Step

The next paid or semi-paid pilot should run the same visible-endorsement sweep
with a small LLM policy and a strict cost cap. The research question should be:

> Does an LLM persona show a graded response to visible endorsement when the
> post content, author, topic, and feed policy are held fixed?

The pilot should also inspect raw action reasons. If LLM agents explicitly cite
visible likes, source credibility, social proof, or skepticism, this becomes a
more credible foundation for studying platform-mediated crowd behavior.
