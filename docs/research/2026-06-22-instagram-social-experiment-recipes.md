# Instagram-Like Social Experiment Recipes

Date: 2026-06-22

## Recipe 1: Mock Feed-Policy Sweep

Command:

```bash
./.venv/bin/python -m society_simulation sweep experiments/instagram_feed_policy_sweep.json
```

Purpose: compare chronological, engagement-ranked, interest-homophily, bridging,
and no-feed controls under identical agents and content.

Primary metrics: exposure diversity, action counts, follow edge delta, final
stance variance, DM count, like count.

Interpretation: this is a zero-cost mechanism test. It can show that platform
exposure rules change observable traces, but it is not evidence that synthetic
mock users predict real social-media users.

## Recipe 2: Tiny Paid LLM Pilot

Purpose: confirm the platform prompt/action contract works with a real model
without claiming real-world validity.

Suggested configuration changes from the mock run:

- `num_users`: 6
- `ticks`: 2
- `feed_size`: 3
- `update_policy.type`: `llm`
- `update_policy.max_estimated_cost_usd`: 3.0
- `update_policy.max_completion_tokens`: 96

Stop conditions: parse failure rate above 20%, estimated cost above cap, or
unexpected action explosion.

## Recipe 3: Visible Endorsement Cascade

Use the same seed post, author, text, graph, and feed policy. Vary only initial
visible like count: zero, moderate, high.

Primary metrics: impression-to-like conversion, downstream follow changes, DM
count, public posts about the topic, cascade reach by tick.

Interpretation: this tests ordinary platform affordances rather than artificial
neighbor vote tables.

## Recipe 4: Public Silence and Private Backchannel

Inject a socially risky post and compare public actions with DMs.

Primary metrics: public post count, DM count, public/private stance divergence,
privacy-preference subgroup effects, and whether DM discussion precedes later
public posting.
