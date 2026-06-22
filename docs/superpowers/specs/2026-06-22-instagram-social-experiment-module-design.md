# Instagram-Like Social Experiment Module Design

Date: 2026-06-22

## Status

Approved direction from user: treat Instagram/SNS as one experiment family among
many social-simulation experiments, not as a pivot. The design should follow
prior work closely enough to avoid toy "neighbor vote" dynamics and should
imagine the experiments we will run after implementation.

## Research Goal

Build a platform-mediated social experiment module for studying how feed
exposure, visible engagement, following and unfollowing, posting, and private
DMs shape crowd behavior.

The module should answer questions such as:

- Does a feed algorithm amplify conformity, polarization, avoidance, or
  cross-cutting exposure?
- Do visible low-cost endorsements change later likes, follows, DMs, and posts?
- When a topic is socially risky, do agents move discussion into DMs instead of
  public posts?
- Does controversial exposure rewire the follow graph?
- Does source status change cascade reach when message text is held constant?

This is not yet a claim about predicting real Instagram users. The first goal is
to create a controlled and auditable experimental apparatus.

## Prior-Art Requirements

The implementation should preserve these lessons from the reviewed work:

- OASIS: represent platform state explicitly, including users, posts, relations,
  recommendations, traces, and dynamic follow graph updates.
- Y Social and real-data-calibrated online conversation work: treat recommender
  systems as experimental variables, not as prompt text; compare feed policies.
- Social Simulacra: seed a populated social space with personas and historical
  content before the simulation starts.
- Generative Agents: keep memory and retrieved prior experiences available for
  agent continuity, but do not make memory the whole simulator.
- AgentSociety: separate static profile, dynamic status, relationships, behavior,
  and replay.
- Self-report grounded agents: keep synthetic profiles framed as engineering
  baselines; add real grounding later before making prediction claims.
- Chuang et al.: include no-interaction and no-feed controls to separate LLM
  prior drift from social influence.
- Rumor-spreading work: measure graph topology, seed-node choice, spread reach,
  and intervention effects.

## Existing Codebase Fit

The repo currently has three experiment families:

- `sequential_information_cascade`: clean information-cascade baseline;
- `network_herding`: graph-first neighbor action dynamics;
- `event_driven_opinion_dynamics`: persona-based public/private opinion and
  message dynamics with optional LLM policy.

The Instagram-like module should be a fourth experiment family, tentatively
named `instagram_social_dynamics`. It should not be folded into
`event_driven_opinion_dynamics` because the core state is different:

- event-driven runner centers days, events, channels, relationships, messages,
  private stance, and public stance;
- social-media runner centers feed generation, posts, engagement, follow graph,
  platform actions, DMs, and recommendation traces.

The module should still reuse existing conventions:

- dataclass models;
- JSON config loading and validation;
- deterministic seeded mock policy;
- OpenAI-compatible LLM policy contract later;
- JSONL replay artifacts;
- metrics JSON;
- CLI dispatch through `society-sim run`.

## Scope

### In Scope for v0

- deterministic social-media seed generator;
- user profiles with interests, stance, activity traits, privacy preference, and
  posting style;
- historical posts per user;
- directed follow graph;
- feed generation from follow graph plus optional explore candidates;
- deterministic recommender scoring;
- action schema with validation;
- mock policy that emits realistic enough actions for zero-cost tests;
- run loop with activation schedule;
- action application and platform state updates;
- DM threads as private directed messages;
- metrics for engagement, exposure diversity, graph churn, and stance movement;
- replay artifacts for feeds, actions, posts, DMs, graph snapshots, and metrics;
- one example mock config.

### Deferred

- comments, reposts, stories, reels, bookmarks, ads, moderation, image content,
  and live streaming;
- real Instagram API integration;
- UI;
- large-scale paid LLM sweeps;
- empirical calibration from real social-media datasets;
- LLM-generated seed worlds.

## User Model

Each simulated account has a static profile and a dynamic state.

### Static Profile

Fields:

- `user_id`: stable integer ID;
- `handle`: short platform handle;
- `display_name`: readable display name;
- `bio`: short natural-language profile text;
- `interests`: tuple of topic tags, for example `transit`, `housing`, `sports`;
- `home_cluster`: social cluster label;
- `stance`: initial numeric stance on the experiment issue, range `[-1.0, 1.0]`;
- `activity_rate`: probability of activation per tick;
- `post_rate`: baseline probability of creating a post when activated;
- `privacy_preference`: tendency toward DM or silence over public posting;
- `conformity`: sensitivity to visible engagement and perceived norms;
- `skepticism`: resistance to content inconsistent with prior stance;
- `conflict_tolerance`: willingness to engage with opposing content;
- `status_weight`: how much source popularity affects trust and attention;
- `posting_style`: short descriptor used by LLM policy and mock text.

### Dynamic State

Fields:

- current stance;
- confidence;
- salience;
- mood label;
- perceived majority;
- social fatigue;
- recent memory IDs;
- follow count and follower count derived from graph;
- last activation tick.

The initial implementation may keep dynamic state as dataclasses and derive
counts from graph state rather than duplicating them.

## Platform State

The runner owns a `SocialMediaWorld` containing:

- `users`: profile and dynamic state by user ID;
- `posts`: generated and historical posts by post ID;
- `follow_edges`: directed edges `follower_id -> followed_id`;
- `feed_items`: emitted feed impressions by tick and viewer;
- `actions`: validated platform actions;
- `dm_threads`: private directed messages;
- `recommendation_snapshots`: scored feed candidates;
- `interventions`: scheduled event injections or recommendation changes;
- `rng_seed`: deterministic seed metadata.

This is intentionally more structured than a transcript. A platform experiment
needs to know who could see what, why they saw it, and what action they took.

## Initial Context Generation

The v0 seed generator should be deterministic and non-LLM. Given a seed and
config, it creates:

1. Profiles.
2. Initial follow graph.
3. Historical posts.
4. Optional seed event posts.

### Profile Generation

The generator should sample from small built-in pools:

- names and handles;
- topic clusters;
- bios;
- posting styles;
- trait values.

Trait values should be numeric and visible in the config/replay. This avoids
the previous weakness where agents had relationships such as "Carlos and Minho
are close" only implicitly or not at all.

### Follow Graph Generation

Use a directed graph generator with three forces:

- homophily: more likely to follow users with similar interests or stance;
- popularity: some users get more incoming follows;
- noise: random cross-cutting ties.

The config should expose:

- `mean_following`;
- `homophily_weight`;
- `popularity_weight`;
- `random_tie_probability`;
- `mutual_follow_probability`.

This makes "Carlos and Minho are close" an explicit graph fact when it exists.
DMs should be more likely between mutual follows or high-strength ties.

### Historical Posts

Each user gets `historical_posts_per_user` posts with:

- `author_id`;
- `topic`;
- `stance`;
- `text`;
- `created_tick`;
- initial `like_count`;
- optional `seed_engagement_boost`.

Historical posts create context for the feed and memory before any experiment
event occurs.

## Feed and Recommendation Layer

The feed generator creates candidate posts for each activated user. The v0 feed
should combine:

- following candidates: posts from followed users;
- explore candidates: posts from non-followed users selected by interest or
  popularity;
- intervention candidates: posts scheduled by experiment events.

The recommender assigns deterministic scores:

```text
score =
  following_bonus
  + interest_similarity_weight * interest_similarity(viewer, post)
  + stance_similarity_weight * stance_similarity(viewer, post)
  + engagement_weight * log(1 + visible_like_count)
  + recency_weight * recency_decay
  + creator_popularity_weight * log(1 + follower_count(author))
  + controversy_weight * stance_distance(viewer, post)
  + noise_weight * seeded_noise
```

Supported feed policies:

- `chronological_following`: followed accounts only, mostly recency sorted;
- `engagement_ranked`: visible engagement has stronger score weight;
- `interest_homophily`: interest and stance similarity dominate;
- `bridging`: candidates with interest overlap but stance difference get a boost;
- `no_feed_control`: activated users receive no social feed items.

The LLM policy should not see this formula. It should only see feed cards:
author, short bio, visible engagement, relationship context, post text, and
recent personal context.

## Action Space

The v0 action space is intentionally small:

- `like_post`
- `follow_user`
- `unfollow_user`
- `send_dm`
- `create_post`
- `do_nothing`

The system should validate every action. Invalid actions should be logged as
policy errors and converted to `do_nothing` for mock policy tests; for paid LLM
runs, parse failures should count against metrics and may stop the run if a
configured failure threshold is exceeded.

### Action Semantics

`like_post`:

- requires visible post ID from current feed;
- increments post like count;
- records impression-to-like conversion.

`follow_user`:

- requires target user ID from current feed, suggested users, or DM context;
- adds directed follow edge if absent;
- changes future feed exposure.

`unfollow_user`:

- requires currently followed user ID;
- removes directed follow edge if present;
- records graph churn.

`send_dm`:

- requires target user ID;
- creates private message in DM thread;
- target must be mutual follow, recently interacted, or explicitly allowed by
  config.

`create_post`:

- creates public post with topic, stance, and text;
- can be generated by mock template or LLM policy;
- enters future feed candidates.

`do_nothing`:

- records activation without platform action.

## Policy Interface

The policy receives a structured observation:

- current user profile and dynamic state;
- recent memories;
- feed cards for the current activation;
- visible social cues such as like counts and author follower count;
- allowed actions;
- recent DMs;
- optional event context.

The policy returns one structured action:

```json
{
  "action_type": "like_post",
  "post_id": "post-12",
  "target_user_id": null,
  "text": null,
  "topic": null,
  "stance": null,
  "reason": "short private audit reason"
}
```

The LLM prompt must not say "you are simulating a social network experiment."
It should say that the agent is using an Instagram-like app and is deciding what
to do next based on what they see.

## Memory and DMs

Memory should be lightweight in v0:

- store impressions that led to actions;
- store created posts and sent/received DMs;
- retrieve the last `N` relevant memories by topic and recency.

DMs are not internal monologue. They are private messages to other accounts. The
relationship rule should make DMs plausible:

- mutual follows are eligible;
- followed accounts are eligible if config allows one-way DMs;
- recently interacted users can be eligible;
- strangers are ineligible by default.

This addresses the current weakness where "private message" could feel like
undefined inner thought. In this module, DM is a platform action with sender,
recipient, text, and thread ID.

## Metrics

The first implementation should compute:

- total actions and action rates;
- feed impressions;
- impression-to-like conversion;
- public post rate;
- DM rate;
- follow and unfollow counts;
- graph density;
- average following and follower counts;
- follow homophily by stance and interest;
- exposure diversity by stance and interest;
- echo-chamber index;
- stance mean, variance, and polarization;
- public/private divergence when DMs and public posts disagree;
- cascade reach for seed posts;
- cascade depth proxy from follow graph;
- cascade breadth by tick;
- invalid policy output count;
- LLM usage and estimated cost when applicable.

Metrics must be JSON-serializable and should support sweep summaries later.

## Replay Artifacts

Each run should write:

- `config.json`;
- `users.jsonl`;
- `posts.jsonl`;
- `follow_edges_initial.jsonl`;
- `follow_edges_final.jsonl`;
- `feed_impressions.jsonl`;
- `recommendation_snapshots.jsonl`;
- `actions.jsonl`;
- `dm_messages.jsonl`;
- `user_states.jsonl`;
- `metrics.json`;
- `summary.txt`.

For paid LLM runs, add:

- `llm_decisions.jsonl`;
- token usage and estimated cost in `metrics.json`.

## Experiment Designs Enabled by v0

### Experiment A: Feed Algorithm Comparison

Same seed, users, graph, and historical posts. Vary only feed policy:

- chronological following;
- engagement-ranked;
- interest-homophily;
- bridging;
- no-feed control.

Question: Does recommender structure change exposure diversity, graph churn,
posting, liking, DMs, and stance polarization?

Expected insight: If differences appear under identical agents and content, the
platform layer is doing causal work.

### Experiment B: Visible Endorsement Cascade

Same seed post, same text, same author. Vary visible initial likes:

- zero likes;
- moderate likes;
- high likes.

Agents are not told this is a herding experiment. They see a normal feed card.

Question: Does low-cost visible endorsement change downstream likes, follows,
DMs, and posts?

Expected insight: Crowd behavior emerges from an ordinary affordance, not from
an artificial neighbor-vote table.

### Experiment C: Public Silence and Private Backchannel

Introduce a socially risky issue. Allow posts, likes, DMs, and no-op. Compare
public expression with private DM discussion.

Question: Do agents with low conflict tolerance or high privacy preference avoid
public posting but still discuss privately?

Expected insight: Social pressure may reduce visible public opinion while
increasing private coordination.

### Experiment D: Graph Rewiring After Shock

Expose users to posts from weak ties, strong ties, and high-status accounts.
Measure unfollows, new follows, homophily shift, and exposure diversity.

Question: Does shock exposure rewire the platform graph toward echo chambers or
cross-cutting ties?

Expected insight: Opinion dynamics and graph dynamics should be coupled.

### Experiment E: Influencer Versus Ordinary Source

Same message, different author status. Compare a high-follower seed account to
an ordinary account.

Question: Does source status affect reach and belief movement independent of
message text?

Expected insight: Platform status is a social signal and should be modeled
separately from content.

## Testing Strategy

Tests should cover:

- config parsing and validation;
- deterministic profile generation;
- deterministic graph generation;
- deterministic historical posts;
- feed candidate generation;
- each feed policy's ordering behavior;
- action validation;
- action application;
- DM eligibility;
- metrics on small hand-built worlds;
- replay artifact writing;
- CLI run with example config.

The first implementation should be mock-policy complete before any paid LLM
pilot. Paid runs should be a later step with a strict cost cap and early stop on
parse errors or token growth.

## Success Criteria

The module is ready for its first paid LLM pilot when:

- a mock run completes from CLI with deterministic artifacts;
- tests pass for config, feed, actions, runner, metrics, and replay;
- feed-policy comparison can run without LLM calls;
- metrics show exposure, graph, engagement, DM, and stance outputs;
- action traces can explain why each aggregate metric changed;
- LLM policy has a JSON contract and cost cap but is not required for mock
  experiments.
