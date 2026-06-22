# Prior-Art Report: Instagram-Like Social Media Experiments

Date: 2026-06-22

## Purpose

This report reviews prior work relevant to adding an Instagram-like social-media
experiment family to the society simulator. This is not a pivot away from the
broader social-simulation program. It is one controlled experiment family inside
that program, focused on platform-mediated crowd behavior: feed exposure,
low-cost endorsement, following and unfollowing, posting, private backchannels,
and recommendation-driven opinion movement.

The key design question is not whether agents can "talk like humans" in an
abstract chat room. The key question is whether a platform environment can be
specified tightly enough that we can study how individual propensities,
relationships, content exposure, and recommender choices produce aggregate
social outcomes.

## Sources Reviewed

Primary sources downloaded to `docs/research/reference_papers/` and reviewed
with local PDF text extraction:

- Park et al. 2023, "Generative Agents: Interactive Simulacra of Human
  Behavior": https://arxiv.org/abs/2304.03442
- Park et al. 2024/2026, "LLM Agents Grounded in Self-Reports Enable
  General-Purpose Simulation of Individuals": https://arxiv.org/abs/2411.10109
- Piao et al. 2025/2026, "AgentSociety": https://arxiv.org/abs/2502.08691
- Yang et al. 2024/2026, "OASIS: Open Agent Social Interaction Simulations with
  One Million Agents": https://arxiv.org/abs/2411.11581
- Rossetti et al. 2024, "Y Social: An LLM-powered Social Media Digital Twin":
  https://arxiv.org/abs/2408.00818
- Park et al. 2022, "Social Simulacra: Creating Populated Prototypes for Social
  Computing Systems": https://arxiv.org/abs/2208.04024
- Chuang et al. 2024, "Simulating Opinion Dynamics with Networks of LLM-based
  Agents": https://arxiv.org/abs/2311.09618 and
  https://aclanthology.org/2024.findings-naacl.211/
- Hu et al. 2025, "Simulating Rumor Spreading in Social Networks using LLM
  Agents": https://arxiv.org/abs/2502.01450
- Composta et al. 2025, "Simulating Online Social Media Conversations on
  Controversial Topics Using AI Agents Calibrated on Real-World Data":
  https://arxiv.org/abs/2509.18985

Reference repositories and documentation checked:

- AgentSociety: https://github.com/tsinghua-fib-lab/agentsociety
- OASIS: https://github.com/camel-ai/oasis
- Rumor spreading with LLM agents:
  https://github.com/UT-SysML/rumors-in-multi-agent

## Executive Takeaways

1. Serious social-media simulations do not only prompt agents to chat. They
   define platform state: users, posts, comments, relations, traces,
   recommendations, and dynamic follow graphs.
2. Recommendation and information-filtering logic must be explicit. Prior work
   treats recommender systems as experimental variables, not background noise.
3. Action space should be intentionally reduced per experiment. OASIS supports a
   broad action set, but individual experiments often restrict actions to a
   smaller subset such as like, repost, follow, and do nothing.
4. Initial context must be richer than random neighbors. The simulator needs
   profiles, interests, stance, activity rates, historical posts, and a seeded
   follow graph with controllable homophily and popularity.
5. Raw LLM agents are not automatically human-like. Chuang et al. show drift
   toward accurate or consensus positions even without social interaction. We
   need no-interaction controls and explicit susceptibility, skepticism, and
   conformity variables.
6. Real predictive claims require grounding. The self-report grounded agent
   paper shows that interview and survey grounding substantially improves
   person-specific prediction versus demographic-only prompting.
7. The first implementation should produce auditable platform traces and
   research metrics before any large paid LLM run.

## OASIS: Social Media as Platform State Plus Action Space

OASIS is the closest architectural reference for an Instagram-like experiment
module. It models social media as a dynamically updated environment with
platform state, recommendation systems, user actions, and a time engine. The
important lesson is structural: the simulator contains an environment server,
recommendation system, agent module, time engine, and scalable inferencer.

The environment state includes users, posts, comments, relations, traces, and
recommendations. Follow relationships and post information update over time,
which lets experiments study not just belief movement but graph movement. That
matters for our user-facing action set: following and unfollowing are not just
messages; they are state transitions that change future exposure.

OASIS also makes action space a controlled variable. Its full social-media
action set includes operations such as creating posts, commenting, following and
unfollowing, liking and unliking, disliking, searching, trending, refreshing,
muting, and doing nothing. But the paper does not always expose all actions. In
the Platform X information-spreading experiment, it restricts action space to a
smaller subset including like, repost, follow, and do nothing. This is the
correct pattern for us: start with a small, interpretable action set, then add
actions only when an experiment needs them.

For our simulator, OASIS suggests these implementation principles:

- first-class platform tables, not only conversation logs;
- a deterministic recommender layer that can be swapped or ablated;
- dynamic follow graph updates;
- explicit action schemas;
- action-level traces that can be replayed;
- per-experiment reduced action spaces.

## Y Social and Real-Data-Calibrated Social Media Conversations

Y Social frames social media simulation as a digital twin: LLM agents consume
content, post, comment, react, follow, and participate in configurable platform
dynamics. It emphasizes that tiny actions such as following or resharing can
become macro-level phenomena when repeated across a population.

The newer online-conversation study builds on this line by calibrating agents
on real-world Italian election conversations. It varies LLM choice, network
structure, and recommendation system, then compares simulated opinion dynamics
and discourse patterns against empirical data. This is exactly the long-term
direction if we want social simulation to become more than a toy: use real
conversation traces, seed initial profiles and opinions from those traces, and
judge whether the simulation reproduces aggregate patterns.

The study also gives a caution. It finds that simulated agents can produce
plausible conversations and form connections, but they may underproduce
heterogeneous tone, toxicity, or unfollow behavior. This implies our design
should not assume that "add LLM" equals realism. We need metrics for low-frequency
but socially important actions such as unfollowing, avoidance, silence, and
private backchannels.

For our simulator, Y Social suggests:

- make recommender policies first-class experiment variables;
- compare chronological, popularity-driven, interest-driven, and bridging feeds;
- separate content generation from exposure generation;
- track whether low-cost actions dominate high-cost actions;
- treat real-data calibration as a later validation tier.

## Social Simulacra: Seeding a Platform Context

Social Simulacra is less about forecasting and more about prototyping social
spaces. It takes a community design goal, rules, and member personas, then
generates users, posts, replies, and anti-social behaviors. Its useful lesson is
not "trust synthetic communities as truth." Its useful lesson is how to bootstrap
an environment that feels populated before a simulation starts.

For an Instagram-like experiment, this maps to initial context generation:

- generate or load user profiles with short bios, interests, and posting style;
- generate historical posts per user with topics, stance, tone, and engagement;
- include occasional norm-violating or controversial posts when an experiment
  calls for them;
- treat initial content as seed state, not as an outcome.

The simulator should support a deterministic non-LLM seed generator first. An
LLM-based context generator can be added later as one seed condition, not as the
only source of initial state.

## Generative Agents and AgentSociety: Internal Continuity

Generative Agents contributes the internal agent architecture: memory stream,
retrieval by recency/relevance/importance, reflection, and planning. It is
useful for making agents coherent across time. It is not enough by itself for a
social-media experiment because it does not define feed algorithms, platform
traces, or intervention metrics.

AgentSociety reinforces the need to separate static profile, dynamic status,
mental processes, social relationships, and behaviors. Its experiments also
make interventions explicit, such as message injection, node intervention, and
edge intervention. For our Instagram-like experiment family, this argues for a
clean separation:

- profile: demographics, interests, posting style, privacy preference;
- dynamic state: mood, stance, salience, perceived norms, social fatigue;
- relationships: follow edges, tie type when known, tie strength if available;
- memory: remembered posts, DMs, follows/unfollows, and personal reflections;
- behavior: platform actions emitted under the current exposure.

This separation lets the same agent profile run under multiple feed algorithms
or interventions without rewriting the agent itself.

## Self-Report Grounded Agents: Validity Boundary

The self-report grounded agent paper is important because it sets the validity
boundary. Agents grounded in interviews, surveys, or both predict individual
responses more accurately than demographic-only prompts, and the evaluation is
benchmarked against participant test-retest consistency.

For the Instagram-like module, the implication is direct: synthetic profiles are
acceptable for engineering and mechanism discovery, but they are not a basis for
strong claims about real populations. If we later claim public-opinion
prediction, we need a grounding pipeline:

- survey-derived or interview-derived agent profiles;
- held-out validation questions or behaviors;
- calibration of content exposure and action rates;
- comparisons against human baseline reliability.

## Chuang et al.: Controls for LLM Opinion Dynamics

Chuang et al. simulate opinion dynamics with LLM agents in a controlled
Twitter-like setting. Their key methodological contribution for us is the
control design. They include no-interaction controls where agents repeatedly
report opinions without communicating. This isolates LLM drift from social
influence.

They also show that raw LLM agents can move toward accurate or scientific
consensus even without interaction, and that confirmation-bias prompting changes
fragmentation. This directly addresses a risk in our previous toy herding work:
if a result looks like convergence, it may reflect model prior, not crowd
dynamics.

For our social-media module, every paid LLM pilot should include:

- no-feed/no-interaction control;
- deterministic mock-policy baseline;
- stable initial content and graph seed;
- identical agents under different feed algorithms;
- logs that separate agent prior drift from social exposure effects.

## Rumor and Intervention Work

The rumor-spreading paper and repository vary network structures, personas,
spreading schemes, patient-zero selection, fact-checking, and filtering. The
important design lesson is that spread is a joint property of graph topology,
exposure policy, seed-node choice, and agent behavior. It is not just a property
of a message.

For Instagram-like experiments, this maps to:

- influencer versus ordinary-account seed posts;
- different initial engagement counts;
- recommendation boosts or suppression;
- follow/unfollow rewiring after controversial exposure;
- fact-check or counter-post interventions;
- cascade reach, depth, breadth, and time-to-peak metrics.

## What This Means for Our Simulator

The next implementation should add a `social_media_dynamics` or
`instagram_social_dynamics` experiment family with explicit platform mechanics.
It should not replace the event-driven conversation work. The existing
event-driven runner remains valuable for group-chat and public/private stance
experiments. The new module should cover the platform-mediated slice: feed,
likes, follows, unfollows, posts, DMs, and exposure metrics.

The v0 action set should be intentionally small:

- `like_post`
- `follow_user`
- `unfollow_user`
- `send_dm`
- `create_post`
- `do_nothing`

The user's phrase "follow button" appeared twice in discussion; in this design,
the second low-cost platform action is interpreted as `like_post`, because
follow/unfollow are already graph-changing actions and likes are the canonical
low-cost endorsement signal.

The v0 platform should include:

- user profiles;
- dynamic user state;
- posts;
- follow graph;
- feed items and recommendation scores;
- DM threads;
- action trace;
- metrics artifact;
- replay metadata.

The v0 recommender should be deterministic and inspectable:

```text
score =
  following_bonus
  + interest_similarity_weight * interest_similarity
  + engagement_weight * log(1 + likes + comments)
  + recency_weight * recency_decay
  + popularity_weight * log(1 + creator_followers)
  + controversy_weight * stance_distance
```

This lets us run experiments where we change the feed algorithm without changing
agents or content. LLMs should see the feed as a user would see it; they do not
need to be told that they are in a social-network experiment.

## Initial Experiment Ideas After Implementation

### Experiment 1: Feed Algorithm Comparison

Run identical agents, posts, and initial graph under four feed policies:
chronological following feed, engagement-ranked feed, interest-homophily feed,
and bridging feed. Measure opinion diversity, exposure diversity, interaction
across stance groups, likes, follows, unfollows, and public-post volume.

This is more interesting than "neighbors A,A,A,B,B imply A" because the social
effect comes from algorithmic exposure and platform actions, not a visible vote
count.

### Experiment 2: Low-Cost Endorsement Cascade

Seed the same post with no visible likes, moderate visible likes, or high visible
likes. Agents see normal feed cards, not a prompt saying "simulate herding."
Measure whether visible endorsement changes like probability, follow probability,
DM discussion, and subsequent public posts.

This directly studies crowd psychology, but through natural platform affordances.

### Experiment 3: Public Silence, Private Backchannel

Introduce a socially risky event or controversial post. Allow agents to like,
post, do nothing, or send DMs. Measure whether agents avoid public posting while
using private DMs to coordinate, vent, or seek reassurance.

This continues the public/private shock work, but with more realistic platform
actions and explicit relationship/DM threads.

### Experiment 4: Follow Graph Rewiring After Shock

Expose agents to controversial posts from weak ties and strong ties. Measure
follow/unfollow churn, homophily changes, echo-chamber index, and stance
polarization. This tests whether the platform graph itself changes after a
social shock.

### Experiment 5: Influencer Versus Ordinary Seed Account

Inject the same content from a high-follower account and from an ordinary
account. Keep text constant. Measure reach, engagement, follow changes, and
stance movement. This isolates source status from message content.

## Non-Goals for the First Implementation

- No UI or browser automation.
- No images, stories, reels, comments, bookmarks, shares, ads, or live streams
  in v0.
- No claim that synthetic agents predict real Instagram users.
- No paid large-scale LLM sweep until the mock runner and metrics are stable.
- No uncontrolled multi-day free-form society simulation.

## Design Decision

Build a new platform-mediated social-media experiment module. The first version
should be mock-policy capable, deterministic, replayable, and testable. It should
be ready for a tiny paid LLM pilot only after the platform state, action schema,
recommender, and metrics are independently verified.
