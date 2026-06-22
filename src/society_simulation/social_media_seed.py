from __future__ import annotations

import random

from society_simulation.social_media_config import InstagramSocialDynamicsConfig
from society_simulation.social_media_models import (
    FollowEdge,
    SocialMediaPost,
    SocialMediaUserProfile,
    SocialMediaUserState,
    SocialMediaWorld,
)

_NAMES = (
    "Minho Park",
    "Carlos Rivera",
    "Aisha Khan",
    "Nora Lee",
    "Sam Patel",
    "Jules Martin",
    "Maya Chen",
    "Theo Brooks",
)
_CLUSTERS = ("urbanists", "parents", "small_business", "students")
_STYLES = ("dry and specific", "warm and personal", "skeptical", "brief and witty")


def build_initial_world(config: InstagramSocialDynamicsConfig) -> SocialMediaWorld:
    rng = random.Random(config.seed)
    profiles = tuple(_build_profile(index, config, rng) for index in range(config.num_users))
    states = tuple(_initial_state(profile) for profile in profiles)
    follow_edges = _build_follow_edges(config, profiles, rng)
    posts = (
        *_build_historical_posts(config, profiles, rng),
        *_build_configured_seed_posts(config),
    )
    return SocialMediaWorld(
        profiles=profiles,
        states=states,
        posts=posts,
        follow_edges=follow_edges,
    )


def _build_profile(
    user_id: int,
    config: InstagramSocialDynamicsConfig,
    rng: random.Random,
) -> SocialMediaUserProfile:
    name = _NAMES[user_id % len(_NAMES)]
    handle = name.lower().replace(" ", "_") + f"_{user_id}"
    interests = tuple(rng.sample(list(config.topics), k=min(2, len(config.topics))))
    stance = round(rng.uniform(-0.8, 0.8), 3)
    return SocialMediaUserProfile(
        user_id=user_id,
        handle=handle,
        display_name=name,
        bio=f"Posts about {', '.join(interests)}.",
        interests=interests,
        home_cluster=_CLUSTERS[user_id % len(_CLUSTERS)],
        initial_stance=stance,
        activity_rate=round(rng.uniform(0.55, 0.95), 3),
        post_rate=round(rng.uniform(0.10, 0.35), 3),
        privacy_preference=round(rng.uniform(0.10, 0.75), 3),
        conformity=round(rng.uniform(0.10, 0.90), 3),
        skepticism=round(rng.uniform(0.10, 0.90), 3),
        conflict_tolerance=round(rng.uniform(0.10, 0.90), 3),
        status_weight=round(rng.uniform(0.05, 0.80), 3),
        posting_style=_STYLES[user_id % len(_STYLES)],
    )


def _initial_state(profile: SocialMediaUserProfile) -> SocialMediaUserState:
    return SocialMediaUserState(
        user_id=profile.user_id,
        tick=0,
        stance=profile.initial_stance,
        confidence=0.55,
        salience=0.45,
        mood="neutral",
        perceived_majority=0.0,
        social_fatigue=0.0,
        last_action_type="initial",
    )


def _build_follow_edges(
    config: InstagramSocialDynamicsConfig,
    profiles: tuple[SocialMediaUserProfile, ...],
    rng: random.Random,
) -> tuple[FollowEdge, ...]:
    mean_following = int(config.seed_generator["mean_following"])
    homophily_weight = float(config.seed_generator["homophily_weight"])
    popularity_weight = float(config.seed_generator["popularity_weight"])
    random_tie_probability = float(config.seed_generator["random_tie_probability"])
    mutual_follow_probability = float(config.seed_generator["mutual_follow_probability"])
    popularity = {profile.user_id: rng.random() for profile in profiles}
    edges: set[tuple[int, int]] = set()
    for follower in profiles:
        candidates = [profile for profile in profiles if profile.user_id != follower.user_id]
        scored = sorted(
            candidates,
            key=lambda candidate: (
                -_tie_score(
                    follower,
                    candidate,
                    homophily_weight,
                    popularity_weight,
                    random_tie_probability,
                    popularity[candidate.user_id],
                    rng,
                ),
                candidate.user_id,
            ),
        )
        for followed in scored[: min(mean_following, len(scored))]:
            edges.add((follower.user_id, followed.user_id))
            if rng.random() < mutual_follow_probability:
                edges.add((followed.user_id, follower.user_id))
    return tuple(
        FollowEdge(follower_id=follower_id, followed_id=followed_id, created_tick=0)
        for follower_id, followed_id in sorted(edges)
        if follower_id != followed_id
    )


def _tie_score(
    follower: SocialMediaUserProfile,
    candidate: SocialMediaUserProfile,
    homophily_weight: float,
    popularity_weight: float,
    random_tie_probability: float,
    candidate_popularity: float,
    rng: random.Random,
) -> float:
    shared_interests = len(set(follower.interests) & set(candidate.interests))
    stance_similarity = 1.0 - min(1.0, abs(follower.initial_stance - candidate.initial_stance))
    homophily = (shared_interests / max(1, len(follower.interests))) + stance_similarity
    return (
        homophily_weight * homophily
        + popularity_weight * candidate_popularity
        + random_tie_probability * rng.random()
    )


def _build_historical_posts(
    config: InstagramSocialDynamicsConfig,
    profiles: tuple[SocialMediaUserProfile, ...],
    rng: random.Random,
) -> tuple[SocialMediaPost, ...]:
    posts: list[SocialMediaPost] = []
    for profile in profiles:
        for index in range(config.historical_posts_per_user):
            topic = profile.interests[index % len(profile.interests)]
            post_id = f"post-{profile.user_id}-{index}"
            stance = max(-1.0, min(1.0, profile.initial_stance + rng.uniform(-0.15, 0.15)))
            posts.append(
                SocialMediaPost(
                    post_id=post_id,
                    author_id=profile.user_id,
                    topic=topic,
                    stance=round(stance, 3),
                    text=(
                        f"{profile.display_name} shares a {profile.posting_style} "
                        f"note about {topic}."
                    ),
                    created_tick=-config.historical_posts_per_user + index,
                    like_count=rng.randint(0, 9),
                    reply_count=0,
                    seed_post=True,
                )
            )
    return tuple(posts)


def _build_configured_seed_posts(
    config: InstagramSocialDynamicsConfig,
) -> tuple[SocialMediaPost, ...]:
    return tuple(
        SocialMediaPost(
            post_id=str(post["post_id"]),
            author_id=int(post["author_id"]),
            topic=str(post["topic"]),
            stance=float(post["stance"]),
            text=str(post["text"]),
            created_tick=int(post["created_tick"]),
            like_count=int(post["like_count"]),
            reply_count=int(post.get("reply_count", 0)),
            seed_post=True,
        )
        for post in config.seed_posts
    )
