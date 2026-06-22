from __future__ import annotations

from collections.abc import Mapping
from math import log
import random

from society_simulation.social_media_models import FeedItem, SocialMediaPost, SocialMediaWorld


def build_feed(
    *,
    world: SocialMediaWorld,
    viewer_id: int,
    tick: int,
    feed_size: int,
    feed_policy: Mapping[str, object],
    seed: int,
) -> tuple[FeedItem, ...]:
    policy_type = str(feed_policy["type"])
    if policy_type == "no_feed_control":
        return ()
    followed_ids = _followed_ids(world, viewer_id)
    candidates = _candidate_posts(world, viewer_id, followed_ids, policy_type, tick)
    rng = random.Random(seed + tick * 1009 + viewer_id * 9173)
    scored = [
        _score_post(
            world=world,
            viewer_id=viewer_id,
            post=post,
            followed_ids=followed_ids,
            tick=tick,
            feed_policy=feed_policy,
            rng=rng,
        )
        for post in candidates
    ]
    ranked = sorted(scored, key=lambda item: (-item[0], item[1].post_id))
    return tuple(
        FeedItem(
            tick=tick,
            viewer_id=viewer_id,
            post_id=post.post_id,
            author_id=post.author_id,
            score=round(score, 6),
            rank=rank,
            source="following" if post.author_id in followed_ids else "explore",
            reason=reason,
            visible_like_count=post.like_count,
            topic=post.topic,
            text=post.text,
            author_handle=world.profile_by_id()[post.author_id].handle,
            campaign_id=post.campaign_id,
            advertiser_id=post.author_id if post.is_ad else None,
        )
        for rank, (score, post, reason) in enumerate(ranked[:feed_size])
    )


def _followed_ids(world: SocialMediaWorld, viewer_id: int) -> set[int]:
    return {
        edge.followed_id
        for edge in world.follow_edges
        if edge.follower_id == viewer_id
    }


def _candidate_posts(
    world: SocialMediaWorld,
    viewer_id: int,
    followed_ids: set[int],
    policy_type: str,
    tick: int,
) -> tuple[SocialMediaPost, ...]:
    if policy_type == "chronological_following":
        return tuple(
            sorted(
                (
                    post
                    for post in world.posts
                    if post.author_id in followed_ids and post.created_tick <= tick
                ),
                key=lambda post: (-post.created_tick, post.post_id),
            )
        )
    return tuple(
        post
        for post in world.posts
        if post.author_id != viewer_id and post.created_tick <= tick
    )


def _score_post(
    *,
    world: SocialMediaWorld,
    viewer_id: int,
    post: SocialMediaPost,
    followed_ids: set[int],
    tick: int,
    feed_policy: Mapping[str, object],
    rng: random.Random,
) -> tuple[float, SocialMediaPost, str]:
    profile_by_id = world.profile_by_id()
    state_by_id = world.state_by_id()
    viewer_profile = profile_by_id[viewer_id]
    viewer_state = state_by_id[viewer_id]
    author_followers = sum(1 for edge in world.follow_edges if edge.followed_id == post.author_id)
    following_bonus = float(feed_policy["following_bonus"]) if post.author_id in followed_ids else 0.0
    interest_similarity = _interest_similarity(viewer_profile.interests, (post.topic,))
    stance_distance = min(1.0, abs(viewer_state.stance - post.stance))
    stance_similarity = 1.0 - stance_distance
    recency_decay = 1.0 / (1.0 + max(0, tick - post.created_tick))
    policy_type = str(feed_policy["type"])
    score = (
        following_bonus
        + _effective_weight(feed_policy, "interest_similarity_weight", policy_type)
        * interest_similarity
        + _effective_weight(feed_policy, "stance_similarity_weight", policy_type)
        * stance_similarity
        + _effective_weight(feed_policy, "engagement_weight", policy_type)
        * log(1 + post.like_count)
        + float(feed_policy["recency_weight"]) * recency_decay
        + float(feed_policy["creator_popularity_weight"]) * log(1 + author_followers)
        + _effective_controversy_weight(feed_policy, policy_type) * stance_distance
        + float(feed_policy["noise_weight"]) * rng.random()
    )
    reason = (
        f"following_bonus={following_bonus:.2f} "
        f"interest_similarity={interest_similarity:.2f} "
        f"stance_similarity={stance_similarity:.2f} "
        f"engagement={log(1 + post.like_count):.2f}"
    )
    return score, post, reason


def _effective_weight(
    feed_policy: Mapping[str, object],
    field: str,
    policy_type: str,
) -> float:
    value = float(feed_policy[field])
    if policy_type == "interest_homophily" and field in {
        "interest_similarity_weight",
        "stance_similarity_weight",
    }:
        return value * 1.75
    if policy_type == "engagement_ranked" and field == "engagement_weight":
        return value * 1.25
    if policy_type == "bridging" and field == "stance_similarity_weight":
        return 0.0
    return value


def _effective_controversy_weight(feed_policy: Mapping[str, object], policy_type: str) -> float:
    if policy_type == "bridging":
        return max(float(feed_policy["controversy_weight"]), 0.75)
    return float(feed_policy["controversy_weight"])


def _interest_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)
