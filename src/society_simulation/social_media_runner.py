from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import random
from typing import Any

from society_simulation.social_media_config import InstagramSocialDynamicsConfig
from society_simulation.social_media_feed import build_feed
from society_simulation.social_media_metrics import compute_social_media_metrics
from society_simulation.social_media_models import (
    DirectMessage,
    FeedItem,
    FollowEdge,
    PlatformAction,
    SocialMediaPost,
    SocialMediaUserState,
    SocialMediaWorld,
)
from society_simulation.social_media_policy import MockSocialMediaPolicy
from society_simulation.social_media_replay import SocialMediaReplayWriter
from society_simulation.social_media_seed import build_initial_world


@dataclass(frozen=True)
class SocialMediaRunResult:
    final_world: SocialMediaWorld
    feed_items: tuple[FeedItem, ...]
    actions: tuple[PlatformAction, ...]
    dm_messages: tuple[DirectMessage, ...]
    states_by_tick: tuple[tuple[SocialMediaUserState, ...], ...]
    metrics: dict[str, Any]
    output_dir: Path


def run_instagram_social_dynamics(
    config: InstagramSocialDynamicsConfig,
) -> SocialMediaRunResult:
    config.validate()
    rng = random.Random(config.seed)
    policy = _build_policy(config)
    world = build_initial_world(config)
    states_by_tick: list[tuple[SocialMediaUserState, ...]] = [world.states]
    feed_items: list[FeedItem] = []
    actions: list[PlatformAction] = []
    dm_messages: list[DirectMessage] = []
    initial_edges = world.follow_edges
    for tick in range(1, config.ticks + 1):
        current_states = {state.user_id: state for state in world.states}
        for profile in world.profiles:
            if rng.random() > config.activation_probability * profile.activity_rate:
                continue
            feed = build_feed(
                world=world,
                viewer_id=profile.user_id,
                tick=tick,
                feed_size=config.feed_size,
                feed_policy=config.feed_policy,
                seed=config.seed,
            )
            feed_items.extend(feed)
            action = policy.decide(
                profile=profile,
                state=current_states[profile.user_id],
                feed=feed,
                tick=tick,
            )
            world, dm = apply_action(world, action)
            actions.append(action)
            if dm is not None:
                dm_messages.append(dm)
        states_by_tick.append(world.states)
    metrics = compute_social_media_metrics(
        initial_edges=initial_edges,
        final_world=world,
        feed_items=tuple(feed_items),
        actions=tuple(actions),
        dm_messages=tuple(dm_messages),
        states_by_tick=tuple(states_by_tick),
    )
    usage_summary = getattr(policy, "usage_summary", None)
    if callable(usage_summary):
        metrics["llm_usage"] = usage_summary()
    audit_records = getattr(policy, "audit_records", None)
    llm_decisions = audit_records() if callable(audit_records) else ()
    output_dir = SocialMediaReplayWriter(config).write(
        final_world=world,
        initial_edges=initial_edges,
        feed_items=tuple(feed_items),
        actions=tuple(actions),
        dm_messages=tuple(dm_messages),
        states_by_tick=tuple(states_by_tick),
        metrics=metrics,
        llm_decisions=llm_decisions,
    )
    return SocialMediaRunResult(
        final_world=world,
        feed_items=tuple(feed_items),
        actions=tuple(actions),
        dm_messages=tuple(dm_messages),
        states_by_tick=tuple(states_by_tick),
        metrics=metrics,
        output_dir=output_dir,
    )


def _build_policy(config: InstagramSocialDynamicsConfig) -> MockSocialMediaPolicy:
    policy_type = str(config.update_policy["type"])
    if policy_type == "mock_social":
        return MockSocialMediaPolicy(
            response_style=str(config.update_policy.get("response_style", "balanced")),
        )
    raise ValueError("unsupported social media update_policy type")


def apply_action(
    world: SocialMediaWorld,
    action: PlatformAction,
) -> tuple[SocialMediaWorld, DirectMessage | None]:
    posts = list(world.posts)
    edges = {
        (edge.follower_id, edge.followed_id): edge.created_tick
        for edge in world.follow_edges
    }
    dm: DirectMessage | None = None
    if action.action_type == "like_post" and action.post_id is not None:
        posts = [post.with_like() if post.post_id == action.post_id else post for post in posts]
    elif action.action_type == "follow_user" and action.target_user_id is not None:
        if action.target_user_id != action.user_id:
            edges[(action.user_id, action.target_user_id)] = action.tick
    elif action.action_type == "unfollow_user" and action.target_user_id is not None:
        edges.pop((action.user_id, action.target_user_id), None)
    elif action.action_type == "send_dm" and action.target_user_id is not None and action.text:
        dm = DirectMessage(
            tick=action.tick,
            sender_id=action.user_id,
            recipient_id=action.target_user_id,
            text=action.text,
            topic=action.topic,
            stance=action.stance,
        )
    elif action.action_type == "create_post" and action.text and action.topic:
        posts.append(
            SocialMediaPost(
                post_id=f"post-generated-{action.tick}-{action.user_id}-{len(posts)}",
                author_id=action.user_id,
                topic=action.topic,
                stance=action.stance if action.stance is not None else 0.0,
                text=action.text,
                created_tick=action.tick,
                like_count=0,
                reply_count=0,
                seed_post=False,
            )
        )
    next_states = tuple(
        replace(state, tick=action.tick, last_action_type=action.action_type)
        if state.user_id == action.user_id
        else state
        for state in world.states
    )
    next_edges = tuple(
        FollowEdge(follower_id=follower, followed_id=followed, created_tick=created_tick)
        for (follower, followed), created_tick in sorted(edges.items())
        if follower != followed
    )
    return (
        SocialMediaWorld(
            profiles=world.profiles,
            states=next_states,
            posts=tuple(posts),
            follow_edges=next_edges,
        ),
        dm,
    )
