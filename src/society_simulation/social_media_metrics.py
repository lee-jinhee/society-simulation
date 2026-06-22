from __future__ import annotations

from collections import Counter
from statistics import mean, variance
from typing import Any

from society_simulation.social_media_models import (
    DirectMessage,
    FeedItem,
    FollowEdge,
    PlatformAction,
    SocialMediaUserState,
    SocialMediaWorld,
)


def compute_social_media_metrics(
    *,
    initial_edges: tuple[FollowEdge, ...],
    final_world: SocialMediaWorld,
    feed_items: tuple[FeedItem, ...],
    actions: tuple[PlatformAction, ...],
    dm_messages: tuple[DirectMessage, ...],
    states_by_tick: tuple[tuple[SocialMediaUserState, ...], ...],
) -> dict[str, Any]:
    action_counts = Counter(action.action_type for action in actions)
    non_noop_action_count = sum(1 for action in actions if action.action_type != "do_nothing")
    stance_values = [state.stance for state in final_world.states]
    initial_edge_set = {(edge.follower_id, edge.followed_id) for edge in initial_edges}
    final_edge_set = {(edge.follower_id, edge.followed_id) for edge in final_world.follow_edges}
    return {
        "experiment_family": "instagram_social_dynamics",
        "user_count": len(final_world.profiles),
        "post_count": len(final_world.posts),
        "feed_impression_count": len(feed_items),
        "action_count": non_noop_action_count,
        "action_counts": dict(action_counts),
        "like_count": action_counts.get("like_post", 0),
        "dm_count": len(dm_messages),
        "follow_count": action_counts.get("follow_user", 0),
        "unfollow_count": action_counts.get("unfollow_user", 0),
        "initial_follow_edge_count": len(initial_edge_set),
        "final_follow_edge_count": len(final_edge_set),
        "follow_edge_delta": len(final_edge_set) - len(initial_edge_set),
        "new_follow_edge_count": len(final_edge_set - initial_edge_set),
        "removed_follow_edge_count": len(initial_edge_set - final_edge_set),
        "final_stance_mean": round(mean(stance_values), 6) if stance_values else 0.0,
        "final_stance_variance": (
            round(variance(stance_values), 6) if len(stance_values) > 1 else 0.0
        ),
        "exposure_diversity": _exposure_diversity(feed_items),
        "states_recorded": sum(len(states) for states in states_by_tick),
    }


def _exposure_diversity(feed_items: tuple[FeedItem, ...]) -> float:
    if not feed_items:
        return 0.0
    authors_by_viewer: dict[int, set[int]] = {}
    for item in feed_items:
        authors_by_viewer.setdefault(item.viewer_id, set()).add(item.author_id)
    return round(mean(len(authors) for authors in authors_by_viewer.values()), 6)
