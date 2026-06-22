from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from statistics import mean, variance
from typing import Any

from society_simulation.social_media_ads import AdImpression
from society_simulation.social_media_config import AdCampaignConfig
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
    ad_impressions: tuple[AdImpression, ...] = (),
    ad_campaigns: tuple[AdCampaignConfig, ...] = (),
    ad_remaining_budget_by_campaign: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    action_counts = Counter(action.action_type for action in actions)
    non_noop_action_count = sum(1 for action in actions if action.action_type != "do_nothing")
    stance_values = [state.stance for state in final_world.states]
    initial_edge_set = {(edge.follower_id, edge.followed_id) for edge in initial_edges}
    final_edge_set = {(edge.follower_id, edge.followed_id) for edge in final_world.follow_edges}
    metrics = {
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
    metrics.update(
        _campaign_metrics(
            final_world=final_world,
            feed_items=feed_items,
            actions=actions,
            ad_impressions=ad_impressions,
            ad_campaigns=ad_campaigns,
            ad_remaining_budget_by_campaign=ad_remaining_budget_by_campaign or {},
        )
    )
    return metrics


def _exposure_diversity(feed_items: tuple[FeedItem, ...]) -> float:
    if not feed_items:
        return 0.0
    authors_by_viewer: dict[int, set[int]] = {}
    for item in feed_items:
        authors_by_viewer.setdefault(item.viewer_id, set()).add(item.author_id)
    return round(mean(len(authors) for authors in authors_by_viewer.values()), 6)


def _campaign_metrics(
    *,
    final_world: SocialMediaWorld,
    feed_items: tuple[FeedItem, ...],
    actions: tuple[PlatformAction, ...],
    ad_impressions: tuple[AdImpression, ...],
    ad_campaigns: tuple[AdCampaignConfig, ...],
    ad_remaining_budget_by_campaign: Mapping[str, int],
) -> dict[str, Any]:
    campaign_by_id = {campaign.campaign_id: campaign for campaign in ad_campaigns}
    campaign_post_ids = {
        post.post_id
        for post in final_world.posts
        if post.campaign_id is not None or post.is_ad
    }
    advertiser_ids = {campaign.advertiser_id for campaign in ad_campaigns}
    campaign_topics = {campaign.topic for campaign in ad_campaigns}
    min_campaign_start = min(
        (
            campaign.start_tick
            for campaign in ad_campaigns
            if campaign.ad_condition != "no_ad"
        ),
        default=None,
    )
    paid_viewers = {impression.viewer_id for impression in ad_impressions}
    organic_items = tuple(
        item
        for item in feed_items
        if item.campaign_id is not None and not item.is_sponsored
    )
    organic_viewers = {item.viewer_id for item in organic_items}
    ad_frequency_by_viewer: Counter[int] = Counter(
        impression.viewer_id for impression in ad_impressions
    )
    frequency_values = list(ad_frequency_by_viewer.values())
    remaining_budget = sum(ad_remaining_budget_by_campaign.values())
    total_ad_budget = sum(
        campaign.budget_impressions
        for campaign in ad_campaigns
        if campaign.ad_condition == "sponsored_ad"
    )
    return {
        "paid_impression_count": len(ad_impressions),
        "unique_paid_reach": len(paid_viewers),
        "organic_ad_impression_count": len(organic_items),
        "unique_organic_ad_reach": len(organic_viewers),
        "unique_total_ad_reach": len(paid_viewers | organic_viewers),
        "relevant_paid_reach": _relevant_paid_reach(
            final_world,
            ad_impressions,
            campaign_by_id,
        ),
        "relevant_total_reach": _relevant_total_reach(
            final_world,
            ad_impressions,
            organic_items,
            campaign_by_id,
        ),
        "mean_ad_frequency": (
            round(mean(frequency_values), 6) if frequency_values else 0.0
        ),
        "max_ad_frequency": max(frequency_values, default=0),
        "frequency_cap_hit_count": _frequency_cap_hit_count(ad_impressions),
        "ad_like_count": sum(
            1
            for action in actions
            if action.action_type == "like_post" and action.post_id in campaign_post_ids
        ),
        "advertiser_follow_count": sum(
            1
            for action in actions
            if action.action_type == "follow_user"
            and action.target_user_id in advertiser_ids
            and _is_after_campaign_start(action.tick, min_campaign_start)
        ),
        "ad_dm_count": sum(
            1
            for action in actions
            if action.action_type == "send_dm"
            and action.target_user_id in advertiser_ids
            and _is_after_campaign_start(action.tick, min_campaign_start)
        ),
        "ad_generated_post_count": sum(
            1
            for action in actions
            if action.action_type == "create_post"
            and action.topic in campaign_topics
            and _is_after_campaign_start(action.tick, min_campaign_start)
        ),
        "ad_negative_action_count": 0,
        "paid_to_organic_spillover_rate": (
            round(len(organic_items) / len(ad_impressions), 6)
            if ad_impressions
            else 0.0
        ),
        "ad_delivery_exhausted_budget": bool(total_ad_budget and remaining_budget == 0),
        "ad_delivery_remaining_budget": remaining_budget,
        "burn_in_action_mean": _burn_in_action_mean(actions, min_campaign_start),
        "burn_in_follow_churn": _burn_in_follow_churn(actions, min_campaign_start),
        "burn_in_exposure_diversity": _burn_in_exposure_diversity(
            feed_items,
            min_campaign_start,
        ),
    }


def _frequency_cap_hit_count(ad_impressions: tuple[AdImpression, ...]) -> int:
    max_seen_by_pair: dict[tuple[int, str], tuple[int, int]] = {}
    for impression in ad_impressions:
        key = (impression.viewer_id, impression.campaign_id)
        seen_count, frequency_cap = max_seen_by_pair.get(key, (0, impression.frequency_cap))
        max_seen_by_pair[key] = (max(seen_count, impression.seen_count), frequency_cap)
    return sum(1 for seen_count, frequency_cap in max_seen_by_pair.values() if seen_count >= frequency_cap)


def _relevant_paid_reach(
    world: SocialMediaWorld,
    ad_impressions: tuple[AdImpression, ...],
    campaign_by_id: dict[str, AdCampaignConfig],
) -> int:
    relevant_viewers: set[int] = set()
    for impression in ad_impressions:
        if _viewer_is_relevant(world, impression.viewer_id, campaign_by_id.get(impression.campaign_id)):
            relevant_viewers.add(impression.viewer_id)
    return len(relevant_viewers)


def _relevant_total_reach(
    world: SocialMediaWorld,
    ad_impressions: tuple[AdImpression, ...],
    organic_items: tuple[FeedItem, ...],
    campaign_by_id: dict[str, AdCampaignConfig],
) -> int:
    relevant_viewers: set[int] = set()
    for impression in ad_impressions:
        if _viewer_is_relevant(world, impression.viewer_id, campaign_by_id.get(impression.campaign_id)):
            relevant_viewers.add(impression.viewer_id)
    for item in organic_items:
        if item.campaign_id is not None and _viewer_is_relevant(
            world,
            item.viewer_id,
            campaign_by_id.get(item.campaign_id),
        ):
            relevant_viewers.add(item.viewer_id)
    return len(relevant_viewers)


def _viewer_is_relevant(
    world: SocialMediaWorld,
    viewer_id: int,
    campaign: AdCampaignConfig | None,
) -> bool:
    if campaign is None:
        return False
    profile = world.profile_by_id()[viewer_id]
    target_topics = set(campaign.targeting_topics or (campaign.topic,))
    return bool(target_topics & set(profile.interests))


def _is_after_campaign_start(tick: int, min_campaign_start: int | None) -> bool:
    return min_campaign_start is not None and tick >= min_campaign_start


def _burn_in_action_mean(
    actions: tuple[PlatformAction, ...],
    min_campaign_start: int | None,
) -> float:
    if min_campaign_start is None or min_campaign_start <= 1:
        return 0.0
    burn_in_ticks = min_campaign_start - 1
    action_count = sum(
        1
        for action in actions
        if action.tick < min_campaign_start and action.action_type != "do_nothing"
    )
    return round(action_count / burn_in_ticks, 6)


def _burn_in_follow_churn(
    actions: tuple[PlatformAction, ...],
    min_campaign_start: int | None,
) -> int:
    if min_campaign_start is None:
        return 0
    return sum(
        1
        for action in actions
        if action.tick < min_campaign_start
        and action.action_type in {"follow_user", "unfollow_user"}
    )


def _burn_in_exposure_diversity(
    feed_items: tuple[FeedItem, ...],
    min_campaign_start: int | None,
) -> float:
    if min_campaign_start is None:
        return 0.0
    return _exposure_diversity(
        tuple(item for item in feed_items if item.tick < min_campaign_start)
    )
