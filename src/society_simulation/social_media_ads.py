from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace

from society_simulation.social_media_config import AdCampaignConfig
from society_simulation.social_media_models import FeedItem, SocialMediaPost, SocialMediaWorld


@dataclass(frozen=True)
class AdImpression:
    tick: int
    viewer_id: int
    campaign_id: str
    advertiser_id: int
    post_id: str
    targeting: str
    source_reason: str
    seen_count: int
    frequency_cap: int
    topic: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AdDeliveryState:
    remaining_budget_by_campaign: dict[str, int]
    seen_by_user_campaign: dict[tuple[int, str], int]
    impressions: list[AdImpression]


def initialize_ad_delivery(
    world: SocialMediaWorld,
    campaigns: Sequence[AdCampaignConfig],
) -> tuple[SocialMediaWorld, AdDeliveryState]:
    campaign_posts = tuple(
        _campaign_post(campaign)
        for campaign in campaigns
        if campaign.ad_condition in {"organic_post", "sponsored_ad"}
    )
    next_world = replace(world, posts=(*world.posts, *campaign_posts))
    return (
        next_world,
        AdDeliveryState(
            remaining_budget_by_campaign={
                campaign.campaign_id: (
                    campaign.budget_impressions
                    if campaign.ad_condition == "sponsored_ad"
                    else 0
                )
                for campaign in campaigns
            },
            seen_by_user_campaign={},
            impressions=[],
        ),
    )


def insert_sponsored_ads(
    *,
    world: SocialMediaWorld,
    viewer_id: int,
    tick: int,
    feed_items: Sequence[FeedItem],
    feed_size: int,
    campaigns: Sequence[AdCampaignConfig],
    state: AdDeliveryState,
) -> tuple[FeedItem, ...]:
    base_feed = tuple(feed_items)
    for campaign in campaigns:
        if not _can_deliver(campaign, world, viewer_id, tick, state):
            continue
        post = world.post_by_id().get(_campaign_post_id(campaign))
        if post is None:
            continue
        seen_key = (viewer_id, campaign.campaign_id)
        seen_count = state.seen_by_user_campaign.get(seen_key, 0) + 1
        state.seen_by_user_campaign[seen_key] = seen_count
        state.remaining_budget_by_campaign[campaign.campaign_id] -= 1
        source_reason = (
            "sponsored_targeted"
            if campaign.targeting == "interest_targeted"
            else "sponsored_broad"
        )
        state.impressions.append(
            AdImpression(
                tick=tick,
                viewer_id=viewer_id,
                campaign_id=campaign.campaign_id,
                advertiser_id=campaign.advertiser_id,
                post_id=post.post_id,
                targeting=campaign.targeting,
                source_reason=source_reason,
                seen_count=seen_count,
                frequency_cap=campaign.frequency_cap,
                topic=campaign.topic,
            )
        )
        return _prepend_sponsored_item(
            world=world,
            viewer_id=viewer_id,
            tick=tick,
            post=post,
            campaign=campaign,
            source_reason=source_reason,
            seen_count=seen_count,
            feed_items=base_feed,
            feed_size=feed_size,
        )
    return base_feed


def _can_deliver(
    campaign: AdCampaignConfig,
    world: SocialMediaWorld,
    viewer_id: int,
    tick: int,
    state: AdDeliveryState,
) -> bool:
    if campaign.ad_condition != "sponsored_ad":
        return False
    if viewer_id == campaign.advertiser_id:
        return False
    if tick < campaign.start_tick or tick > campaign.end_tick:
        return False
    if state.remaining_budget_by_campaign.get(campaign.campaign_id, 0) <= 0:
        return False
    if state.seen_by_user_campaign.get((viewer_id, campaign.campaign_id), 0) >= (
        campaign.frequency_cap
    ):
        return False
    if campaign.targeting == "interest_targeted" and not _viewer_matches_targeting(
        campaign,
        world,
        viewer_id,
    ):
        return False
    return True


def _viewer_matches_targeting(
    campaign: AdCampaignConfig,
    world: SocialMediaWorld,
    viewer_id: int,
) -> bool:
    profile = world.profile_by_id()[viewer_id]
    target_topics = set(campaign.targeting_topics or (campaign.topic,))
    return bool(target_topics & set(profile.interests))


def _prepend_sponsored_item(
    *,
    world: SocialMediaWorld,
    viewer_id: int,
    tick: int,
    post: SocialMediaPost,
    campaign: AdCampaignConfig,
    source_reason: str,
    seen_count: int,
    feed_items: tuple[FeedItem, ...],
    feed_size: int,
) -> tuple[FeedItem, ...]:
    max_score = max((item.score for item in feed_items), default=0.0)
    sponsored = FeedItem(
        tick=tick,
        viewer_id=viewer_id,
        post_id=post.post_id,
        author_id=post.author_id,
        score=round(max_score + 1.0, 6),
        rank=0,
        source="sponsored",
        reason=source_reason,
        visible_like_count=post.like_count,
        topic=post.topic,
        text=post.text,
        author_handle=world.profile_by_id()[post.author_id].handle,
        campaign_id=campaign.campaign_id,
        is_sponsored=True,
        advertiser_id=campaign.advertiser_id,
        ad_seen_count=seen_count,
    )
    deduped = tuple(item for item in feed_items if item.post_id != post.post_id)
    reranked = tuple(
        replace(item, rank=rank)
        for rank, item in enumerate((sponsored, *deduped)[:feed_size])
    )
    return reranked


def _campaign_post(campaign: AdCampaignConfig) -> SocialMediaPost:
    return SocialMediaPost(
        post_id=_campaign_post_id(campaign),
        author_id=campaign.advertiser_id,
        topic=campaign.topic,
        stance=campaign.stance,
        text=campaign.creative_text,
        created_tick=campaign.start_tick,
        like_count=campaign.sponsored_like_count,
        reply_count=0,
        seed_post=False,
        campaign_id=campaign.campaign_id,
        is_ad=True,
    )


def _campaign_post_id(campaign: AdCampaignConfig) -> str:
    return f"ad-{campaign.campaign_id}"
