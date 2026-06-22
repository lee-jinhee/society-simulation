from types import MappingProxyType

from society_simulation.social_media_ads import (
    initialize_ad_delivery,
    insert_sponsored_ads,
)
from society_simulation.social_media_config import AdCampaignConfig
from society_simulation.social_media_feed import build_feed
from society_simulation.social_media_models import (
    FeedItem,
    FollowEdge,
    SocialMediaPost,
    SocialMediaUserProfile,
    SocialMediaUserState,
    SocialMediaWorld,
)


def _profile(user_id: int, interests: tuple[str, ...]) -> SocialMediaUserProfile:
    return SocialMediaUserProfile(
        user_id=user_id,
        handle=f"user_{user_id}",
        display_name=f"User {user_id}",
        bio="bio",
        interests=interests,
        home_cluster="cluster",
        initial_stance=0.2,
        activity_rate=1.0,
        post_rate=0.2,
        privacy_preference=0.3,
        conformity=0.5,
        skepticism=0.4,
        conflict_tolerance=0.5,
        status_weight=0.2,
        posting_style="plain",
    )


def _world() -> SocialMediaWorld:
    profiles = (
        _profile(0, ("coffee", "food")),
        _profile(1, ("coffee", "commute")),
        _profile(2, ("sports",)),
    )
    states = tuple(
        SocialMediaUserState(
            user_id=profile.user_id,
            tick=0,
            stance=profile.initial_stance,
            confidence=0.5,
            salience=0.5,
            mood="neutral",
            perceived_majority=0.0,
            social_fatigue=0.0,
            last_action_type="initial",
        )
        for profile in profiles
    )
    posts = (
        SocialMediaPost("post-1", 1, "coffee", 0.2, "coffee post", 0, 2, 0, True),
        SocialMediaPost("post-2", 2, "sports", 0.0, "sports post", 0, 2, 0, True),
    )
    return SocialMediaWorld(
        profiles=profiles,
        states=states,
        posts=posts,
        follow_edges=(FollowEdge(1, 0, 0), FollowEdge(2, 0, 0)),
    )


def _campaign(**overrides: object) -> AdCampaignConfig:
    data = {
        "campaign_id": "maple_3rd_opening",
        "advertiser_id": 0,
        "ad_condition": "sponsored_ad",
        "creative_id": "discount_offer",
        "creative_text": "First 100 visitors get a free pastry with any drink.",
        "topic": "coffee",
        "stance": 0.2,
        "start_tick": 3,
        "end_tick": 6,
        "budget_impressions": 4,
        "frequency_cap": 2,
        "targeting": "interest_targeted",
        "sponsored_like_count": 25,
        "targeting_topics": ("coffee",),
    }
    data.update(overrides)
    return AdCampaignConfig(**data)  # type: ignore[arg-type]


def _feed(viewer_id: int, tick: int = 3) -> tuple[FeedItem, ...]:
    return (
        FeedItem(tick, viewer_id, "post-1", 1, 1.0, 0, "following", "x"),
        FeedItem(tick, viewer_id, "post-2", 2, 0.5, 1, "explore", "x"),
    )


def test_no_ad_campaign_creates_no_campaign_post_or_paid_impressions() -> None:
    world, state = initialize_ad_delivery(_world(), (_campaign(ad_condition="no_ad"),))

    feed = insert_sponsored_ads(
        world=world,
        viewer_id=1,
        tick=3,
        feed_items=_feed(1),
        feed_size=2,
        campaigns=(_campaign(ad_condition="no_ad"),),
        state=state,
    )

    assert all(post.campaign_id is None for post in world.posts)
    assert feed == _feed(1)
    assert state.impressions == []


def test_organic_post_creates_campaign_post_without_paid_impressions() -> None:
    campaign = _campaign(ad_condition="organic_post")
    world, state = initialize_ad_delivery(_world(), (campaign,))

    assert [post.post_id for post in world.posts if post.campaign_id == campaign.campaign_id] == [
        "ad-maple_3rd_opening"
    ]

    feed = insert_sponsored_ads(
        world=world,
        viewer_id=1,
        tick=3,
        feed_items=_feed(1),
        feed_size=2,
        campaigns=(campaign,),
        state=state,
    )

    assert feed == _feed(1)
    assert state.impressions == []


def test_organic_campaign_post_is_hidden_before_start_tick() -> None:
    campaign = _campaign(ad_condition="organic_post")
    world, _state = initialize_ad_delivery(_world(), (campaign,))
    feed_policy = MappingProxyType(
        {
            "type": "engagement_ranked",
            "following_bonus": 1.0,
            "interest_similarity_weight": 0.7,
            "stance_similarity_weight": 0.3,
            "engagement_weight": 0.8,
            "recency_weight": 0.5,
            "creator_popularity_weight": 0.2,
            "controversy_weight": 0.0,
            "noise_weight": 0.0,
            "explore_fraction": 0.5,
        }
    )

    before = build_feed(
        world=world,
        viewer_id=1,
        tick=2,
        feed_size=5,
        feed_policy=feed_policy,
        seed=1,
    )
    during = build_feed(
        world=world,
        viewer_id=1,
        tick=3,
        feed_size=5,
        feed_policy=feed_policy,
        seed=1,
    )

    assert "ad-maple_3rd_opening" not in [item.post_id for item in before]
    assert "ad-maple_3rd_opening" in [item.post_id for item in during]
    organic_item = next(item for item in during if item.post_id == "ad-maple_3rd_opening")
    assert organic_item.reason.startswith("organic_spillover")


def test_sponsored_ad_does_not_deliver_before_start_tick() -> None:
    campaign = _campaign()
    world, state = initialize_ad_delivery(_world(), (campaign,))

    feed = insert_sponsored_ads(
        world=world,
        viewer_id=1,
        tick=2,
        feed_items=_feed(1, tick=2),
        feed_size=2,
        campaigns=(campaign,),
        state=state,
    )

    assert feed == _feed(1, tick=2)
    assert state.impressions == []


def test_sponsored_ad_does_not_deliver_to_advertiser() -> None:
    campaign = _campaign(targeting="broad")
    world, state = initialize_ad_delivery(_world(), (campaign,))

    feed = insert_sponsored_ads(
        world=world,
        viewer_id=0,
        tick=3,
        feed_items=_feed(0),
        feed_size=2,
        campaigns=(campaign,),
        state=state,
    )

    assert feed == _feed(0)
    assert state.impressions == []


def test_sponsored_ad_respects_total_budget_and_frequency_cap() -> None:
    campaign = _campaign(budget_impressions=2, frequency_cap=1, targeting="broad")
    world, state = initialize_ad_delivery(_world(), (campaign,))

    first = insert_sponsored_ads(
        world=world,
        viewer_id=1,
        tick=3,
        feed_items=_feed(1),
        feed_size=2,
        campaigns=(campaign,),
        state=state,
    )
    second_same_user = insert_sponsored_ads(
        world=world,
        viewer_id=1,
        tick=4,
        feed_items=_feed(1, tick=4),
        feed_size=2,
        campaigns=(campaign,),
        state=state,
    )
    second_user = insert_sponsored_ads(
        world=world,
        viewer_id=2,
        tick=4,
        feed_items=_feed(2, tick=4),
        feed_size=2,
        campaigns=(campaign,),
        state=state,
    )
    exhausted = insert_sponsored_ads(
        world=world,
        viewer_id=0,
        tick=4,
        feed_items=_feed(0, tick=4),
        feed_size=2,
        campaigns=(campaign,),
        state=state,
    )

    assert first[0].source == "sponsored"
    assert second_same_user == _feed(1, tick=4)
    assert second_user[0].source == "sponsored"
    assert exhausted == _feed(0, tick=4)
    assert len(state.impressions) == 2
    assert state.remaining_budget_by_campaign[campaign.campaign_id] == 0


def test_targeting_modes_control_eligibility() -> None:
    broad = _campaign(targeting="broad", budget_impressions=4)
    targeted = _campaign(targeting="interest_targeted", budget_impressions=4)

    broad_world, broad_state = initialize_ad_delivery(_world(), (broad,))
    targeted_world, targeted_state = initialize_ad_delivery(_world(), (targeted,))

    broad_feed = insert_sponsored_ads(
        world=broad_world,
        viewer_id=2,
        tick=3,
        feed_items=_feed(2),
        feed_size=2,
        campaigns=(broad,),
        state=broad_state,
    )
    targeted_feed = insert_sponsored_ads(
        world=targeted_world,
        viewer_id=2,
        tick=3,
        feed_items=_feed(2),
        feed_size=2,
        campaigns=(targeted,),
        state=targeted_state,
    )

    assert broad_feed[0].source == "sponsored"
    assert broad_feed[0].reason == "sponsored_broad"
    assert targeted_feed == _feed(2)


def test_sponsored_insertion_preserves_feed_size_and_campaign_metadata() -> None:
    campaign = _campaign(targeting="broad")
    world, state = initialize_ad_delivery(_world(), (campaign,))

    feed = insert_sponsored_ads(
        world=world,
        viewer_id=1,
        tick=3,
        feed_items=_feed(1),
        feed_size=2,
        campaigns=(campaign,),
        state=state,
    )

    assert len(feed) == 2
    assert [item.rank for item in feed] == [0, 1]
    assert feed[0].source == "sponsored"
    assert feed[0].campaign_id == "maple_3rd_opening"
    assert feed[0].is_sponsored is True
    assert feed[0].advertiser_id == 0
    assert feed[0].ad_seen_count == 1
    assert state.impressions[0].source_reason == "sponsored_broad"
