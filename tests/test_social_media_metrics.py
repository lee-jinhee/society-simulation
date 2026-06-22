from society_simulation.social_media_ads import AdImpression
from society_simulation.social_media_config import AdCampaignConfig
from society_simulation.social_media_metrics import compute_social_media_metrics
from society_simulation.social_media_models import (
    DirectMessage,
    FeedItem,
    FollowEdge,
    PlatformAction,
    SocialMediaPost,
    SocialMediaUserProfile,
    SocialMediaUserState,
    SocialMediaWorld,
)


def _profile(
    user_id: int,
    stance: float,
    interests: tuple[str, ...] = ("transit",),
) -> SocialMediaUserProfile:
    return SocialMediaUserProfile(
        user_id,
        f"user_{user_id}",
        f"User {user_id}",
        "bio",
        interests,
        "cluster",
        stance,
        1.0,
        0.2,
        0.3,
        0.5,
        0.4,
        0.5,
        0.2,
        "plain",
    )


def test_metrics_count_actions_and_graph_churn() -> None:
    profiles = (_profile(0, 0.1), _profile(1, 0.2))
    states = tuple(
        SocialMediaUserState(p.user_id, 1, p.initial_stance, 0.5, 0.5, "neutral", 0, 0, "x")
        for p in profiles
    )
    world = SocialMediaWorld(
        profiles=profiles,
        states=states,
        posts=(SocialMediaPost("post-1", 1, "transit", 0.2, "text", 0, 1, 0, True),),
        follow_edges=(FollowEdge(0, 1, 0), FollowEdge(1, 0, 1)),
    )
    actions = (
        PlatformAction(1, 0, "like_post", "post-1", None, None, None, None, "liked"),
        PlatformAction(1, 1, "follow_user", None, 0, None, None, None, "followed"),
        PlatformAction(1, 0, "do_nothing", None, None, None, None, None, "ignored"),
    )

    metrics = compute_social_media_metrics(
        initial_edges=(FollowEdge(0, 1, 0),),
        final_world=world,
        feed_items=(FeedItem(1, 0, "post-1", 1, 1.0, 0, "following", "x"),),
        actions=actions,
        dm_messages=(DirectMessage(1, 0, 1, "hi", "transit", 0.1),),
        states_by_tick=(states,),
    )

    assert metrics["action_count"] == 2
    assert metrics["action_counts"]["like_post"] == 1
    assert metrics["action_counts"]["do_nothing"] == 1
    assert metrics["follow_edge_delta"] == 1
    assert metrics["dm_count"] == 1


def test_metrics_return_zero_campaign_metrics_without_campaigns() -> None:
    profiles = (_profile(0, 0.1), _profile(1, 0.2))
    states = tuple(
        SocialMediaUserState(p.user_id, 1, p.initial_stance, 0.5, 0.5, "neutral", 0, 0, "x")
        for p in profiles
    )
    world = SocialMediaWorld(
        profiles=profiles,
        states=states,
        posts=(),
        follow_edges=(),
    )

    metrics = compute_social_media_metrics(
        initial_edges=(),
        final_world=world,
        feed_items=(),
        actions=(),
        dm_messages=(),
        states_by_tick=(states,),
    )

    assert metrics["paid_impression_count"] == 0
    assert metrics["unique_total_ad_reach"] == 0
    assert metrics["ad_like_count"] == 0
    assert metrics["paid_to_organic_spillover_rate"] == 0.0


def test_metrics_compute_campaign_reach_frequency_and_engagement() -> None:
    campaign = AdCampaignConfig(
        campaign_id="maple_3rd_opening",
        advertiser_id=0,
        ad_condition="sponsored_ad",
        creative_id="discount_offer",
        creative_text="First 100 visitors get a free pastry with any drink.",
        topic="coffee",
        stance=0.2,
        start_tick=2,
        end_tick=4,
        budget_impressions=4,
        frequency_cap=2,
        targeting="broad",
        sponsored_like_count=25,
        targeting_topics=("coffee",),
    )
    profiles = (
        _profile(0, 0.1, ("coffee",)),
        _profile(1, 0.2, ("coffee", "commute")),
        _profile(2, -0.1, ("sports",)),
    )
    states = tuple(
        SocialMediaUserState(p.user_id, 4, p.initial_stance, 0.5, 0.5, "neutral", 0, 0, "x")
        for p in profiles
    )
    world = SocialMediaWorld(
        profiles=profiles,
        states=states,
        posts=(
            SocialMediaPost(
                "ad-maple_3rd_opening",
                0,
                "coffee",
                0.2,
                "Free pastry with any drink.",
                2,
                1,
                0,
                False,
                campaign_id="maple_3rd_opening",
                is_ad=True,
            ),
        ),
        follow_edges=(),
    )
    ad_impressions = (
        AdImpression(2, 1, "maple_3rd_opening", 0, "ad-maple_3rd_opening", "broad", "x", 1, 2, "coffee"),
        AdImpression(3, 1, "maple_3rd_opening", 0, "ad-maple_3rd_opening", "broad", "x", 2, 2, "coffee"),
        AdImpression(3, 2, "maple_3rd_opening", 0, "ad-maple_3rd_opening", "broad", "x", 1, 2, "coffee"),
    )
    feed_items = (
        FeedItem(1, 1, "burn-in-post", 2, 1.0, 0, "explore", "x"),
        FeedItem(
            2,
            1,
            "ad-maple_3rd_opening",
            0,
            1.0,
            0,
            "sponsored",
            "sponsored_broad",
            campaign_id="maple_3rd_opening",
            is_sponsored=True,
            advertiser_id=0,
            topic="coffee",
        ),
        FeedItem(
            3,
            1,
            "ad-maple_3rd_opening",
            0,
            1.0,
            1,
            "explore",
            "organic_spillover",
            campaign_id="maple_3rd_opening",
            advertiser_id=0,
            topic="coffee",
        ),
        FeedItem(
            3,
            2,
            "ad-maple_3rd_opening",
            0,
            1.0,
            1,
            "explore",
            "organic_spillover",
            campaign_id="maple_3rd_opening",
            advertiser_id=0,
            topic="coffee",
        ),
    )
    actions = (
        PlatformAction(1, 1, "follow_user", None, 2, None, None, None, "burn-in follow"),
        PlatformAction(2, 1, "like_post", "ad-maple_3rd_opening", None, None, None, None, "liked"),
        PlatformAction(3, 2, "follow_user", None, 0, None, None, None, "followed advertiser"),
        PlatformAction(3, 1, "send_dm", None, 0, "Is this place open?", "coffee", 0.2, "asked"),
        PlatformAction(4, 1, "create_post", None, None, "Trying Maple soon.", "coffee", 0.2, "posted"),
    )

    metrics = compute_social_media_metrics(
        initial_edges=(),
        final_world=world,
        feed_items=feed_items,
        actions=actions,
        dm_messages=(DirectMessage(3, 1, 0, "Is this place open?", "coffee", 0.2),),
        states_by_tick=(states,),
        ad_impressions=ad_impressions,
        ad_campaigns=(campaign,),
        ad_remaining_budget_by_campaign={"maple_3rd_opening": 1},
    )

    assert metrics["paid_impression_count"] == 3
    assert metrics["unique_paid_reach"] == 2
    assert metrics["organic_ad_impression_count"] == 2
    assert metrics["unique_organic_ad_reach"] == 2
    assert metrics["unique_total_ad_reach"] == 2
    assert metrics["relevant_paid_reach"] == 1
    assert metrics["relevant_total_reach"] == 1
    assert metrics["mean_ad_frequency"] == 1.5
    assert metrics["max_ad_frequency"] == 2
    assert metrics["frequency_cap_hit_count"] == 1
    assert metrics["ad_like_count"] == 1
    assert metrics["advertiser_follow_count"] == 1
    assert metrics["ad_dm_count"] == 1
    assert metrics["ad_generated_post_count"] == 1
    assert metrics["paid_to_organic_spillover_rate"] == 0.666667
    assert metrics["ad_delivery_exhausted_budget"] is False
    assert metrics["ad_delivery_remaining_budget"] == 1
    assert metrics["burn_in_action_mean"] == 1.0
    assert metrics["burn_in_follow_churn"] == 1
    assert metrics["burn_in_exposure_diversity"] == 1.0
