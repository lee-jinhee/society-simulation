from society_simulation.social_media_models import (
    FeedItem,
    FollowEdge,
    PlatformAction,
    SocialMediaPost,
    SocialMediaUserProfile,
    SocialMediaUserState,
)


def test_profile_to_dict_is_json_ready() -> None:
    profile = SocialMediaUserProfile(
        user_id=1,
        handle="minho",
        display_name="Minho Park",
        bio="Transit photographer and cafe regular.",
        interests=("transit", "photography"),
        home_cluster="urbanists",
        initial_stance=0.4,
        activity_rate=0.75,
        post_rate=0.2,
        privacy_preference=0.35,
        conformity=0.5,
        skepticism=0.3,
        conflict_tolerance=0.6,
        status_weight=0.2,
        posting_style="dry and specific",
    )

    assert profile.to_dict() == {
        "user_id": 1,
        "handle": "minho",
        "display_name": "Minho Park",
        "bio": "Transit photographer and cafe regular.",
        "interests": ["transit", "photography"],
        "home_cluster": "urbanists",
        "initial_stance": 0.4,
        "activity_rate": 0.75,
        "post_rate": 0.2,
        "privacy_preference": 0.35,
        "conformity": 0.5,
        "skepticism": 0.3,
        "conflict_tolerance": 0.6,
        "status_weight": 0.2,
        "posting_style": "dry and specific",
    }


def test_platform_action_requires_supported_type() -> None:
    action = PlatformAction(
        tick=3,
        user_id=1,
        action_type="like_post",
        post_id="post-1",
        target_user_id=None,
        text=None,
        topic=None,
        stance=None,
        reason="The post already has visible traction.",
    )

    assert action.to_dict()["action_type"] == "like_post"


def test_post_like_increments_without_mutating_original() -> None:
    post = SocialMediaPost(
        post_id="post-1",
        author_id=2,
        topic="transit",
        stance=0.7,
        text="The new bus lane is overdue.",
        created_tick=0,
        like_count=4,
        reply_count=0,
        seed_post=True,
    )

    updated = post.with_like()

    assert post.like_count == 4
    assert updated.like_count == 5


def test_post_preserves_campaign_metadata_when_liked() -> None:
    post = SocialMediaPost(
        post_id="ad-maple",
        author_id=2,
        topic="coffee",
        stance=0.2,
        text="Free pastry with any drink.",
        created_tick=2,
        like_count=4,
        reply_count=0,
        seed_post=False,
        campaign_id="maple_3rd_opening",
        is_ad=True,
    )

    updated = post.with_like()

    assert updated.like_count == 5
    assert updated.campaign_id == "maple_3rd_opening"
    assert updated.is_ad is True


def test_feed_item_records_recommendation_reason() -> None:
    item = FeedItem(
        tick=2,
        viewer_id=1,
        post_id="post-9",
        author_id=4,
        score=1.25,
        rank=0,
        source="explore",
        reason="interest_similarity=0.50 engagement=0.69",
    )

    assert item.to_dict()["source"] == "explore"


def test_feed_item_records_sponsored_campaign_metadata() -> None:
    item = FeedItem(
        tick=3,
        viewer_id=1,
        post_id="ad-maple",
        author_id=0,
        score=10.0,
        rank=0,
        source="sponsored",
        reason="sponsored_targeted",
        visible_like_count=25,
        topic="coffee",
        text="Free pastry with any drink.",
        author_handle="maple_3rd_coffee",
        campaign_id="maple_3rd_opening",
        is_sponsored=True,
        advertiser_id=0,
        ad_seen_count=1,
    )

    assert item.to_dict()["campaign_id"] == "maple_3rd_opening"
    assert item.to_dict()["is_sponsored"] is True
    assert item.to_dict()["ad_seen_count"] == 1


def test_follow_edge_round_trip() -> None:
    edge = FollowEdge(follower_id=1, followed_id=2, created_tick=0)

    assert edge.to_dict() == {
        "follower_id": 1,
        "followed_id": 2,
        "created_tick": 0,
    }


def test_user_state_tracks_dynamic_fields() -> None:
    state = SocialMediaUserState(
        user_id=1,
        tick=4,
        stance=0.15,
        confidence=0.6,
        salience=0.7,
        mood="curious",
        perceived_majority=0.25,
        social_fatigue=0.1,
        last_action_type="send_dm",
    )

    assert state.to_dict()["last_action_type"] == "send_dm"
