from types import MappingProxyType

from society_simulation.social_media_feed import build_feed
from society_simulation.social_media_models import (
    FollowEdge,
    SocialMediaPost,
    SocialMediaUserProfile,
    SocialMediaUserState,
    SocialMediaWorld,
)


def _profile(user_id: int, stance: float, interests: tuple[str, ...]) -> SocialMediaUserProfile:
    return SocialMediaUserProfile(
        user_id=user_id,
        handle=f"user_{user_id}",
        display_name=f"User {user_id}",
        bio="bio",
        interests=interests,
        home_cluster="cluster",
        initial_stance=stance,
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
        _profile(0, 0.2, ("transit",)),
        _profile(1, 0.3, ("transit",)),
        _profile(2, -0.8, ("housing",)),
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
        SocialMediaPost("post-1", 1, "transit", 0.3, "bus lane update", -1, 2, 0, True),
        SocialMediaPost("post-2", 2, "housing", -0.8, "rent thread", -1, 40, 0, True),
    )
    return SocialMediaWorld(
        profiles=profiles,
        states=states,
        posts=posts,
        follow_edges=(FollowEdge(0, 1, 0),),
    )


def _policy(policy_type: str) -> MappingProxyType[str, object]:
    return MappingProxyType(
        {
            "type": policy_type,
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


def test_chronological_following_only_returns_followed_posts() -> None:
    feed = build_feed(
        world=_world(),
        viewer_id=0,
        tick=1,
        feed_size=5,
        feed_policy=_policy("chronological_following"),
        seed=99,
    )

    assert [item.post_id for item in feed] == ["post-1"]
    assert feed[0].source == "following"


def test_engagement_ranked_can_surface_explore_post() -> None:
    feed = build_feed(
        world=_world(),
        viewer_id=0,
        tick=1,
        feed_size=2,
        feed_policy=_policy("engagement_ranked"),
        seed=99,
    )

    assert [item.post_id for item in feed][0] == "post-2"
    assert feed[0].source == "explore"
    assert feed[0].visible_like_count == 40
    assert feed[0].topic == "housing"
    assert feed[0].text == "rent thread"
    assert feed[0].author_handle == "user_2"


def test_no_feed_control_returns_empty_feed() -> None:
    feed = build_feed(
        world=_world(),
        viewer_id=0,
        tick=1,
        feed_size=2,
        feed_policy=_policy("no_feed_control"),
        seed=99,
    )

    assert feed == ()


def test_feed_excludes_posts_created_after_current_tick() -> None:
    world = _world()
    future_post = SocialMediaPost(
        "future-ad",
        1,
        "transit",
        0.2,
        "This campaign starts later.",
        5,
        99,
        0,
        False,
        campaign_id="maple_3rd_opening",
        is_ad=True,
    )
    world = SocialMediaWorld(
        profiles=world.profiles,
        states=world.states,
        posts=(*world.posts, future_post),
        follow_edges=world.follow_edges,
    )

    before_start = build_feed(
        world=world,
        viewer_id=0,
        tick=4,
        feed_size=5,
        feed_policy=_policy("engagement_ranked"),
        seed=99,
    )
    after_start = build_feed(
        world=world,
        viewer_id=0,
        tick=5,
        feed_size=5,
        feed_policy=_policy("engagement_ranked"),
        seed=99,
    )

    assert "future-ad" not in [item.post_id for item in before_start]
    assert "future-ad" in [item.post_id for item in after_start]
