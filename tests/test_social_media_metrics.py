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


def _profile(user_id: int, stance: float) -> SocialMediaUserProfile:
    return SocialMediaUserProfile(
        user_id,
        f"user_{user_id}",
        f"User {user_id}",
        "bio",
        ("transit",),
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
