import json

import pytest

from society_simulation.social_media_models import FeedItem, SocialMediaUserProfile, SocialMediaUserState
from society_simulation.social_media_policy import MockSocialMediaPolicy, parse_social_action_content


def _profile() -> SocialMediaUserProfile:
    return SocialMediaUserProfile(
        user_id=1,
        handle="minho",
        display_name="Minho Park",
        bio="bio",
        interests=("transit",),
        home_cluster="urbanists",
        initial_stance=0.3,
        activity_rate=1.0,
        post_rate=0.2,
        privacy_preference=0.2,
        conformity=0.9,
        skepticism=0.2,
        conflict_tolerance=0.6,
        status_weight=0.4,
        posting_style="brief",
    )


def _state() -> SocialMediaUserState:
    return SocialMediaUserState(1, 0, 0.3, 0.5, 0.5, "neutral", 0.0, 0.0, "initial")


def test_parse_social_action_content() -> None:
    content = json.dumps(
        {
            "action_type": "send_dm",
            "post_id": None,
            "target_user_id": 2,
            "text": "Did you see this?",
            "topic": "transit",
            "stance": 0.1,
            "reason": "private concern",
        }
    )

    action = parse_social_action_content(content, tick=2, user_id=1)

    assert action.action_type == "send_dm"
    assert action.target_user_id == 2


def test_parse_rejects_unknown_action() -> None:
    content = json.dumps(
        {
            "action_type": "bookmark",
            "post_id": None,
            "target_user_id": None,
            "text": None,
            "topic": None,
            "stance": None,
            "reason": "not supported",
        }
    )

    with pytest.raises(ValueError, match="unsupported social action"):
        parse_social_action_content(content, tick=2, user_id=1)


def test_mock_policy_likes_high_engagement_feed_item() -> None:
    policy = MockSocialMediaPolicy(response_style="endorsement_sensitive")
    feed = (
        FeedItem(1, 1, "post-low", 2, 0.1, 1, "following", "x"),
        FeedItem(1, 1, "post-high", 3, 2.0, 0, "explore", "x"),
    )

    action = policy.decide(profile=_profile(), state=_state(), feed=feed, tick=1)

    assert action.action_type == "like_post"
    assert action.post_id == "post-high"


def test_mock_policy_can_do_nothing_on_empty_feed() -> None:
    policy = MockSocialMediaPolicy(response_style="balanced")

    action = policy.decide(profile=_profile(), state=_state(), feed=(), tick=1)

    assert action.action_type == "do_nothing"
