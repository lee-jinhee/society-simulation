import json
from dataclasses import replace

import pytest

from society_simulation.social_media_models import FeedItem, SocialMediaUserProfile, SocialMediaUserState
from society_simulation.social_media_policy import (
    MockSocialMediaPolicy,
    OpenAICompatibleSocialMediaPolicy,
    build_social_media_prompt,
    parse_social_action_content,
)


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


def test_parse_accepts_common_llm_null_strings_for_optional_fields() -> None:
    content = json.dumps(
        {
            "action_type": "like_post",
            "post_id": "post-1",
            "target_user_id": "none",
            "text": "",
            "topic": "null",
            "stance": "N/A",
            "reason": "visible signal",
        }
    )

    action = parse_social_action_content(content, tick=2, user_id=1)

    assert action.action_type == "like_post"
    assert action.post_id == "post-1"
    assert action.target_user_id is None
    assert action.text is None
    assert action.topic is None
    assert action.stance is None


def test_parse_accepts_numeric_strings_for_optional_numeric_fields() -> None:
    content = json.dumps(
        {
            "action_type": "send_dm",
            "post_id": "post-1",
            "target_user_id": "2",
            "text": "Did you see this?",
            "topic": "transit",
            "stance": "0.25",
            "reason": "private concern",
        }
    )

    action = parse_social_action_content(content, tick=2, user_id=1)

    assert action.target_user_id == 2
    assert action.stance == 0.25


def test_mock_policy_likes_high_engagement_feed_item() -> None:
    policy = MockSocialMediaPolicy(response_style="balanced")
    feed = (
        FeedItem(1, 1, "post-low", 2, 0.1, 1, "following", "x"),
        FeedItem(1, 1, "post-high", 3, 2.0, 0, "explore", "x"),
    )

    action = policy.decide(profile=_profile(), state=_state(), feed=feed, tick=1)

    assert action.action_type == "like_post"
    assert action.post_id == "post-high"


def test_mock_policy_likes_relevant_discount_ad() -> None:
    policy = MockSocialMediaPolicy(response_style="balanced")
    feed = (
        FeedItem(
            1,
            1,
            "ad-maple",
            0,
            3.0,
            0,
            "sponsored",
            "sponsored_broad",
            visible_like_count=20,
            topic="transit",
            text="First 100 visitors get a free pastry with any drink.",
            author_handle="maple_3rd_coffee",
            campaign_id="maple_3rd_opening",
            is_sponsored=True,
            advertiser_id=0,
            ad_seen_count=1,
        ),
    )

    action = policy.decide(profile=_profile(), state=_state(), feed=feed, tick=1)

    assert action.action_type == "like_post"
    assert action.post_id == "ad-maple"
    assert "relevant sponsored offer" in action.reason


def test_mock_policy_creates_post_for_relevant_social_proof_ad() -> None:
    policy = MockSocialMediaPolicy(response_style="balanced")
    feed = (
        FeedItem(
            1,
            1,
            "ad-maple",
            0,
            3.0,
            0,
            "sponsored",
            "sponsored_targeted",
            visible_like_count=90,
            topic="transit",
            text="Neighbors are already sharing Maple & 3rd Coffee's opening weekend menu.",
            author_handle="maple_3rd_coffee",
            campaign_id="maple_3rd_opening",
            is_sponsored=True,
            advertiser_id=0,
            ad_seen_count=1,
        ),
    )

    action = policy.decide(profile=_profile(), state=_state(), feed=feed, tick=1)

    assert action.action_type == "create_post"
    assert action.topic == "transit"
    assert "social proof" in action.reason


def test_mock_policy_ignores_repeated_irrelevant_sponsored_ad() -> None:
    policy = MockSocialMediaPolicy(response_style="balanced")
    skeptical_profile = replace(_profile(), interests=("transit",), skepticism=0.9)
    feed = (
        FeedItem(
            3,
            1,
            "ad-maple",
            0,
            3.0,
            0,
            "sponsored",
            "sponsored_broad",
            visible_like_count=5,
            topic="coffee",
            text="First 100 visitors get a free pastry with any drink.",
            author_handle="maple_3rd_coffee",
            campaign_id="maple_3rd_opening",
            is_sponsored=True,
            advertiser_id=0,
            ad_seen_count=3,
        ),
    )

    action = policy.decide(profile=skeptical_profile, state=_state(), feed=feed, tick=3)

    assert action.action_type == "do_nothing"
    assert "repetition" in action.reason


def test_endorsement_sensitive_mock_policy_does_not_like_score_without_visible_endorsement() -> None:
    policy = MockSocialMediaPolicy(response_style="endorsement_sensitive")
    feed = (
        FeedItem(
            1,
            1,
            "post-score-high",
            2,
            2.0,
            0,
            "explore",
            "x",
            visible_like_count=0,
            topic="transit",
            text="Algorithmically relevant but not visibly endorsed.",
            author_handle="carlos",
        ),
    )

    action = policy.decide(profile=_profile(), state=_state(), feed=feed, tick=1)

    assert action.action_type == "do_nothing"


def test_endorsement_sensitive_mock_policy_ignores_low_visible_endorsement() -> None:
    policy = MockSocialMediaPolicy(response_style="endorsement_sensitive")
    feed = (
        FeedItem(
            1,
            1,
            "post-low",
            2,
            0.2,
            0,
            "explore",
            "x",
            visible_like_count=0,
            topic="transit",
            text="A low-signal post.",
            author_handle="carlos",
        ),
    )

    action = policy.decide(profile=_profile(), state=_state(), feed=feed, tick=1)

    assert action.action_type == "do_nothing"


def test_endorsement_sensitive_mock_policy_likes_high_visible_endorsement() -> None:
    policy = MockSocialMediaPolicy(response_style="endorsement_sensitive")
    feed = (
        FeedItem(
            1,
            1,
            "post-high-likes",
            2,
            0.2,
            0,
            "explore",
            "x",
            visible_like_count=80,
            topic="transit",
            text="The same post with a lot of visible endorsement.",
            author_handle="carlos",
        ),
    )

    action = policy.decide(profile=_profile(), state=_state(), feed=feed, tick=1)

    assert action.action_type == "like_post"
    assert action.post_id == "post-high-likes"
    assert "visible endorsement" in action.reason


def test_mock_policy_can_do_nothing_on_empty_feed() -> None:
    policy = MockSocialMediaPolicy(response_style="balanced")

    action = policy.decide(profile=_profile(), state=_state(), feed=(), tick=1)

    assert action.action_type == "do_nothing"


def test_social_media_prompt_does_not_reveal_experiment() -> None:
    prompt = build_social_media_prompt(
        profile=_profile(),
        state=_state(),
        feed=(
            FeedItem(
                1,
                1,
                "post-1",
                2,
                1.0,
                0,
                "following",
                "x",
                visible_like_count=80,
                topic="transit",
                text="The new bus lane finally makes sense.",
                author_handle="carlos",
            ),
        ),
        recent_memories=(),
    )

    assert "simulating a social network experiment" not in prompt.lower()
    assert "instagram-like app" in prompt.lower()
    assert "80 likes" in prompt
    assert "@carlos" in prompt
    assert "author_id=2" in prompt
    assert "The new bus lane finally makes sense." in prompt


def test_social_media_prompt_requires_null_for_unused_optional_fields() -> None:
    prompt = build_social_media_prompt(
        profile=_profile(),
        state=_state(),
        feed=(),
        recent_memories=(),
    )

    assert "Use null, not strings like \"none\" or \"N/A\"" in prompt


def test_social_media_prompt_exposes_sponsored_ad_card_facts() -> None:
    prompt = build_social_media_prompt(
        profile=_profile(),
        state=_state(),
        feed=(
            FeedItem(
                1,
                1,
                "ad-maple",
                0,
                3.0,
                0,
                "sponsored",
                "sponsored_targeted",
                visible_like_count=25,
                topic="transit",
                text="First 100 visitors get a free pastry with any drink.",
                author_handle="maple_3rd_coffee",
                campaign_id="maple_3rd_opening",
                is_sponsored=True,
                advertiser_id=0,
                ad_seen_count=2,
            ),
        ),
        recent_memories=(),
    )

    assert "label=sponsored" in prompt
    assert "campaign=maple_3rd_opening" in prompt
    assert "seen_before=2" in prompt
    assert "author_id=0" in prompt


def test_parse_create_post_action() -> None:
    content = json.dumps(
        {
            "action_type": "create_post",
            "post_id": None,
            "target_user_id": None,
            "text": "This feels like it will affect my commute.",
            "topic": "transit",
            "stance": -0.2,
            "reason": "public concern",
        }
    )

    action = parse_social_action_content(content, tick=3, user_id=1)

    assert action.action_type == "create_post"
    assert action.topic == "transit"


def test_openai_compatible_social_policy_tracks_usage() -> None:
    captured: dict[str, object] = {}

    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        captured["timeout_seconds"] = timeout_seconds
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "action_type": "like_post",
                                "post_id": "post-1",
                                "target_user_id": None,
                                "text": None,
                                "topic": None,
                                "stance": None,
                                "reason": "visible signal",
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 13, "completion_tokens": 5},
        }

    policy = OpenAICompatibleSocialMediaPolicy(
        model="cheap-chat",
        api_key="test-key",
        base_url="https://example.test/v1",
        input_cost_per_1m_tokens=1.0,
        output_cost_per_1m_tokens=2.0,
        transport=transport,
    )

    action = policy.decide(
        profile=_profile(),
        state=_state(),
        feed=(FeedItem(1, 1, "post-1", 2, 1.0, 0, "following", "x"),),
        tick=1,
    )

    assert action.action_type == "like_post"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "cheap-chat"
    usage = policy.usage_summary()
    assert usage["calls"] == 1
    assert usage["prompt_tokens"] == 13
    assert usage["completion_tokens"] == 5
    assert usage["total_cost_usd"] == pytest.approx(23 / 1_000_000)


def test_openai_compatible_social_policy_cost_cap_stops_before_call() -> None:
    called = False

    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    policy = OpenAICompatibleSocialMediaPolicy(
        model="cheap-chat",
        api_key="test-key",
        input_cost_per_1m_tokens=1_000_000.0,
        max_estimated_cost_usd=0.01,
        transport=transport,
    )

    with pytest.raises(ValueError, match="social media llm cost cap exceeded"):
        policy.decide(profile=_profile(), state=_state(), feed=(), tick=1)

    assert called is False
