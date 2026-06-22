import json

import pytest

from society_simulation.config import load_config
from society_simulation.social_media_config import AdCampaignConfig, InstagramSocialDynamicsConfig


def valid_social_media_config() -> dict[str, object]:
    return {
        "experiment_name": "instagram_social_dynamics",
        "seed": 7,
        "scenario_name": "lite_mock",
        "ticks": 4,
        "num_users": 8,
        "historical_posts_per_user": 2,
        "feed_size": 3,
        "activation_probability": 1.0,
        "topics": ["transit", "housing"],
        "seed_generator": {
            "type": "synthetic_profiles",
            "mean_following": 3,
            "homophily_weight": 0.55,
            "popularity_weight": 0.25,
            "random_tie_probability": 0.10,
            "mutual_follow_probability": 0.40,
        },
        "feed_policy": {
            "type": "engagement_ranked",
            "following_bonus": 1.0,
            "interest_similarity_weight": 0.7,
            "stance_similarity_weight": 0.3,
            "engagement_weight": 0.8,
            "recency_weight": 0.5,
            "creator_popularity_weight": 0.2,
            "controversy_weight": 0.0,
            "noise_weight": 0.01,
            "explore_fraction": 0.33,
        },
        "update_policy": {
            "type": "mock_social",
            "response_style": "balanced",
            "input_cost_per_1m_tokens": 0.0,
            "output_cost_per_1m_tokens": 0.0,
        },
        "memory_retrieval": {"enabled": True, "limit": 5},
        "output_dir": "runs/instagram_social_lite_mock",
    }


def test_load_instagram_social_config(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(valid_social_media_config()), encoding="utf-8")

    config = load_config(path)

    assert isinstance(config, InstagramSocialDynamicsConfig)
    assert config.experiment_name == "instagram_social_dynamics"
    assert config.feed_policy["type"] == "engagement_ranked"
    assert config.seed_posts == ()
    assert config.ad_campaigns == ()


def test_reject_invalid_feed_policy_type() -> None:
    data = valid_social_media_config()
    data["feed_policy"] = dict(data["feed_policy"], type="magic")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unsupported feed_policy type"):
        InstagramSocialDynamicsConfig.from_dict(data).validate()


def test_reject_secret_bearing_update_policy_keys() -> None:
    data = valid_social_media_config()
    data["update_policy"] = {"type": "llm", "model": "x", "api_key": "secret"}

    with pytest.raises(ValueError, match="secret-bearing update_policy key"):
        InstagramSocialDynamicsConfig.from_dict(data)


def test_accept_seed_posts_for_visible_endorsement_scenarios() -> None:
    data = valid_social_media_config()
    data["seed_posts"] = [
        {
            "post_id": "endorsement-seed",
            "author_id": 0,
            "topic": "transit",
            "stance": 0.35,
            "text": "The new bus lane finally makes sense.",
            "created_tick": 0,
            "like_count": 80,
            "reply_count": 3,
        }
    ]

    config = InstagramSocialDynamicsConfig.from_dict(data)
    config.validate()

    assert config.seed_posts[0]["post_id"] == "endorsement-seed"
    assert config.seed_posts[0]["like_count"] == 80
    assert config.to_dict()["seed_posts"] == data["seed_posts"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda post: post.update({"author_id": 99}),
            "seed_posts\\[0\\].author_id must reference an existing user",
        ),
        (
            lambda post: post.update({"like_count": -1}),
            "seed_posts\\[0\\].like_count must be non-negative",
        ),
        (
            lambda post: post.update({"topic": "weather"}),
            "seed_posts\\[0\\].topic must be listed in topics",
        ),
    ],
)
def test_reject_invalid_seed_posts(mutation, message: str) -> None:
    data = valid_social_media_config()
    post = {
        "post_id": "endorsement-seed",
        "author_id": 0,
        "topic": "transit",
        "stance": 0.35,
        "text": "The new bus lane finally makes sense.",
        "created_tick": 0,
        "like_count": 80,
        "reply_count": 3,
    }
    mutation(post)
    data["seed_posts"] = [post]

    config = InstagramSocialDynamicsConfig.from_dict(data)
    with pytest.raises(ValueError, match=message):
        config.validate()


def _campaign(**overrides: object) -> dict[str, object]:
    campaign = {
        "campaign_id": "maple_3rd_opening",
        "advertiser_id": 0,
        "ad_condition": "sponsored_ad",
        "creative_id": "discount_offer",
        "creative_text": "First 100 visitors get a free pastry with any drink.",
        "topic": "transit",
        "stance": 0.2,
        "start_tick": 2,
        "end_tick": 4,
        "budget_impressions": 8,
        "frequency_cap": 2,
        "targeting": "interest_targeted",
        "sponsored_like_count": 25,
        "targeting_topics": ["transit"],
    }
    campaign.update(overrides)
    return campaign


def test_accept_ad_campaign_config_and_round_trip() -> None:
    data = valid_social_media_config()
    data["ad_campaigns"] = [_campaign()]

    config = InstagramSocialDynamicsConfig.from_dict(data)
    config.validate()

    assert config.ad_campaigns == (
        AdCampaignConfig(
            campaign_id="maple_3rd_opening",
            advertiser_id=0,
            ad_condition="sponsored_ad",
            creative_id="discount_offer",
            creative_text="First 100 visitors get a free pastry with any drink.",
            topic="transit",
            stance=0.2,
            start_tick=2,
            end_tick=4,
            budget_impressions=8,
            frequency_cap=2,
            targeting="interest_targeted",
            sponsored_like_count=25,
            targeting_topics=("transit",),
        ),
    )
    assert config.to_dict()["ad_campaigns"] == data["ad_campaigns"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda campaign: campaign.update({"ad_condition": "magic"}),
            "unsupported ad_campaigns\\[0\\].ad_condition",
        ),
        (
            lambda campaign: campaign.update({"targeting": "magic"}),
            "unsupported ad_campaigns\\[0\\].targeting",
        ),
        (
            lambda campaign: campaign.update({"advertiser_id": 99}),
            "ad_campaigns\\[0\\].advertiser_id must reference an existing user",
        ),
        (
            lambda campaign: campaign.update({"topic": "coffee"}),
            "ad_campaigns\\[0\\].topic must be listed in topics",
        ),
        (
            lambda campaign: campaign.update({"budget_impressions": 0}),
            "ad_campaigns\\[0\\].budget_impressions must be positive for sponsored_ad",
        ),
        (
            lambda campaign: campaign.update({"frequency_cap": 0}),
            "ad_campaigns\\[0\\].frequency_cap must be positive for sponsored_ad",
        ),
    ],
)
def test_reject_invalid_ad_campaigns(mutation, message: str) -> None:
    data = valid_social_media_config()
    campaign = _campaign()
    mutation(campaign)
    data["ad_campaigns"] = [campaign]

    config = InstagramSocialDynamicsConfig.from_dict(data)
    with pytest.raises(ValueError, match=message):
        config.validate()
