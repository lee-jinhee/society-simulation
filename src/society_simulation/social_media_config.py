from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Literal


def _require_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _require_field(data: dict[str, object], field: str, path: str) -> object:
    if field not in data:
        raise ValueError(f"{path} is required")
    return data[field]


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _require_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{field} must be a finite number")
    return parsed


def _require_non_empty_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_str_sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return tuple(
        _require_non_empty_str(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )


def _freeze_json_value(value: object, field: str) -> object:
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("object keys must be strings")
            frozen[key] = _freeze_json_value(item, f"{field}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_freeze_json_value(item, field) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("free-form config values must contain finite numbers")
        return value
    raise ValueError("free-form config values must be JSON-compatible")


def _freeze_json_mapping(value: object, field: str) -> Mapping[str, object]:
    frozen = _freeze_json_value(_require_mapping(value, field), field)
    if not isinstance(frozen, Mapping):
        raise ValueError(f"{field} must be an object")
    return frozen


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _validate_probability(value: object, field: str) -> float:
    parsed = _require_float(value, field)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return parsed


def _validate_positive_int(value: object, field: str) -> int:
    parsed = _require_int(value, field)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


_FEED_POLICY_TYPES = {
    "chronological_following",
    "engagement_ranked",
    "interest_homophily",
    "bridging",
    "no_feed_control",
}
_SECRET_UPDATE_POLICY_KEY_PATTERNS = (
    "api_key",
    "api-key",
    "apikey",
    "authorization",
    "header",
)
_AD_CONDITIONS = {"no_ad", "organic_post", "sponsored_ad"}
_AD_TARGETING_TYPES = {"broad", "interest_targeted"}


@dataclass(frozen=True)
class AdCampaignConfig:
    campaign_id: str
    advertiser_id: int
    ad_condition: Literal["no_ad", "organic_post", "sponsored_ad"]
    creative_id: str
    creative_text: str
    topic: str
    stance: float
    start_tick: int
    end_tick: int
    budget_impressions: int
    frequency_cap: int
    targeting: Literal["broad", "interest_targeted"]
    sponsored_like_count: int = 0
    targeting_topics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "advertiser_id": self.advertiser_id,
            "ad_condition": self.ad_condition,
            "creative_id": self.creative_id,
            "creative_text": self.creative_text,
            "topic": self.topic,
            "stance": self.stance,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "budget_impressions": self.budget_impressions,
            "frequency_cap": self.frequency_cap,
            "targeting": self.targeting,
            "sponsored_like_count": self.sponsored_like_count,
            "targeting_topics": list(self.targeting_topics),
        }


@dataclass(frozen=True)
class InstagramSocialDynamicsConfig:
    experiment_name: str
    seed: int
    scenario_name: str
    ticks: int
    num_users: int
    historical_posts_per_user: int
    feed_size: int
    activation_probability: float
    topics: tuple[str, ...]
    seed_generator: Mapping[str, object]
    feed_policy: Mapping[str, object]
    update_policy: Mapping[str, object]
    memory_retrieval: Mapping[str, object]
    seed_posts: tuple[Mapping[str, object], ...]
    ad_campaigns: tuple[AdCampaignConfig, ...]
    output_dir: str

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> InstagramSocialDynamicsConfig:
        data = _require_mapping(data, "social media config")
        return cls(
            experiment_name=_require_non_empty_str(
                _require_field(data, "experiment_name", "experiment_name"),
                "experiment_name",
            ),
            seed=_require_int(_require_field(data, "seed", "seed"), "seed"),
            scenario_name=_require_non_empty_str(
                _require_field(data, "scenario_name", "scenario_name"),
                "scenario_name",
            ),
            ticks=_require_int(_require_field(data, "ticks", "ticks"), "ticks"),
            num_users=_require_int(_require_field(data, "num_users", "num_users"), "num_users"),
            historical_posts_per_user=_require_int(
                _require_field(
                    data,
                    "historical_posts_per_user",
                    "historical_posts_per_user",
                ),
                "historical_posts_per_user",
            ),
            feed_size=_require_int(_require_field(data, "feed_size", "feed_size"), "feed_size"),
            activation_probability=_require_float(
                _require_field(data, "activation_probability", "activation_probability"),
                "activation_probability",
            ),
            topics=_require_str_sequence(_require_field(data, "topics", "topics"), "topics"),
            seed_generator=_freeze_json_mapping(
                _require_field(data, "seed_generator", "seed_generator"),
                "seed_generator",
            ),
            feed_policy=_freeze_json_mapping(
                _require_field(data, "feed_policy", "feed_policy"),
                "feed_policy",
            ),
            update_policy=_normalize_update_policy(
                _require_field(data, "update_policy", "update_policy"),
            ),
            memory_retrieval=_normalize_memory_retrieval(data.get("memory_retrieval")),
            seed_posts=_normalize_seed_posts(data.get("seed_posts")),
            ad_campaigns=_normalize_ad_campaigns(data.get("ad_campaigns")),
            output_dir=_require_non_empty_str(
                _require_field(data, "output_dir", "output_dir"),
                "output_dir",
            ),
        )

    def validate(self) -> None:
        if self.experiment_name != "instagram_social_dynamics":
            raise ValueError("unsupported experiment_name")
        _validate_positive_int(self.ticks, "ticks")
        _validate_positive_int(self.num_users, "num_users")
        _validate_positive_int(self.historical_posts_per_user, "historical_posts_per_user")
        _validate_positive_int(self.feed_size, "feed_size")
        _validate_probability(self.activation_probability, "activation_probability")
        _validate_seed_generator(self.seed_generator)
        _validate_feed_policy(self.feed_policy)
        _validate_update_policy(self.update_policy)
        _validate_seed_posts(self)
        _validate_ad_campaigns(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "scenario_name": self.scenario_name,
            "ticks": self.ticks,
            "num_users": self.num_users,
            "historical_posts_per_user": self.historical_posts_per_user,
            "feed_size": self.feed_size,
            "activation_probability": self.activation_probability,
            "topics": list(self.topics),
            "seed_generator": _json_ready(self.seed_generator),
            "feed_policy": _json_ready(self.feed_policy),
            "update_policy": _json_ready(self.update_policy),
            "memory_retrieval": _json_ready(self.memory_retrieval),
            "seed_posts": _json_ready(self.seed_posts),
            "ad_campaigns": [campaign.to_dict() for campaign in self.ad_campaigns],
            "output_dir": self.output_dir,
        }


def _normalize_update_policy(value: object) -> Mapping[str, object]:
    policy = dict(_require_mapping(value, "update_policy"))
    for key in policy:
        if not isinstance(key, str):
            raise ValueError("object keys must be strings")
        if key == "api_key_env":
            continue
        if any(pattern in key.lower() for pattern in _SECRET_UPDATE_POLICY_KEY_PATTERNS):
            raise ValueError(f"secret-bearing update_policy key is not allowed: {key}")
    return _freeze_json_mapping(policy, "update_policy")


def _normalize_memory_retrieval(value: object | None) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({"enabled": False, "limit": 5})
    data = dict(_require_mapping(value, "memory_retrieval"))
    for key in data:
        if key not in {"enabled", "limit"}:
            raise ValueError(f"unsupported memory_retrieval key: {key}")
    enabled = data.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("memory_retrieval.enabled must be a boolean")
    limit = _require_int(data.get("limit", 5), "memory_retrieval.limit")
    if limit <= 0:
        raise ValueError("memory_retrieval.limit must be positive")
    return MappingProxyType({"enabled": enabled, "limit": limit})


def _normalize_seed_posts(value: object | None) -> tuple[Mapping[str, object], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("seed_posts must be a list")
    return tuple(
        _freeze_json_mapping(post, f"seed_posts[{index}]")
        for index, post in enumerate(value)
    )


def _normalize_ad_campaigns(value: object | None) -> tuple[AdCampaignConfig, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("ad_campaigns must be a list")
    return tuple(
        _normalize_ad_campaign(campaign, f"ad_campaigns[{index}]")
        for index, campaign in enumerate(value)
    )


def _normalize_ad_campaign(value: object, prefix: str) -> AdCampaignConfig:
    campaign = _require_mapping(value, prefix)
    topic = _require_non_empty_str(campaign.get("topic"), f"{prefix}.topic")
    targeting_topics_value = campaign.get("targeting_topics")
    targeting_topics = (
        _require_str_sequence(targeting_topics_value, f"{prefix}.targeting_topics")
        if targeting_topics_value is not None
        else (topic,)
    )
    return AdCampaignConfig(
        campaign_id=_require_non_empty_str(campaign.get("campaign_id"), f"{prefix}.campaign_id"),
        advertiser_id=_require_int(campaign.get("advertiser_id"), f"{prefix}.advertiser_id"),
        ad_condition=_require_non_empty_str(
            campaign.get("ad_condition"),
            f"{prefix}.ad_condition",
        ),  # type: ignore[arg-type]
        creative_id=_require_non_empty_str(campaign.get("creative_id"), f"{prefix}.creative_id"),
        creative_text=_require_non_empty_str(
            campaign.get("creative_text"),
            f"{prefix}.creative_text",
        ),
        topic=topic,
        stance=_require_float(campaign.get("stance"), f"{prefix}.stance"),
        start_tick=_require_int(campaign.get("start_tick"), f"{prefix}.start_tick"),
        end_tick=_require_int(campaign.get("end_tick"), f"{prefix}.end_tick"),
        budget_impressions=_require_int(
            campaign.get("budget_impressions"),
            f"{prefix}.budget_impressions",
        ),
        frequency_cap=_require_int(campaign.get("frequency_cap"), f"{prefix}.frequency_cap"),
        targeting=_require_non_empty_str(
            campaign.get("targeting"),
            f"{prefix}.targeting",
        ),  # type: ignore[arg-type]
        sponsored_like_count=_require_int(
            campaign.get("sponsored_like_count", 0),
            f"{prefix}.sponsored_like_count",
        ),
        targeting_topics=targeting_topics,
    )


def _validate_seed_generator(seed_generator: Mapping[str, object]) -> None:
    generator_type = _require_non_empty_str(
        seed_generator.get("type"),
        "seed_generator.type",
    )
    if generator_type != "synthetic_profiles":
        raise ValueError("unsupported seed_generator type")
    _validate_positive_int(seed_generator.get("mean_following"), "seed_generator.mean_following")
    for field in (
        "homophily_weight",
        "popularity_weight",
        "random_tie_probability",
        "mutual_follow_probability",
    ):
        _validate_probability(seed_generator.get(field), f"seed_generator.{field}")


def _validate_feed_policy(feed_policy: Mapping[str, object]) -> None:
    policy_type = _require_non_empty_str(feed_policy.get("type"), "feed_policy.type")
    if policy_type not in _FEED_POLICY_TYPES:
        raise ValueError("unsupported feed_policy type")
    for field in (
        "following_bonus",
        "interest_similarity_weight",
        "stance_similarity_weight",
        "engagement_weight",
        "recency_weight",
        "creator_popularity_weight",
        "controversy_weight",
        "noise_weight",
    ):
        _require_float(feed_policy.get(field), f"feed_policy.{field}")
    _validate_probability(feed_policy.get("explore_fraction"), "feed_policy.explore_fraction")


def _validate_update_policy(update_policy: Mapping[str, object]) -> None:
    policy_type = _require_non_empty_str(update_policy.get("type"), "update_policy.type")
    if policy_type == "mock_social":
        style = update_policy.get("response_style", "balanced")
        if style not in ("balanced", "endorsement_sensitive", "privacy_sensitive"):
            raise ValueError("unsupported mock_social response_style")
        for field in ("input_cost_per_1m_tokens", "output_cost_per_1m_tokens"):
            if field in update_policy:
                parsed = _require_float(update_policy[field], f"update_policy.{field}")
                if parsed < 0:
                    raise ValueError(f"update_policy.{field} must be non-negative")
        return
    if policy_type == "llm":
        provider = update_policy.get("provider", "openai_compatible")
        if provider != "openai_compatible":
            raise ValueError("unsupported llm provider")
        _require_non_empty_str(update_policy.get("model"), "update_policy.model")
        if "max_estimated_cost_usd" in update_policy:
            value = _require_float(
                update_policy["max_estimated_cost_usd"],
                "update_policy.max_estimated_cost_usd",
            )
            if value < 0:
                raise ValueError("update_policy.max_estimated_cost_usd must be non-negative")
        return
    raise ValueError("unsupported update_policy type")


def _validate_seed_posts(config: InstagramSocialDynamicsConfig) -> None:
    seen_post_ids: set[str] = set()
    topics = set(config.topics)
    for index, post in enumerate(config.seed_posts):
        prefix = f"seed_posts[{index}]"
        post_id = _require_non_empty_str(post.get("post_id"), f"{prefix}.post_id")
        if post_id in seen_post_ids:
            raise ValueError(f"{prefix}.post_id must be unique")
        seen_post_ids.add(post_id)
        author_id = _require_int(post.get("author_id"), f"{prefix}.author_id")
        if author_id < 0 or author_id >= config.num_users:
            raise ValueError(f"{prefix}.author_id must reference an existing user")
        topic = _require_non_empty_str(post.get("topic"), f"{prefix}.topic")
        if topic not in topics:
            raise ValueError(f"{prefix}.topic must be listed in topics")
        stance = _require_float(post.get("stance"), f"{prefix}.stance")
        if not -1.0 <= stance <= 1.0:
            raise ValueError(f"{prefix}.stance must be between -1 and 1")
        _require_non_empty_str(post.get("text"), f"{prefix}.text")
        _require_int(post.get("created_tick"), f"{prefix}.created_tick")
        like_count = _require_int(post.get("like_count"), f"{prefix}.like_count")
        if like_count < 0:
            raise ValueError(f"{prefix}.like_count must be non-negative")
        reply_count = _require_int(post.get("reply_count", 0), f"{prefix}.reply_count")
        if reply_count < 0:
            raise ValueError(f"{prefix}.reply_count must be non-negative")


def _validate_ad_campaigns(config: InstagramSocialDynamicsConfig) -> None:
    seen_campaign_ids: set[str] = set()
    topics = set(config.topics)
    for index, campaign in enumerate(config.ad_campaigns):
        prefix = f"ad_campaigns[{index}]"
        if campaign.campaign_id in seen_campaign_ids:
            raise ValueError(f"{prefix}.campaign_id must be unique")
        seen_campaign_ids.add(campaign.campaign_id)
        if campaign.ad_condition not in _AD_CONDITIONS:
            raise ValueError(f"unsupported {prefix}.ad_condition")
        if campaign.targeting not in _AD_TARGETING_TYPES:
            raise ValueError(f"unsupported {prefix}.targeting")
        if campaign.advertiser_id < 0 or campaign.advertiser_id >= config.num_users:
            raise ValueError(f"{prefix}.advertiser_id must reference an existing user")
        if campaign.topic not in topics:
            raise ValueError(f"{prefix}.topic must be listed in topics")
        if not -1.0 <= campaign.stance <= 1.0:
            raise ValueError(f"{prefix}.stance must be between -1 and 1")
        if campaign.start_tick < 1:
            raise ValueError(f"{prefix}.start_tick must be at least 1")
        if campaign.end_tick < campaign.start_tick:
            raise ValueError(f"{prefix}.end_tick must be greater than or equal to start_tick")
        if campaign.end_tick > config.ticks:
            raise ValueError(f"{prefix}.end_tick must not exceed ticks")
        if campaign.budget_impressions < 0:
            raise ValueError(f"{prefix}.budget_impressions must be non-negative")
        if campaign.frequency_cap < 0:
            raise ValueError(f"{prefix}.frequency_cap must be non-negative")
        if campaign.ad_condition == "sponsored_ad" and campaign.budget_impressions <= 0:
            raise ValueError(f"{prefix}.budget_impressions must be positive for sponsored_ad")
        if campaign.ad_condition == "sponsored_ad" and campaign.frequency_cap <= 0:
            raise ValueError(f"{prefix}.frequency_cap must be positive for sponsored_ad")
        if campaign.sponsored_like_count < 0:
            raise ValueError(f"{prefix}.sponsored_like_count must be non-negative")
        for topic in campaign.targeting_topics:
            if topic not in topics:
                raise ValueError(f"{prefix}.targeting_topics must be listed in topics")
