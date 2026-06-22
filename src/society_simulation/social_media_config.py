from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType


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
