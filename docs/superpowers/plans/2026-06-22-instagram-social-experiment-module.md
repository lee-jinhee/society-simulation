# Instagram-Like Social Experiment Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, replayable `instagram_social_dynamics` experiment family for platform-mediated social behavior: feeds, likes, follows, unfollows, public posts, DMs, graph rewiring, and recommendation-driven exposure.

**Architecture:** Add a fourth experiment family alongside sequential cascade, network herding, and event-driven opinion dynamics. Keep platform state, recommender logic, policy decisions, metrics, and replay in focused `social_media_*` modules so this does not contaminate the existing event-driven conversation runner.

**Tech Stack:** Python 3.11, dataclasses, JSON configs, pytest, JSONL replay, existing OpenAI-compatible policy utilities.

---

## File Structure

Create these files:

- `src/society_simulation/social_media_models.py`
  - User profiles, dynamic state, posts, feed items, actions, DMs, world snapshot.
- `src/society_simulation/social_media_config.py`
  - Config dataclasses and validation for the new experiment family.
- `src/society_simulation/social_media_seed.py`
  - Deterministic generation of profiles, follow graph, historical posts, and initial world.
- `src/society_simulation/social_media_feed.py`
  - Feed candidate selection and deterministic recommendation scoring.
- `src/society_simulation/social_media_policy.py`
  - Policy interface, mock policy, LLM response parser, LLM policy wrapper.
- `src/society_simulation/social_media_runner.py`
  - Activation loop, action validation, state transitions, partial replay on failure.
- `src/society_simulation/social_media_metrics.py`
  - Engagement, exposure, graph, cascade, stance, and policy-error metrics.
- `src/society_simulation/social_media_replay.py`
  - JSON and JSONL artifacts for replay.

Modify these files:

- `src/society_simulation/config.py`
  - Include `InstagramSocialDynamicsConfig` in `Config` union and `load_config`.
- `src/society_simulation/runner.py`
  - Dispatch to `run_instagram_social_dynamics`.
- `src/society_simulation/cli.py`
  - Print social-media metrics when present.
- `src/society_simulation/sweep_config.py`
  - Ensure sweep configs can override social-media experiment fields.

Create tests:

- `tests/test_social_media_models.py`
- `tests/test_social_media_config.py`
- `tests/test_social_media_seed.py`
- `tests/test_social_media_feed.py`
- `tests/test_social_media_policy.py`
- `tests/test_social_media_runner.py`
- `tests/test_social_media_metrics.py`
- `tests/test_social_media_replay.py`

Create config:

- `experiments/instagram_social_lite_mock.json`

---

## Task 1: Social Media Models

**Files:**
- Create: `src/society_simulation/social_media_models.py`
- Test: `tests/test_social_media_models.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/test_social_media_models.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify import failure**

Run:

```bash
./.venv/bin/python -m pytest tests/test_social_media_models.py -q
```

Expected: FAIL because `society_simulation.social_media_models` does not exist.

- [ ] **Step 3: Add model dataclasses**

Create `src/society_simulation/social_media_models.py` with this structure:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

SocialActionType = Literal[
    "like_post",
    "follow_user",
    "unfollow_user",
    "send_dm",
    "create_post",
    "do_nothing",
]
FeedSource = Literal["following", "explore", "intervention"]


def _tuple_to_list(data: dict[str, object]) -> dict[str, object]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in data.items()
    }


@dataclass(frozen=True)
class SocialMediaUserProfile:
    user_id: int
    handle: str
    display_name: str
    bio: str
    interests: tuple[str, ...]
    home_cluster: str
    initial_stance: float
    activity_rate: float
    post_rate: float
    privacy_preference: float
    conformity: float
    skepticism: float
    conflict_tolerance: float
    status_weight: float
    posting_style: str

    def to_dict(self) -> dict[str, object]:
        return _tuple_to_list(asdict(self))


@dataclass(frozen=True)
class SocialMediaUserState:
    user_id: int
    tick: int
    stance: float
    confidence: float
    salience: float
    mood: str
    perceived_majority: float
    social_fatigue: float
    last_action_type: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SocialMediaPost:
    post_id: str
    author_id: int
    topic: str
    stance: float
    text: str
    created_tick: int
    like_count: int
    reply_count: int
    seed_post: bool

    def with_like(self) -> SocialMediaPost:
        return replace(self, like_count=self.like_count + 1)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FollowEdge:
    follower_id: int
    followed_id: int
    created_tick: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FeedItem:
    tick: int
    viewer_id: int
    post_id: str
    author_id: int
    score: float
    rank: int
    source: FeedSource
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlatformAction:
    tick: int
    user_id: int
    action_type: SocialActionType
    post_id: str | None
    target_user_id: int | None
    text: str | None
    topic: str | None
    stance: float | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DirectMessage:
    tick: int
    sender_id: int
    recipient_id: int
    text: str
    topic: str | None
    stance: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SocialMediaWorld:
    profiles: tuple[SocialMediaUserProfile, ...]
    states: tuple[SocialMediaUserState, ...]
    posts: tuple[SocialMediaPost, ...]
    follow_edges: tuple[FollowEdge, ...]

    def profile_by_id(self) -> dict[int, SocialMediaUserProfile]:
        return {profile.user_id: profile for profile in self.profiles}

    def state_by_id(self) -> dict[int, SocialMediaUserState]:
        return {state.user_id: state for state in self.states}

    def post_by_id(self) -> dict[str, SocialMediaPost]:
        return {post.post_id: post for post in self.posts}
```

- [ ] **Step 4: Run model tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_social_media_models.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/social_media_models.py tests/test_social_media_models.py
git commit -m "feat: add social media experiment models"
```

---

## Task 2: Config Parsing and Validation

**Files:**
- Create: `src/society_simulation/social_media_config.py`
- Modify: `src/society_simulation/config.py`
- Test: `tests/test_social_media_config.py`

- [ ] **Step 1: Write config tests**

Create `tests/test_social_media_config.py`:

```python
import json

import pytest

from society_simulation.config import load_config
from society_simulation.social_media_config import InstagramSocialDynamicsConfig


def _valid_config() -> dict[str, object]:
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
    path.write_text(json.dumps(_valid_config()), encoding="utf-8")

    config = load_config(path)

    assert isinstance(config, InstagramSocialDynamicsConfig)
    assert config.experiment_name == "instagram_social_dynamics"
    assert config.feed_policy["type"] == "engagement_ranked"


def test_reject_invalid_feed_policy_type() -> None:
    data = _valid_config()
    data["feed_policy"] = dict(data["feed_policy"], type="magic")

    with pytest.raises(ValueError, match="unsupported feed_policy type"):
        InstagramSocialDynamicsConfig.from_dict(data).validate()


def test_reject_secret_bearing_update_policy_keys() -> None:
    data = _valid_config()
    data["update_policy"] = {"type": "llm", "model": "x", "api_key": "secret"}

    with pytest.raises(ValueError, match="secret-bearing update_policy key"):
        InstagramSocialDynamicsConfig.from_dict(data)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
./.venv/bin/python -m pytest tests/test_social_media_config.py -q
```

Expected: FAIL because config module does not exist.

- [ ] **Step 3: Implement social media config**

Create `src/society_simulation/social_media_config.py`:

```python
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
    parsed: list[str] = []
    for index, item in enumerate(value):
        parsed.append(_require_non_empty_str(item, f"{field}[{index}]"))
    return tuple(parsed)


def _freeze_json_mapping(value: object, field: str) -> Mapping[str, object]:
    data = _require_mapping(value, field)
    frozen: dict[str, object] = {}
    for key, item in data.items():
        if not isinstance(key, str):
            raise ValueError("object keys must be strings")
        if isinstance(item, dict):
            frozen[key] = _freeze_json_mapping(item, f"{field}.{key}")
        elif isinstance(item, list):
            frozen[key] = tuple(item)
        elif item is None or isinstance(item, (str, bool, int, float)):
            if isinstance(item, float) and not isfinite(item):
                raise ValueError("free-form config values must contain finite numbers")
            frozen[key] = item
        else:
            raise ValueError("free-form config values must be JSON-compatible")
    return MappingProxyType(frozen)


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
_SECRET_UPDATE_POLICY_KEY_PATTERNS = ("api_key", "api-key", "apikey", "authorization", "header")


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
```

Continue the same file with validation helpers:

```python
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
        "explore_fraction",
    ):
        _require_float(feed_policy.get(field), f"feed_policy.{field}")
    _validate_probability(feed_policy.get("explore_fraction"), "feed_policy.explore_fraction")


def _validate_update_policy(update_policy: Mapping[str, object]) -> None:
    policy_type = _require_non_empty_str(update_policy.get("type"), "update_policy.type")
    if policy_type == "mock_social":
        style = update_policy.get("response_style", "balanced")
        if style not in ("balanced", "endorsement_sensitive", "privacy_sensitive"):
            raise ValueError("unsupported mock_social response_style")
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
```

- [ ] **Step 4: Wire config loader**

Modify `src/society_simulation/config.py`:

```python
from society_simulation.social_media_config import InstagramSocialDynamicsConfig
```

Change the `Config` alias:

```python
Config = (
    ExperimentConfig
    | NetworkHerdingConfig
    | EventDrivenOpinionConfig
    | InstagramSocialDynamicsConfig
)
```

Add this branch to `load_config` before the sequential fallback:

```python
    elif data.get("experiment_name") == "instagram_social_dynamics":
        config = InstagramSocialDynamicsConfig.from_dict(data)
```

- [ ] **Step 5: Run config tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_social_media_config.py tests/test_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/society_simulation/social_media_config.py src/society_simulation/config.py tests/test_social_media_config.py
git commit -m "feat: add instagram social config"
```

---

## Task 3: Deterministic Seed Generator

**Files:**
- Create: `src/society_simulation/social_media_seed.py`
- Test: `tests/test_social_media_seed.py`

- [ ] **Step 1: Write seed tests**

Create `tests/test_social_media_seed.py`:

```python
from society_simulation.social_media_config import InstagramSocialDynamicsConfig
from society_simulation.social_media_seed import build_initial_world


def _config(seed: int = 11) -> InstagramSocialDynamicsConfig:
    return InstagramSocialDynamicsConfig.from_dict(
        {
            "experiment_name": "instagram_social_dynamics",
            "seed": seed,
            "scenario_name": "seed_test",
            "ticks": 2,
            "num_users": 6,
            "historical_posts_per_user": 2,
            "feed_size": 3,
            "activation_probability": 1.0,
            "topics": ["transit", "housing"],
            "seed_generator": {
                "type": "synthetic_profiles",
                "mean_following": 2,
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
            "update_policy": {"type": "mock_social", "response_style": "balanced"},
            "memory_retrieval": {"enabled": True, "limit": 5},
            "output_dir": "runs/seed_test",
        }
    )


def test_seed_generator_is_deterministic() -> None:
    first = build_initial_world(_config(seed=11))
    second = build_initial_world(_config(seed=11))

    assert [profile.to_dict() for profile in first.profiles] == [
        profile.to_dict() for profile in second.profiles
    ]
    assert [edge.to_dict() for edge in first.follow_edges] == [
        edge.to_dict() for edge in second.follow_edges
    ]
    assert [post.to_dict() for post in first.posts] == [
        post.to_dict() for post in second.posts
    ]


def test_seed_generator_creates_historical_posts_for_each_user() -> None:
    world = build_initial_world(_config())

    assert len(world.profiles) == 6
    assert len(world.states) == 6
    assert len(world.posts) == 12
    assert {post.author_id for post in world.posts} == {0, 1, 2, 3, 4, 5}


def test_seed_generator_has_no_self_follows() -> None:
    world = build_initial_world(_config())

    assert all(edge.follower_id != edge.followed_id for edge in world.follow_edges)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
./.venv/bin/python -m pytest tests/test_social_media_seed.py -q
```

Expected: FAIL because `build_initial_world` does not exist.

- [ ] **Step 3: Implement deterministic seed generator**

Create `src/society_simulation/social_media_seed.py`:

```python
from __future__ import annotations

import random

from society_simulation.social_media_config import InstagramSocialDynamicsConfig
from society_simulation.social_media_models import (
    FollowEdge,
    SocialMediaPost,
    SocialMediaUserProfile,
    SocialMediaUserState,
    SocialMediaWorld,
)

_NAMES = (
    "Minho Park",
    "Carlos Rivera",
    "Aisha Khan",
    "Nora Lee",
    "Sam Patel",
    "Jules Martin",
    "Maya Chen",
    "Theo Brooks",
)
_CLUSTERS = ("urbanists", "parents", "small_business", "students")
_STYLES = ("dry and specific", "warm and personal", "skeptical", "brief and witty")


def build_initial_world(config: InstagramSocialDynamicsConfig) -> SocialMediaWorld:
    rng = random.Random(config.seed)
    profiles = tuple(_build_profile(index, config, rng) for index in range(config.num_users))
    states = tuple(_initial_state(profile) for profile in profiles)
    follow_edges = _build_follow_edges(config, profiles, rng)
    posts = _build_historical_posts(config, profiles, rng)
    return SocialMediaWorld(
        profiles=profiles,
        states=states,
        posts=posts,
        follow_edges=follow_edges,
    )


def _build_profile(
    user_id: int,
    config: InstagramSocialDynamicsConfig,
    rng: random.Random,
) -> SocialMediaUserProfile:
    name = _NAMES[user_id % len(_NAMES)]
    handle = name.lower().replace(" ", "_") + f"_{user_id}"
    interests = tuple(rng.sample(list(config.topics), k=min(2, len(config.topics))))
    stance = round(rng.uniform(-0.8, 0.8), 3)
    return SocialMediaUserProfile(
        user_id=user_id,
        handle=handle,
        display_name=name,
        bio=f"Posts about {', '.join(interests)}.",
        interests=interests,
        home_cluster=_CLUSTERS[user_id % len(_CLUSTERS)],
        initial_stance=stance,
        activity_rate=round(rng.uniform(0.55, 0.95), 3),
        post_rate=round(rng.uniform(0.10, 0.35), 3),
        privacy_preference=round(rng.uniform(0.10, 0.75), 3),
        conformity=round(rng.uniform(0.10, 0.90), 3),
        skepticism=round(rng.uniform(0.10, 0.90), 3),
        conflict_tolerance=round(rng.uniform(0.10, 0.90), 3),
        status_weight=round(rng.uniform(0.05, 0.80), 3),
        posting_style=_STYLES[user_id % len(_STYLES)],
    )
```

Continue with graph and posts:

```python
def _initial_state(profile: SocialMediaUserProfile) -> SocialMediaUserState:
    return SocialMediaUserState(
        user_id=profile.user_id,
        tick=0,
        stance=profile.initial_stance,
        confidence=0.55,
        salience=0.45,
        mood="neutral",
        perceived_majority=0.0,
        social_fatigue=0.0,
        last_action_type="initial",
    )


def _build_follow_edges(
    config: InstagramSocialDynamicsConfig,
    profiles: tuple[SocialMediaUserProfile, ...],
    rng: random.Random,
) -> tuple[FollowEdge, ...]:
    mean_following = int(config.seed_generator["mean_following"])
    homophily_weight = float(config.seed_generator["homophily_weight"])
    random_tie_probability = float(config.seed_generator["random_tie_probability"])
    mutual_follow_probability = float(config.seed_generator["mutual_follow_probability"])
    edges: set[tuple[int, int]] = set()
    for follower in profiles:
        candidates = [profile for profile in profiles if profile.user_id != follower.user_id]
        scored = sorted(
            candidates,
            key=lambda candidate: (
                -_tie_score(follower, candidate, homophily_weight, random_tie_probability, rng),
                candidate.user_id,
            ),
        )
        for followed in scored[:mean_following]:
            edges.add((follower.user_id, followed.user_id))
            if rng.random() < mutual_follow_probability:
                edges.add((followed.user_id, follower.user_id))
    return tuple(
        FollowEdge(follower_id=follower_id, followed_id=followed_id, created_tick=0)
        for follower_id, followed_id in sorted(edges)
        if follower_id != followed_id
    )


def _tie_score(
    follower: SocialMediaUserProfile,
    candidate: SocialMediaUserProfile,
    homophily_weight: float,
    random_tie_probability: float,
    rng: random.Random,
) -> float:
    shared_interests = len(set(follower.interests) & set(candidate.interests))
    stance_similarity = 1.0 - min(1.0, abs(follower.initial_stance - candidate.initial_stance))
    homophily = (shared_interests / max(1, len(follower.interests))) + stance_similarity
    return homophily_weight * homophily + random_tie_probability * rng.random()


def _build_historical_posts(
    config: InstagramSocialDynamicsConfig,
    profiles: tuple[SocialMediaUserProfile, ...],
    rng: random.Random,
) -> tuple[SocialMediaPost, ...]:
    posts: list[SocialMediaPost] = []
    for profile in profiles:
        for index in range(config.historical_posts_per_user):
            topic = profile.interests[index % len(profile.interests)]
            post_id = f"post-{profile.user_id}-{index}"
            posts.append(
                SocialMediaPost(
                    post_id=post_id,
                    author_id=profile.user_id,
                    topic=topic,
                    stance=round(profile.initial_stance + rng.uniform(-0.15, 0.15), 3),
                    text=f"{profile.display_name} shares a {profile.posting_style} note about {topic}.",
                    created_tick=-config.historical_posts_per_user + index,
                    like_count=rng.randint(0, 9),
                    reply_count=0,
                    seed_post=True,
                )
            )
    return tuple(posts)
```

- [ ] **Step 4: Run seed tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_social_media_seed.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/social_media_seed.py tests/test_social_media_seed.py
git commit -m "feat: seed instagram social worlds"
```

---

## Task 4: Feed and Recommendation Layer

**Files:**
- Create: `src/society_simulation/social_media_feed.py`
- Test: `tests/test_social_media_feed.py`

- [ ] **Step 1: Write feed tests**

Create `tests/test_social_media_feed.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
./.venv/bin/python -m pytest tests/test_social_media_feed.py -q
```

Expected: FAIL because feed module does not exist.

- [ ] **Step 3: Implement feed builder**

Create `src/society_simulation/social_media_feed.py`:

```python
from __future__ import annotations

from collections.abc import Mapping
from math import log
import random

from society_simulation.social_media_models import FeedItem, SocialMediaPost, SocialMediaWorld


def build_feed(
    *,
    world: SocialMediaWorld,
    viewer_id: int,
    tick: int,
    feed_size: int,
    feed_policy: Mapping[str, object],
    seed: int,
) -> tuple[FeedItem, ...]:
    policy_type = str(feed_policy["type"])
    if policy_type == "no_feed_control":
        return ()
    followed_ids = _followed_ids(world, viewer_id)
    candidates = _candidate_posts(world, viewer_id, followed_ids, policy_type)
    rng = random.Random(seed + tick * 1009 + viewer_id * 9173)
    scored = [
        _score_post(
            world=world,
            viewer_id=viewer_id,
            post=post,
            followed_ids=followed_ids,
            tick=tick,
            feed_policy=feed_policy,
            rng=rng,
        )
        for post in candidates
    ]
    ranked = sorted(scored, key=lambda item: (-item[0], item[1].post_id))
    return tuple(
        FeedItem(
            tick=tick,
            viewer_id=viewer_id,
            post_id=post.post_id,
            author_id=post.author_id,
            score=round(score, 6),
            rank=rank,
            source="following" if post.author_id in followed_ids else "explore",
            reason=reason,
        )
        for rank, (score, post, reason) in enumerate(ranked[:feed_size])
    )
```

Continue the same file:

```python
def _followed_ids(world: SocialMediaWorld, viewer_id: int) -> set[int]:
    return {
        edge.followed_id
        for edge in world.follow_edges
        if edge.follower_id == viewer_id
    }


def _candidate_posts(
    world: SocialMediaWorld,
    viewer_id: int,
    followed_ids: set[int],
    policy_type: str,
) -> tuple[SocialMediaPost, ...]:
    if policy_type == "chronological_following":
        return tuple(post for post in world.posts if post.author_id in followed_ids)
    return tuple(post for post in world.posts if post.author_id != viewer_id)


def _score_post(
    *,
    world: SocialMediaWorld,
    viewer_id: int,
    post: SocialMediaPost,
    followed_ids: set[int],
    tick: int,
    feed_policy: Mapping[str, object],
    rng: random.Random,
) -> tuple[float, SocialMediaPost, str]:
    profile_by_id = world.profile_by_id()
    state_by_id = world.state_by_id()
    viewer_profile = profile_by_id[viewer_id]
    viewer_state = state_by_id[viewer_id]
    author_followers = sum(1 for edge in world.follow_edges if edge.followed_id == post.author_id)
    following_bonus = float(feed_policy["following_bonus"]) if post.author_id in followed_ids else 0.0
    interest_similarity = _interest_similarity(viewer_profile.interests, (post.topic,))
    stance_similarity = 1.0 - min(1.0, abs(viewer_state.stance - post.stance))
    stance_distance = min(1.0, abs(viewer_state.stance - post.stance))
    recency_decay = 1.0 / (1.0 + max(0, tick - post.created_tick))
    score = (
        following_bonus
        + float(feed_policy["interest_similarity_weight"]) * interest_similarity
        + float(feed_policy["stance_similarity_weight"]) * stance_similarity
        + float(feed_policy["engagement_weight"]) * log(1 + post.like_count)
        + float(feed_policy["recency_weight"]) * recency_decay
        + float(feed_policy["creator_popularity_weight"]) * log(1 + author_followers)
        + float(feed_policy["controversy_weight"]) * stance_distance
        + float(feed_policy["noise_weight"]) * rng.random()
    )
    reason = (
        f"following_bonus={following_bonus:.2f} "
        f"interest_similarity={interest_similarity:.2f} "
        f"stance_similarity={stance_similarity:.2f} "
        f"engagement={log(1 + post.like_count):.2f}"
    )
    return score, post, reason


def _interest_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)
```

- [ ] **Step 4: Run feed tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_social_media_feed.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/social_media_feed.py tests/test_social_media_feed.py
git commit -m "feat: add social media feed ranking"
```

---

## Task 5: Mock Social Policy and LLM Parser Contract

**Files:**
- Create: `src/society_simulation/social_media_policy.py`
- Test: `tests/test_social_media_policy.py`

- [ ] **Step 1: Write policy tests**

Create `tests/test_social_media_policy.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
./.venv/bin/python -m pytest tests/test_social_media_policy.py -q
```

Expected: FAIL because policy module does not exist.

- [ ] **Step 3: Implement parser and mock policy**

Create `src/society_simulation/social_media_policy.py`:

```python
from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Protocol

from society_simulation.social_media_models import (
    FeedItem,
    PlatformAction,
    SocialActionType,
    SocialMediaUserProfile,
    SocialMediaUserState,
)

_SUPPORTED_ACTIONS: set[str] = {
    "like_post",
    "follow_user",
    "unfollow_user",
    "send_dm",
    "create_post",
    "do_nothing",
}


class SocialMediaPolicy(Protocol):
    def decide(
        self,
        *,
        profile: SocialMediaUserProfile,
        state: SocialMediaUserState,
        feed: Sequence[FeedItem],
        tick: int,
    ) -> PlatformAction:
        raise NotImplementedError


def parse_social_action_content(content: str, *, tick: int, user_id: int) -> PlatformAction:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("social llm response content must be JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("social llm response content must be a JSON object")
    action_type = _require_action_type(data.get("action_type"))
    return PlatformAction(
        tick=tick,
        user_id=user_id,
        action_type=action_type,
        post_id=_optional_str(data.get("post_id"), "post_id"),
        target_user_id=_optional_int(data.get("target_user_id"), "target_user_id"),
        text=_optional_str(data.get("text"), "text"),
        topic=_optional_str(data.get("topic"), "topic"),
        stance=_optional_float(data.get("stance"), "stance"),
        reason=_require_str(data.get("reason"), "reason"),
    )
```

Continue the same file:

```python
class MockSocialMediaPolicy:
    name = "mock_social"

    def __init__(self, *, response_style: str = "balanced") -> None:
        if response_style not in ("balanced", "endorsement_sensitive", "privacy_sensitive"):
            raise ValueError("unsupported mock_social response_style")
        self.response_style = response_style

    def decide(
        self,
        *,
        profile: SocialMediaUserProfile,
        state: SocialMediaUserState,
        feed: Sequence[FeedItem],
        tick: int,
    ) -> PlatformAction:
        if not feed:
            return _do_nothing(tick, profile.user_id, "No feed items were available.")
        top_item = sorted(feed, key=lambda item: (-item.score, item.rank))[0]
        if self.response_style == "privacy_sensitive" and profile.privacy_preference > 0.5:
            return PlatformAction(
                tick=tick,
                user_id=profile.user_id,
                action_type="send_dm",
                post_id=top_item.post_id,
                target_user_id=top_item.author_id,
                text="I saw your post and wanted to ask about it privately.",
                topic=None,
                stance=state.stance,
                reason="High privacy preference favors a private backchannel.",
            )
        if self.response_style == "endorsement_sensitive" or top_item.score >= 1.0:
            return PlatformAction(
                tick=tick,
                user_id=profile.user_id,
                action_type="like_post",
                post_id=top_item.post_id,
                target_user_id=None,
                text=None,
                topic=None,
                stance=None,
                reason="Top-ranked feed item had enough visible social signal.",
            )
        return _do_nothing(tick, profile.user_id, "Feed did not cross action threshold.")


def _do_nothing(tick: int, user_id: int, reason: str) -> PlatformAction:
    return PlatformAction(
        tick=tick,
        user_id=user_id,
        action_type="do_nothing",
        post_id=None,
        target_user_id=None,
        text=None,
        topic=None,
        stance=None,
        reason=reason,
    )


def _require_action_type(value: object) -> SocialActionType:
    text = _require_str(value, "action_type")
    if text not in _SUPPORTED_ACTIONS:
        raise ValueError("unsupported social action")
    return text  # type: ignore[return-value]


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_str(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, field)


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _optional_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)
```

- [ ] **Step 4: Run policy tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_social_media_policy.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/social_media_policy.py tests/test_social_media_policy.py
git commit -m "feat: add social media policy contract"
```

---

## Task 6: Runner and Action Application

**Files:**
- Create: `src/society_simulation/social_media_runner.py`
- Modify: `src/society_simulation/runner.py`
- Test: `tests/test_social_media_runner.py`

- [ ] **Step 1: Write runner tests**

Create `tests/test_social_media_runner.py`:

```python
from pathlib import Path

from society_simulation.config import load_config
from society_simulation.runner import run_experiment
from society_simulation.social_media_runner import run_instagram_social_dynamics


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "experiment_name": "instagram_social_dynamics",
        "seed": 3,
        "scenario_name": "runner_test",
        "ticks": 2,
        "num_users": 5,
        "historical_posts_per_user": 1,
        "feed_size": 2,
        "activation_probability": 1.0,
        "topics": ["transit", "housing"],
        "seed_generator": {
            "type": "synthetic_profiles",
            "mean_following": 2,
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
        "update_policy": {"type": "mock_social", "response_style": "endorsement_sensitive"},
        "memory_retrieval": {"enabled": True, "limit": 5},
        "output_dir": str(tmp_path / "run"),
    }


def test_run_instagram_social_dynamics_writes_actions(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(__import__("json").dumps(_config(tmp_path)), encoding="utf-8")
    config = load_config(config_path)

    result = run_instagram_social_dynamics(config)

    assert result.metrics["action_count"] > 0
    assert result.output_dir == tmp_path / "run"


def test_runner_dispatches_from_generic_run_experiment(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(__import__("json").dumps(_config(tmp_path)), encoding="utf-8")
    config = load_config(config_path)

    result = run_experiment(config)

    assert "like_post" in result.metrics["action_counts"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
./.venv/bin/python -m pytest tests/test_social_media_runner.py -q
```

Expected: FAIL because runner module does not exist and dispatch is missing.

- [ ] **Step 3: Implement runner skeleton and action application**

Create `src/society_simulation/social_media_runner.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import random
from typing import Any

from society_simulation.social_media_config import InstagramSocialDynamicsConfig
from society_simulation.social_media_feed import build_feed
from society_simulation.social_media_metrics import compute_social_media_metrics
from society_simulation.social_media_models import (
    DirectMessage,
    FeedItem,
    FollowEdge,
    PlatformAction,
    SocialMediaPost,
    SocialMediaUserState,
    SocialMediaWorld,
)
from society_simulation.social_media_policy import MockSocialMediaPolicy
from society_simulation.social_media_replay import SocialMediaReplayWriter
from society_simulation.social_media_seed import build_initial_world


@dataclass(frozen=True)
class SocialMediaRunResult:
    final_world: SocialMediaWorld
    feed_items: tuple[FeedItem, ...]
    actions: tuple[PlatformAction, ...]
    dm_messages: tuple[DirectMessage, ...]
    states_by_tick: tuple[tuple[SocialMediaUserState, ...], ...]
    metrics: dict[str, Any]
    output_dir: Path


def run_instagram_social_dynamics(
    config: InstagramSocialDynamicsConfig,
) -> SocialMediaRunResult:
    config.validate()
    rng = random.Random(config.seed)
    policy = _build_policy(config)
    world = build_initial_world(config)
    states_by_tick: list[tuple[SocialMediaUserState, ...]] = [world.states]
    feed_items: list[FeedItem] = []
    actions: list[PlatformAction] = []
    dm_messages: list[DirectMessage] = []
    initial_edges = world.follow_edges
    for tick in range(1, config.ticks + 1):
        current_states = {state.user_id: state for state in world.states}
        for profile in world.profiles:
            if rng.random() > config.activation_probability * profile.activity_rate:
                continue
            feed = build_feed(
                world=world,
                viewer_id=profile.user_id,
                tick=tick,
                feed_size=config.feed_size,
                feed_policy=config.feed_policy,
                seed=config.seed,
            )
            feed_items.extend(feed)
            action = policy.decide(
                profile=profile,
                state=current_states[profile.user_id],
                feed=feed,
                tick=tick,
            )
            world, dm = apply_action(world, action)
            actions.append(action)
            if dm is not None:
                dm_messages.append(dm)
        states_by_tick.append(world.states)
    metrics = compute_social_media_metrics(
        initial_edges=initial_edges,
        final_world=world,
        feed_items=tuple(feed_items),
        actions=tuple(actions),
        dm_messages=tuple(dm_messages),
        states_by_tick=tuple(states_by_tick),
    )
    output_dir = SocialMediaReplayWriter(config).write(
        final_world=world,
        initial_edges=initial_edges,
        feed_items=tuple(feed_items),
        actions=tuple(actions),
        dm_messages=tuple(dm_messages),
        states_by_tick=tuple(states_by_tick),
        metrics=metrics,
        llm_decisions=(),
    )
    return SocialMediaRunResult(
        final_world=world,
        feed_items=tuple(feed_items),
        actions=tuple(actions),
        dm_messages=tuple(dm_messages),
        states_by_tick=tuple(states_by_tick),
        metrics=metrics,
        output_dir=output_dir,
    )
```

Continue the same file:

```python
def _build_policy(config: InstagramSocialDynamicsConfig) -> MockSocialMediaPolicy:
    policy_type = str(config.update_policy["type"])
    if policy_type == "mock_social":
        return MockSocialMediaPolicy(
            response_style=str(config.update_policy.get("response_style", "balanced")),
        )
    raise ValueError("social media llm policy is not wired in this task")


def apply_action(
    world: SocialMediaWorld,
    action: PlatformAction,
) -> tuple[SocialMediaWorld, DirectMessage | None]:
    posts = list(world.posts)
    edges = set((edge.follower_id, edge.followed_id) for edge in world.follow_edges)
    dm: DirectMessage | None = None
    if action.action_type == "like_post" and action.post_id is not None:
        posts = [post.with_like() if post.post_id == action.post_id else post for post in posts]
    elif action.action_type == "follow_user" and action.target_user_id is not None:
        if action.target_user_id != action.user_id:
            edges.add((action.user_id, action.target_user_id))
    elif action.action_type == "unfollow_user" and action.target_user_id is not None:
        edges.discard((action.user_id, action.target_user_id))
    elif action.action_type == "send_dm" and action.target_user_id is not None and action.text:
        dm = DirectMessage(
            tick=action.tick,
            sender_id=action.user_id,
            recipient_id=action.target_user_id,
            text=action.text,
            topic=action.topic,
            stance=action.stance,
        )
    elif action.action_type == "create_post" and action.text and action.topic:
        posts.append(
            SocialMediaPost(
                post_id=f"post-generated-{action.tick}-{action.user_id}-{len(posts)}",
                author_id=action.user_id,
                topic=action.topic,
                stance=action.stance if action.stance is not None else 0.0,
                text=action.text,
                created_tick=action.tick,
                like_count=0,
                reply_count=0,
                seed_post=False,
            )
        )
    next_states = tuple(
        replace(state, tick=action.tick, last_action_type=action.action_type)
        if state.user_id == action.user_id
        else state
        for state in world.states
    )
    next_edges = tuple(
        FollowEdge(follower_id=follower, followed_id=followed, created_tick=0)
        for follower, followed in sorted(edges)
        if follower != followed
    )
    return (
        SocialMediaWorld(
            profiles=world.profiles,
            states=next_states,
            posts=tuple(posts),
            follow_edges=next_edges,
        ),
        dm,
    )
```

- [ ] **Step 4: Wire generic runner**

Modify `src/society_simulation/runner.py` imports:

```python
from society_simulation.social_media_config import InstagramSocialDynamicsConfig
from society_simulation.social_media_runner import SocialMediaRunResult, run_instagram_social_dynamics
```

Change return type and dispatch:

```python
def run_experiment(
    config: Config,
) -> RunResult | NetworkRunResult | EventRunResult | SocialMediaRunResult:
    if isinstance(config, NetworkHerdingConfig):
        return run_network_herding(config)
    if isinstance(config, EventDrivenOpinionConfig):
        return run_event_driven_opinion_dynamics(config)
    if isinstance(config, InstagramSocialDynamicsConfig):
        return run_instagram_social_dynamics(config)
    return run_sequential_information_cascade(config)
```

- [ ] **Step 5: Run runner tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_social_media_runner.py -q
```

Expected: PASS after Task 7 adds metrics and replay. If this task is implemented before Task 7, keep the test committed with Task 7.

- [ ] **Step 6: Commit after Task 7 dependencies pass**

```bash
git add src/society_simulation/social_media_runner.py src/society_simulation/runner.py tests/test_social_media_runner.py
git commit -m "feat: run instagram social dynamics"
```

---

## Task 7: Metrics and Replay

**Files:**
- Create: `src/society_simulation/social_media_metrics.py`
- Create: `src/society_simulation/social_media_replay.py`
- Test: `tests/test_social_media_metrics.py`
- Test: `tests/test_social_media_replay.py`
- Test: `tests/test_social_media_runner.py`

- [ ] **Step 1: Write metrics tests**

Create `tests/test_social_media_metrics.py`:

```python
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
    assert metrics["follow_edge_delta"] == 1
    assert metrics["dm_count"] == 1
```

- [ ] **Step 2: Implement metrics**

Create `src/society_simulation/social_media_metrics.py`:

```python
from __future__ import annotations

from collections import Counter
from statistics import mean, variance
from typing import Any

from society_simulation.social_media_models import (
    DirectMessage,
    FeedItem,
    FollowEdge,
    PlatformAction,
    SocialMediaUserState,
    SocialMediaWorld,
)


def compute_social_media_metrics(
    *,
    initial_edges: tuple[FollowEdge, ...],
    final_world: SocialMediaWorld,
    feed_items: tuple[FeedItem, ...],
    actions: tuple[PlatformAction, ...],
    dm_messages: tuple[DirectMessage, ...],
    states_by_tick: tuple[tuple[SocialMediaUserState, ...], ...],
) -> dict[str, Any]:
    action_counts = Counter(action.action_type for action in actions)
    stance_values = [state.stance for state in final_world.states]
    initial_edge_set = {(edge.follower_id, edge.followed_id) for edge in initial_edges}
    final_edge_set = {(edge.follower_id, edge.followed_id) for edge in final_world.follow_edges}
    return {
        "experiment_family": "instagram_social_dynamics",
        "user_count": len(final_world.profiles),
        "post_count": len(final_world.posts),
        "feed_impression_count": len(feed_items),
        "action_count": len(actions),
        "action_counts": dict(action_counts),
        "like_count": action_counts.get("like_post", 0),
        "dm_count": len(dm_messages),
        "follow_count": action_counts.get("follow_user", 0),
        "unfollow_count": action_counts.get("unfollow_user", 0),
        "initial_follow_edge_count": len(initial_edge_set),
        "final_follow_edge_count": len(final_edge_set),
        "follow_edge_delta": len(final_edge_set) - len(initial_edge_set),
        "new_follow_edge_count": len(final_edge_set - initial_edge_set),
        "removed_follow_edge_count": len(initial_edge_set - final_edge_set),
        "final_stance_mean": round(mean(stance_values), 6) if stance_values else 0.0,
        "final_stance_variance": round(variance(stance_values), 6) if len(stance_values) > 1 else 0.0,
        "exposure_diversity": _exposure_diversity(feed_items),
        "states_recorded": sum(len(states) for states in states_by_tick),
    }


def _exposure_diversity(feed_items: tuple[FeedItem, ...]) -> float:
    if not feed_items:
        return 0.0
    authors_by_viewer: dict[int, set[int]] = {}
    for item in feed_items:
        authors_by_viewer.setdefault(item.viewer_id, set()).add(item.author_id)
    return round(mean(len(authors) for authors in authors_by_viewer.values()), 6)
```

- [ ] **Step 3: Write replay tests**

Create `tests/test_social_media_replay.py`:

```python
import json

from society_simulation.social_media_config import InstagramSocialDynamicsConfig
from society_simulation.social_media_replay import SocialMediaReplayWriter
from society_simulation.social_media_seed import build_initial_world


def test_replay_writer_writes_core_artifacts(tmp_path) -> None:
    config = InstagramSocialDynamicsConfig.from_dict(
        {
            "experiment_name": "instagram_social_dynamics",
            "seed": 1,
            "scenario_name": "replay_test",
            "ticks": 1,
            "num_users": 3,
            "historical_posts_per_user": 1,
            "feed_size": 2,
            "activation_probability": 1.0,
            "topics": ["transit"],
            "seed_generator": {
                "type": "synthetic_profiles",
                "mean_following": 1,
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
            "update_policy": {"type": "mock_social", "response_style": "balanced"},
            "memory_retrieval": {"enabled": False, "limit": 5},
            "output_dir": str(tmp_path / "run"),
        }
    )
    world = build_initial_world(config)

    output_dir = SocialMediaReplayWriter(config).write(
        final_world=world,
        initial_edges=world.follow_edges,
        feed_items=(),
        actions=(),
        dm_messages=(),
        states_by_tick=(world.states,),
        metrics={"action_count": 0},
        llm_decisions=(),
    )

    assert (output_dir / "config.json").exists()
    assert (output_dir / "users.jsonl").exists()
    assert json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))["action_count"] == 0
```

- [ ] **Step 4: Implement replay writer**

Create `src/society_simulation/social_media_replay.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from society_simulation.social_media_config import InstagramSocialDynamicsConfig
from society_simulation.social_media_models import (
    DirectMessage,
    FeedItem,
    FollowEdge,
    PlatformAction,
    SocialMediaUserState,
    SocialMediaWorld,
)


class SocialMediaReplayWriter:
    def __init__(self, config: InstagramSocialDynamicsConfig) -> None:
        self.config = config

    def write(
        self,
        *,
        final_world: SocialMediaWorld,
        initial_edges: tuple[FollowEdge, ...],
        feed_items: tuple[FeedItem, ...],
        actions: tuple[PlatformAction, ...],
        dm_messages: tuple[DirectMessage, ...],
        states_by_tick: tuple[tuple[SocialMediaUserState, ...], ...],
        metrics: dict[str, Any],
        llm_decisions: tuple[dict[str, Any], ...],
    ) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(output_dir / "config.json", self.config.to_dict())
        self._write_jsonl(output_dir / "users.jsonl", tuple(p.to_dict() for p in final_world.profiles))
        self._write_jsonl(output_dir / "posts.jsonl", tuple(p.to_dict() for p in final_world.posts))
        self._write_jsonl(output_dir / "follow_edges_initial.jsonl", tuple(e.to_dict() for e in initial_edges))
        self._write_jsonl(output_dir / "follow_edges_final.jsonl", tuple(e.to_dict() for e in final_world.follow_edges))
        self._write_jsonl(output_dir / "feed_impressions.jsonl", tuple(item.to_dict() for item in feed_items))
        self._write_jsonl(output_dir / "actions.jsonl", tuple(action.to_dict() for action in actions))
        self._write_jsonl(output_dir / "dm_messages.jsonl", tuple(message.to_dict() for message in dm_messages))
        self._write_jsonl(
            output_dir / "user_states.jsonl",
            tuple(state.to_dict() for tick_states in states_by_tick for state in tick_states),
        )
        self._write_json(output_dir / "metrics.json", metrics)
        self._write_jsonl(output_dir / "llm_decisions.jsonl", llm_decisions)
        self._write_summary(output_dir / "summary.md", metrics)
        return output_dir

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_jsonl(self, path: Path, rows: tuple[dict[str, Any], ...]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")

    def _write_summary(self, path: Path, metrics: dict[str, Any]) -> None:
        lines = [
            f"# {self.config.scenario_name}",
            "",
            f"- experiment_name: `{self.config.experiment_name}`",
            f"- user_count: `{metrics.get('user_count')}`",
            f"- action_count: `{metrics.get('action_count')}`",
            f"- feed_impression_count: `{metrics.get('feed_impression_count')}`",
            f"- action_counts: `{metrics.get('action_counts')}`",
            f"- final_follow_edge_count: `{metrics.get('final_follow_edge_count')}`",
            f"- final_stance_mean: `{metrics.get('final_stance_mean')}`",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 5: Run metrics, replay, and runner tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_social_media_metrics.py tests/test_social_media_replay.py tests/test_social_media_runner.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/society_simulation/social_media_metrics.py src/society_simulation/social_media_replay.py tests/test_social_media_metrics.py tests/test_social_media_replay.py tests/test_social_media_runner.py
git commit -m "feat: add social media metrics and replay"
```

---

## Task 8: CLI, Example Config, and Smoke Run

**Files:**
- Modify: `src/society_simulation/cli.py`
- Create: `experiments/instagram_social_lite_mock.json`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add CLI test**

Modify `tests/test_cli.py` with a social-media run test:

```python
def test_cli_runs_instagram_social_lite_mock(tmp_path, capsys) -> None:
    import json
    from society_simulation.cli import main

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_name": "instagram_social_dynamics",
                "seed": 5,
                "scenario_name": "cli_social",
                "ticks": 1,
                "num_users": 4,
                "historical_posts_per_user": 1,
                "feed_size": 2,
                "activation_probability": 1.0,
                "topics": ["transit", "housing"],
                "seed_generator": {
                    "type": "synthetic_profiles",
                    "mean_following": 1,
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
                "update_policy": {"type": "mock_social", "response_style": "balanced"},
                "memory_retrieval": {"enabled": False, "limit": 5},
                "output_dir": str(tmp_path / "run"),
            }
        ),
        encoding="utf-8",
    )

    assert main(["run", str(config_path)]) == 0
    captured = capsys.readouterr()
    assert "experiment=instagram_social_dynamics" in captured.out
    assert "feed_impression_count=" in captured.out
```

- [ ] **Step 2: Modify CLI output**

In `src/society_simulation/cli.py`, add:

```python
SOCIAL_MEDIA_SUMMARY_FIELDS = (
    "feed_impression_count",
    "action_count",
    "final_follow_edge_count",
    "final_stance_mean",
)


def _has_social_media_summary_metrics(metrics: dict[str, object]) -> bool:
    return all(field in metrics for field in SOCIAL_MEDIA_SUMMARY_FIELDS)
```

In `_run_single_config`, after event metrics branch:

```python
    elif _has_social_media_summary_metrics(metrics):
        print(f"feed_impression_count={metrics['feed_impression_count']}")
        print(f"action_count={metrics['action_count']}")
        print(f"action_counts={metrics.get('action_counts')}")
        print(f"final_follow_edge_count={metrics['final_follow_edge_count']}")
        print(f"final_stance_mean={metrics['final_stance_mean']}")
    else:
        print(f"action_counts={action_counts}")
```

Adjust `action_counts` assignment so `_require_action_counts` is only called
when neither event nor social-media metrics are present:

```python
        has_event_metrics = _has_event_summary_metrics(metrics)
        has_social_metrics = _has_social_media_summary_metrics(metrics)
        action_counts = None if has_event_metrics or has_social_metrics else _require_action_counts(metrics)
```

- [ ] **Step 3: Create example config**

Create `experiments/instagram_social_lite_mock.json`:

```json
{
  "experiment_name": "instagram_social_dynamics",
  "seed": 20260622,
  "scenario_name": "instagram_social_lite_mock",
  "ticks": 4,
  "num_users": 12,
  "historical_posts_per_user": 2,
  "feed_size": 4,
  "activation_probability": 1.0,
  "topics": ["transit", "housing", "local_business", "schools"],
  "seed_generator": {
    "type": "synthetic_profiles",
    "mean_following": 4,
    "homophily_weight": 0.55,
    "popularity_weight": 0.25,
    "random_tie_probability": 0.1,
    "mutual_follow_probability": 0.4
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
    "explore_fraction": 0.33
  },
  "update_policy": {
    "type": "mock_social",
    "response_style": "endorsement_sensitive",
    "input_cost_per_1m_tokens": 0.0,
    "output_cost_per_1m_tokens": 0.0
  },
  "memory_retrieval": {
    "enabled": true,
    "limit": 5
  },
  "output_dir": "runs/instagram_social_lite_mock"
}
```

- [ ] **Step 4: Run CLI tests and smoke run**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cli.py tests/test_social_media_config.py tests/test_social_media_runner.py -q
./.venv/bin/python -m society_simulation run experiments/instagram_social_lite_mock.json
```

Expected CLI output includes:

```text
experiment=instagram_social_dynamics
feed_impression_count=
action_count=
final_follow_edge_count=
final_stance_mean=
output_dir=runs/instagram_social_lite_mock
```

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/cli.py tests/test_cli.py experiments/instagram_social_lite_mock.json
git commit -m "feat: expose instagram social mock run"
```

---

## Task 9: LLM Policy Wrapper With Cost Guard

**Files:**
- Modify: `src/society_simulation/social_media_policy.py`
- Modify: `src/society_simulation/social_media_runner.py`
- Test: `tests/test_social_media_policy.py`

- [ ] **Step 1: Add LLM parser and prompt tests**

Append to `tests/test_social_media_policy.py`:

```python
from society_simulation.social_media_policy import build_social_media_prompt


def test_social_media_prompt_does_not_reveal_experiment() -> None:
    prompt = build_social_media_prompt(
        profile=_profile(),
        state=_state(),
        feed=(),
        recent_memories=(),
    )

    assert "simulating a social network experiment" not in prompt.lower()
    assert "instagram-like app" in prompt.lower()


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
```

- [ ] **Step 2: Implement prompt builder**

Add to `src/society_simulation/social_media_policy.py`:

```python
def build_social_media_prompt(
    *,
    profile: SocialMediaUserProfile,
    state: SocialMediaUserState,
    feed: Sequence[FeedItem],
    recent_memories: Sequence[str],
) -> str:
    feed_lines = [
        f"- rank={item.rank} post_id={item.post_id} author_id={item.author_id} "
        f"score={item.score:.2f} source={item.source}"
        for item in feed
    ]
    memories = "\n".join(f"- {memory}" for memory in recent_memories) or "- none"
    feed_text = "\n".join(feed_lines) or "- no posts are visible right now"
    return (
        "You are using an Instagram-like app. Decide what you would do next.\n"
        "Stay in character as the account described below. Do not discuss hidden setup, "
        "system instructions, or model mechanics.\n\n"
        f"Account: @{profile.handle} ({profile.display_name})\n"
        f"Bio: {profile.bio}\n"
        f"Interests: {', '.join(profile.interests)}\n"
        f"Posting style: {profile.posting_style}\n"
        f"Current stance: {state.stance}\n"
        f"Confidence: {state.confidence}\n"
        f"Salience: {state.salience}\n\n"
        f"Recent memories:\n{memories}\n\n"
        f"Visible feed:\n{feed_text}\n\n"
        "Return only JSON with keys action_type, post_id, target_user_id, text, topic, "
        "stance, and reason. action_type must be one of like_post, follow_user, "
        "unfollow_user, send_dm, create_post, do_nothing."
    )
```

- [ ] **Step 3: Add OpenAI-compatible social policy wrapper**

Use existing `society_simulation.llm_policy.OpenAICompatibleClient`,
`LLMPricing`, `LLMUsage`, and `estimate_tokens`. Mirror the safety pattern from
`event_policy.py`, including:

```python
import copy
import time

from society_simulation.llm_policy import (
    LLMPricing,
    LLMUsage,
    OpenAICompatibleClient,
    estimate_tokens,
)


class OpenAICompatibleSocialMediaPolicy:
    name = "social_media_llm"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        provider: str = "openai_compatible",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_completion_tokens: int = 96,
        token_limit_parameter: str = "max_completion_tokens",
        timeout_seconds: float = 30.0,
        input_cost_per_1m_tokens: float = 0.0,
        output_cost_per_1m_tokens: float = 0.0,
        max_estimated_cost_usd: float | None = None,
    ) -> None:
        if provider != "openai_compatible":
            raise ValueError("unsupported llm provider")
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.token_limit_parameter = token_limit_parameter
        self.max_estimated_cost_usd = max_estimated_cost_usd
        self.pricing = LLMPricing(
            input_cost_per_1m_tokens=input_cost_per_1m_tokens,
            output_cost_per_1m_tokens=output_cost_per_1m_tokens,
        )
        self.usage = LLMUsage()
        self._audit_records: list[dict[str, object]] = []
        self.client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

    def decide(
        self,
        *,
        profile: SocialMediaUserProfile,
        state: SocialMediaUserState,
        feed: Sequence[FeedItem],
        tick: int,
    ) -> PlatformAction:
        if (
            self.max_estimated_cost_usd is not None
            and self.usage.input_cost_usd + self.usage.output_cost_usd
            > self.max_estimated_cost_usd
        ):
            raise ValueError("social media llm cost cap exceeded")
        prompt = build_social_media_prompt(
            profile=profile,
            state=state,
            feed=feed,
            recent_memories=(),
        )
        started = time.monotonic()
        response = self.client.create_chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            token_limit_parameter=self.token_limit_parameter,
            max_completion_tokens=self.max_completion_tokens,
        )
        latency_ms = (time.monotonic() - started) * 1000
        content = _extract_content(response)
        action = parse_social_action_content(content, tick=tick, user_id=profile.user_id)
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(content)
        self.usage.record(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            pricing=self.pricing,
        )
        self._audit_records.append(
            {
                "tick": tick,
                "user_id": profile.user_id,
                "provider": self.provider,
                "model": self.model,
                "policy_type": self.name,
                "prompt": prompt,
                "raw_response": _redact_response(response),
                "action": action.to_dict(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
            }
        )
        return action

    def usage_summary(self) -> dict[str, object]:
        return self.usage.summary(provider=self.provider, model=self.model)

    def audit_records(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(record) for record in self._audit_records)


def _extract_content(response: dict[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("social media llm response must include choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("social media llm choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("social media llm choice must include message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("social media llm message content must be non-empty")
    return content


def _redact_response(value: object) -> object:
    if isinstance(value, dict):
        redacted: dict[str, object] = {}
        for key, item in value.items():
            lowered = key.lower()
            if any(pattern in lowered for pattern in ("authorization", "api_key", "header")):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_response(item)
        return redacted
    if isinstance(value, list):
        return [_redact_response(item) for item in value]
    if isinstance(value, str) and "bearer " in value.lower():
        return "[REDACTED]"
    return copy.deepcopy(value)
```

The `decide` method should:

1. Build the prompt with `build_social_media_prompt`.
2. Call the OpenAI-compatible client.
3. Parse content with `parse_social_action_content`.
4. Record prompt tokens, completion tokens, latency, estimated cost, and raw
   redacted response.
5. Raise `ValueError("social media llm cost cap exceeded")` before a call if the
   current estimated cost is already greater than the configured cap.

- [ ] **Step 4: Wire LLM policy in runner**

Modify `_build_policy` in `social_media_runner.py`:

```python
def _build_policy(config: InstagramSocialDynamicsConfig):
    policy_type = str(config.update_policy["type"])
    if policy_type == "mock_social":
        return MockSocialMediaPolicy(
            response_style=str(config.update_policy.get("response_style", "balanced")),
        )
    if policy_type == "llm":
        api_key_env = str(config.update_policy.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} environment variable is required")
        return OpenAICompatibleSocialMediaPolicy(
            model=str(config.update_policy["model"]),
            api_key=api_key,
            provider=str(config.update_policy.get("provider", "openai_compatible")),
            base_url=str(config.update_policy.get("base_url", "https://api.openai.com/v1")),
            temperature=float(config.update_policy.get("temperature", 0.0)),
            max_completion_tokens=int(config.update_policy.get("max_completion_tokens", 96)),
            token_limit_parameter=str(
                config.update_policy.get("token_limit_parameter", "max_completion_tokens")
            ),
            timeout_seconds=float(config.update_policy.get("timeout_seconds", 30.0)),
            input_cost_per_1m_tokens=float(config.update_policy.get("input_cost_per_1m_tokens", 0.0)),
            output_cost_per_1m_tokens=float(config.update_policy.get("output_cost_per_1m_tokens", 0.0)),
            max_estimated_cost_usd=(
                float(config.update_policy["max_estimated_cost_usd"])
                if "max_estimated_cost_usd" in config.update_policy
                else None
            ),
        )
    raise ValueError("unsupported social media update_policy type")
```

Add usage summary and audit records to metrics/replay after the loop, mirroring
the event runner:

```python
    usage_summary = getattr(policy, "usage_summary", None)
    if callable(usage_summary):
        metrics["llm_usage"] = usage_summary()
    audit_records = getattr(policy, "audit_records", None)
    llm_decisions = audit_records() if callable(audit_records) else ()
```

- [ ] **Step 5: Run policy tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_social_media_policy.py tests/test_social_media_runner.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/society_simulation/social_media_policy.py src/society_simulation/social_media_runner.py tests/test_social_media_policy.py
git commit -m "feat: add llm policy for social media dynamics"
```

---

## Task 10: Sweep Compatibility and Planned Experiment Recipes

**Files:**
- Modify: `src/society_simulation/sweep_config.py`
- Test: `tests/test_sweep_config.py`
- Create: `experiments/instagram_feed_policy_sweep.json`
- Create: `docs/research/2026-06-22-instagram-social-experiment-recipes.md`

- [ ] **Step 1: Add sweep config test**

Append to `tests/test_sweep_config.py`:

```python
def test_social_media_sweep_accepts_nested_feed_policy_override(tmp_path) -> None:
    import json
    from society_simulation.sweep_config import expand_sweep, load_sweep_config

    path = tmp_path / "sweep.json"
    path.write_text(
        json.dumps(
            {
                "sweep_name": "instagram_feed_policy_sweep",
                "base_config": {
                    "experiment_name": "instagram_social_dynamics",
                    "seed": 1,
                    "scenario_name": "sweep_base",
                    "ticks": 2,
                    "num_users": 6,
                    "historical_posts_per_user": 1,
                    "feed_size": 2,
                    "activation_probability": 1.0,
                    "topics": ["transit", "housing"],
                    "seed_generator": {
                        "type": "synthetic_profiles",
                        "mean_following": 2,
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
                    "update_policy": {"type": "mock_social", "response_style": "balanced"},
                    "memory_retrieval": {"enabled": False, "limit": 5},
                    "output_dir": "runs/ignored",
                },
                "factors": [
                    {"name": "seed", "path": "seed", "values": [1, 2]},
                    {
                        "name": "feed_policy",
                        "path": "feed_policy.type",
                        "values": ["chronological_following", "engagement_ranked"],
                    },
                ],
                "output_dir": "runs/sweeps/instagram_feed_policy_sweep",
            }
        ),
        encoding="utf-8",
    )

    sweep = load_sweep_config(path)

    assert sweep.sweep_name == "instagram_feed_policy_sweep"
    assert len(expand_sweep(sweep)) == 4
```

- [ ] **Step 2: Add social media experiment support to sweep config**

`sweep_config.py` already supports dotted `path` overrides through
`apply_path_override`. Add the new experiment type to validation.

Modify the imports in `src/society_simulation/sweep_config.py`:

```python
from society_simulation.config import (
    Config,
    EventDrivenOpinionConfig,
    ExperimentConfig,
    NetworkHerdingConfig,
)
from society_simulation.social_media_config import InstagramSocialDynamicsConfig
```

Add:

```python
_INSTAGRAM_SOCIAL_DYNAMICS_KEYS = {
    "experiment_name",
    "seed",
    "scenario_name",
    "ticks",
    "num_users",
    "historical_posts_per_user",
    "feed_size",
    "activation_probability",
    "topics",
    "seed_generator",
    "feed_policy",
    "update_policy",
    "memory_retrieval",
    "output_dir",
}
```

Add a branch to `build_experiment_config` before the sequential fallback:

```python
    elif data.get("experiment_name") == "instagram_social_dynamics":
        _reject_unknown_keys(
            data,
            allowed=_INSTAGRAM_SOCIAL_DYNAMICS_KEYS,
            path="experiment config",
        )
        config = InstagramSocialDynamicsConfig.from_dict(data)
```

- [ ] **Step 3: Create feed-policy sweep config**

Create `experiments/instagram_feed_policy_sweep.json`:

```json
{
  "sweep_name": "instagram_feed_policy_sweep",
  "base_config": {
    "experiment_name": "instagram_social_dynamics",
    "seed": 20260622,
    "scenario_name": "instagram_feed_policy_sweep_base",
    "ticks": 4,
    "num_users": 12,
    "historical_posts_per_user": 2,
    "feed_size": 4,
    "activation_probability": 1.0,
    "topics": ["transit", "housing", "local_business", "schools"],
    "seed_generator": {
      "type": "synthetic_profiles",
      "mean_following": 4,
      "homophily_weight": 0.55,
      "popularity_weight": 0.25,
      "random_tie_probability": 0.1,
      "mutual_follow_probability": 0.4
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
      "explore_fraction": 0.33
    },
    "update_policy": {
      "type": "mock_social",
      "response_style": "endorsement_sensitive"
    },
    "memory_retrieval": {
      "enabled": true,
      "limit": 5
    },
    "output_dir": "runs/ignored-by-sweep"
  },
  "factors": [
    {
      "name": "seed",
      "path": "seed",
      "values": [20260622, 20260623]
    },
    {
      "name": "feed_policy",
      "path": "feed_policy.type",
      "values": [
        "chronological_following",
        "engagement_ranked",
        "interest_homophily",
        "bridging",
        "no_feed_control"
      ]
    }
  ],
  "output_dir": "runs/sweeps/instagram_feed_policy_sweep"
}
```

- [ ] **Step 4: Create experiment recipes doc**

Create `docs/research/2026-06-22-instagram-social-experiment-recipes.md`:

```markdown
# Instagram-Like Social Experiment Recipes

Date: 2026-06-22

## Recipe 1: Mock Feed-Policy Sweep

Command:

```bash
./.venv/bin/python -m society_simulation sweep experiments/instagram_feed_policy_sweep.json
```

Purpose: Compare chronological, engagement-ranked, interest-homophily,
bridging, and no-feed controls under identical agents and content.

Primary metrics: exposure diversity, action counts, follow edge delta,
final stance variance, DM count, like count.

## Recipe 2: Tiny Paid LLM Pilot

Purpose: Confirm the platform prompt/action contract works with a real model
without claiming real-world validity.

Configuration changes from the mock run:

- `num_users`: 6
- `ticks`: 2
- `feed_size`: 3
- `update_policy.type`: `llm`
- `update_policy.max_estimated_cost_usd`: 3.0
- `update_policy.max_completion_tokens`: 96

Stop conditions: parse failure rate above 20%, estimated cost above cap, or
unexpected action explosion.
```

- [ ] **Step 5: Run sweep-related tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sweep_config.py tests/test_sweep_runner.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/society_simulation/sweep_config.py tests/test_sweep_config.py experiments/instagram_feed_policy_sweep.json docs/research/2026-06-22-instagram-social-experiment-recipes.md
git commit -m "feat: add instagram social experiment sweep recipes"
```

---

## Final Verification

- [ ] **Step 1: Run targeted test suite**

```bash
./.venv/bin/python -m pytest \
  tests/test_social_media_models.py \
  tests/test_social_media_config.py \
  tests/test_social_media_seed.py \
  tests/test_social_media_feed.py \
  tests/test_social_media_policy.py \
  tests/test_social_media_runner.py \
  tests/test_social_media_metrics.py \
  tests/test_social_media_replay.py \
  tests/test_cli.py \
  tests/test_sweep_config.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

```bash
./.venv/bin/python -m pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run mock experiment**

```bash
./.venv/bin/python -m society_simulation run experiments/instagram_social_lite_mock.json
```

Expected output includes:

```text
experiment=instagram_social_dynamics
feed_impression_count=
action_count=
action_counts=
final_follow_edge_count=
final_stance_mean=
output_dir=runs/instagram_social_lite_mock
```

- [ ] **Step 4: Inspect replay artifacts**

```bash
test -f runs/instagram_social_lite_mock/config.json
test -f runs/instagram_social_lite_mock/users.jsonl
test -f runs/instagram_social_lite_mock/posts.jsonl
test -f runs/instagram_social_lite_mock/follow_edges_initial.jsonl
test -f runs/instagram_social_lite_mock/follow_edges_final.jsonl
test -f runs/instagram_social_lite_mock/feed_impressions.jsonl
test -f runs/instagram_social_lite_mock/actions.jsonl
test -f runs/instagram_social_lite_mock/dm_messages.jsonl
test -f runs/instagram_social_lite_mock/user_states.jsonl
test -f runs/instagram_social_lite_mock/metrics.json
```

Expected: all commands exit 0.

- [ ] **Step 5: Check for secrets and formatting issues**

```bash
rg -n "OPENAI_API_KEY=|Authorization:|Bearer [A-Za-z0-9._-]{20,}|sk-[A-Za-z0-9]{20,}" src tests experiments docs/research docs/superpowers/specs || true
git diff --check
```

Expected: no real secrets and no whitespace errors.

---

## First Experiments After Implementation

Run these before any large paid LLM experiment:

1. `instagram_social_lite_mock.json`
   - Purpose: verify platform state, action traces, metrics, and replay.
   - Cost: zero LLM cost.

2. `instagram_feed_policy_sweep.json`
   - Purpose: compare feed algorithms under deterministic mock policy.
   - Cost: zero LLM cost.

3. Tiny paid LLM contract pilot
   - Suggested shape: 6 users, 2 ticks, feed size 3, max completion 96, cost cap
     3 USD.
   - Purpose: verify that real LLM actions parse cleanly and produce plausible
     platform actions without telling agents they are part of an experiment.
   - Not a publishable result by itself.

4. Visible endorsement cascade
   - Same seed post under different visible initial like counts.
   - Purpose: test low-cost endorsement and crowd response through normal feed
     affordances.

5. Public silence and private backchannel
   - Socially risky post plus DMs.
   - Purpose: measure whether disagreement moves to private messages.

The first paper-worthy direction is not "LLMs can click like." It is whether a
controlled platform model can produce measurable differences in exposure,
expression, graph rewiring, and private/public divergence under interventions
that resemble real social-media mechanisms.
