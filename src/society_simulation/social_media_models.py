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
FeedSource = Literal["following", "explore", "intervention", "sponsored"]

SUPPORTED_SOCIAL_ACTIONS: tuple[str, ...] = (
    "like_post",
    "follow_user",
    "unfollow_user",
    "send_dm",
    "create_post",
    "do_nothing",
)
SUPPORTED_FEED_SOURCES: tuple[str, ...] = (
    "following",
    "explore",
    "intervention",
    "sponsored",
)


def _tuple_to_list(data: dict[str, object]) -> dict[str, object]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in data.items()
    }


def _validate_probability(value: float, field: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")


def _validate_stance(value: float, field: str) -> None:
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"{field} must be between -1 and 1")


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

    def __post_init__(self) -> None:
        _validate_stance(self.initial_stance, "initial_stance")
        for field in (
            "activity_rate",
            "post_rate",
            "privacy_preference",
            "conformity",
            "skepticism",
            "conflict_tolerance",
            "status_weight",
        ):
            _validate_probability(getattr(self, field), field)

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

    def __post_init__(self) -> None:
        _validate_stance(self.stance, "stance")
        _validate_stance(self.perceived_majority, "perceived_majority")
        _validate_probability(self.confidence, "confidence")
        _validate_probability(self.salience, "salience")
        _validate_probability(self.social_fatigue, "social_fatigue")

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
    campaign_id: str | None = None
    is_ad: bool = False

    def __post_init__(self) -> None:
        _validate_stance(self.stance, "stance")
        if self.like_count < 0:
            raise ValueError("like_count must be non-negative")
        if self.reply_count < 0:
            raise ValueError("reply_count must be non-negative")

    def with_like(self) -> SocialMediaPost:
        return replace(self, like_count=self.like_count + 1)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FollowEdge:
    follower_id: int
    followed_id: int
    created_tick: int

    def __post_init__(self) -> None:
        if self.follower_id == self.followed_id:
            raise ValueError("follow edge cannot point to self")

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
    visible_like_count: int = 0
    topic: str | None = None
    text: str | None = None
    author_handle: str | None = None
    campaign_id: str | None = None
    is_sponsored: bool = False
    advertiser_id: int | None = None
    ad_seen_count: int = 0

    def __post_init__(self) -> None:
        if self.source not in SUPPORTED_FEED_SOURCES:
            raise ValueError("unsupported feed source")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.visible_like_count < 0:
            raise ValueError("visible_like_count must be non-negative")
        if self.ad_seen_count < 0:
            raise ValueError("ad_seen_count must be non-negative")

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

    def __post_init__(self) -> None:
        if self.action_type not in SUPPORTED_SOCIAL_ACTIONS:
            raise ValueError("unsupported social action")
        if self.stance is not None:
            _validate_stance(self.stance, "stance")

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

    def __post_init__(self) -> None:
        if self.sender_id == self.recipient_id:
            raise ValueError("direct message recipient must differ from sender")
        if self.stance is not None:
            _validate_stance(self.stance, "stance")

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
