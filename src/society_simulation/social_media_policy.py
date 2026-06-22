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
