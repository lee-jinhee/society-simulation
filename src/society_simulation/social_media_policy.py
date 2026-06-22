from __future__ import annotations

from collections.abc import Sequence
import copy
import json
import time
from typing import Protocol

from society_simulation.llm_policy import (
    JSONTransport,
    LLMPricing,
    LLMUsage,
    OpenAICompatibleClient,
    _urllib_json_transport,
    estimate_tokens,
)
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


def build_social_media_prompt(
    *,
    profile: SocialMediaUserProfile,
    state: SocialMediaUserState,
    feed: Sequence[FeedItem],
    recent_memories: Sequence[str],
) -> str:
    feed_lines = [
        f"- rank={item.rank} post_id={item.post_id} author_id={item.author_id} "
        f"score={item.score:.2f} source={item.source} reason={item.reason}"
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
        transport: JSONTransport = _urllib_json_transport,
    ) -> None:
        if provider != "openai_compatible":
            raise ValueError("unsupported llm provider")
        if not model:
            raise ValueError("model must be a non-empty string")
        if max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
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
        self._client = OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )

    def decide(
        self,
        *,
        profile: SocialMediaUserProfile,
        state: SocialMediaUserState,
        feed: Sequence[FeedItem],
        tick: int,
    ) -> PlatformAction:
        messages = [
            {
                "role": "user",
                "content": build_social_media_prompt(
                    profile=profile,
                    state=state,
                    feed=feed,
                    recent_memories=(),
                ),
            }
        ]
        prompt_text = _messages_text(messages)
        prompt_tokens_estimate = estimate_tokens(prompt_text)
        self._raise_if_cost_cap_exceeded(
            self.usage.input_cost_usd
            + self.usage.output_cost_usd
            + self.pricing.input_cost(prompt_tokens_estimate)
        )
        started_at = time.perf_counter()
        response = self._client.create_chat_completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            token_limit_parameter=self.token_limit_parameter,
        )
        content = _extract_content(response)
        action = parse_social_action_content(content, tick=tick, user_id=profile.user_id)
        prompt_tokens, completion_tokens = _response_usage_tokens(
            response,
            default_prompt_tokens=prompt_tokens_estimate,
            default_completion_tokens=estimate_tokens(content),
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        self.usage.record(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            pricing=self.pricing,
        )
        self._raise_if_cost_cap_exceeded(self.usage.input_cost_usd + self.usage.output_cost_usd)
        input_cost_usd = self.pricing.input_cost(prompt_tokens)
        output_cost_usd = self.pricing.output_cost(completion_tokens)
        self._audit_records.append(
            {
                "tick": tick,
                "user_id": profile.user_id,
                "provider": self.provider,
                "model": self.model,
                "policy_type": self.name,
                "prompt": prompt_text,
                "messages": copy.deepcopy(messages),
                "raw_response": _redact_response(response),
                "action": action.to_dict(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "input_cost_usd": input_cost_usd,
                "output_cost_usd": output_cost_usd,
                "total_cost_usd": input_cost_usd + output_cost_usd,
                "latency_ms": latency_ms,
            }
        )
        return action

    def usage_summary(self) -> dict[str, object]:
        return self.usage.summary(provider=self.provider, model=self.model)

    def audit_records(self) -> tuple[dict[str, object], ...]:
        return tuple(copy.deepcopy(record) for record in self._audit_records)

    def _raise_if_cost_cap_exceeded(self, estimated_cost_usd: float) -> None:
        if self.max_estimated_cost_usd is None:
            return
        if estimated_cost_usd > self.max_estimated_cost_usd:
            raise ValueError("social media llm cost cap exceeded")


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


def _messages_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{message['role']}:{message['content']}" for message in messages)


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


def _response_usage_tokens(
    response: dict[str, object],
    *,
    default_prompt_tokens: int,
    default_completion_tokens: int,
) -> tuple[int, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return default_prompt_tokens, default_completion_tokens
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < 0:
        prompt_tokens = default_prompt_tokens
    if (
        isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens < 0
    ):
        completion_tokens = default_completion_tokens
    return prompt_tokens, completion_tokens


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
