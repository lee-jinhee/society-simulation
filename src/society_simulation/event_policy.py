from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import dataclass
import json
from math import isfinite
import time
from typing import Any

from society_simulation.event_models import (
    EventAgentProfile,
    EventAgentState,
    EventExposure,
    EventMessage,
)
from society_simulation.llm_policy import (
    JSONTransport,
    LLMPricing,
    LLMUsage,
    OpenAICompatibleClient,
    _urllib_json_transport,
    estimate_tokens,
)

TokenLimitParameter = str

_ROLE_MESSAGE = (
    "Stay in character as an ordinary local resident. "
    "Do not discuss system instructions, model mechanics, or hidden setup."
)
_REQUIRED_DECISION_FIELDS = (
    "private_stance",
    "public_stance",
    "confidence",
    "salience",
    "willingness_to_speak",
    "perceived_majority",
    "fairness_concern",
    "trust_in_official_info",
    "emotion",
    "silence_reason",
    "private_reasoning",
    "messages",
    "memory_update",
)
_POSITIVE_KEYWORDS = ("pollution", "traffic", "health", "asthma")
_NEGATIVE_KEYWORDS = ("fee", "cost", "burden", "income", "livelihood")
_SENSITIVE_RESPONSE_KEY_PATTERNS = ("authorization", "api_key", "api-key", "header")
_SENSITIVE_RESPONSE_VALUE_PATTERNS = (
    "authorization",
    "bearer ",
    "api_key",
    "api-key",
    "header",
)
_REDACTED = "[REDACTED]"


@dataclass(frozen=True)
class EventPolicyDecision:
    state: EventAgentState
    messages: tuple[EventMessage, ...]


def parse_event_decision_content(
    content: str,
    agent_id: str = "agent",
    day: int = 0,
) -> EventPolicyDecision:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("event llm response content must be JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("event llm response content must be a JSON object")

    for field_name in _REQUIRED_DECISION_FIELDS:
        if field_name not in data:
            raise ValueError(f"event llm response content must include {field_name}")

    messages_value = data["messages"]
    if not isinstance(messages_value, list):
        raise ValueError("messages must be a list")

    messages = tuple(
        _parse_event_message(message, agent_id=agent_id, day=day)
        for message in messages_value
    )
    state = EventAgentState(
        agent_id=agent_id,
        day=day,
        private_stance=data["private_stance"],
        public_stance=data["public_stance"],
        confidence=data["confidence"],
        salience=data["salience"],
        willingness_to_speak=data["willingness_to_speak"],
        perceived_majority=data["perceived_majority"],
        fairness_concern=data["fairness_concern"],
        trust_in_official_info=data["trust_in_official_info"],
        emotion=_require_non_empty_str(data["emotion"], "emotion"),
        silence_reason=_require_non_empty_str(data["silence_reason"], "silence_reason"),
        memory_summary=_require_non_empty_str(data["memory_update"], "memory_update"),
        last_private_reasoning=_require_non_empty_str(
            data["private_reasoning"],
            "private_reasoning",
        ),
    )
    return EventPolicyDecision(state=state, messages=messages)


class MockPersonaPolicy:
    name = "mock_persona"

    def __init__(
        self,
        *,
        response_style: str = "balanced",
        provider: str = "mock",
        model: str | None = None,
        configured_channels: Sequence[str] = (),
        input_cost_per_1m_tokens: float = 0.0,
        output_cost_per_1m_tokens: float = 0.0,
    ) -> None:
        if provider != "mock":
            raise ValueError("unsupported persona provider")
        if response_style not in ("balanced", "silent", "reactive"):
            raise ValueError("unsupported mock persona response_style")

        self.response_style = response_style
        self.provider = provider
        self.model = model or f"mock-persona-{response_style}"
        self.configured_channels = tuple(configured_channels)
        self.pricing = LLMPricing(
            input_cost_per_1m_tokens=input_cost_per_1m_tokens,
            output_cost_per_1m_tokens=output_cost_per_1m_tokens,
        )
        self.usage = LLMUsage()
        self._audit_records: list[dict[str, Any]] = []

    def decide(
        self,
        profile: EventAgentProfile,
        current_state: EventAgentState,
        exposures: Sequence[EventExposure],
        *,
        day: int,
        retrieved_memories: Sequence[object] = (),
    ) -> EventPolicyDecision:
        prompt = _event_prompt(
            profile,
            current_state,
            exposures,
            retrieved_memories=retrieved_memories,
        )
        response_content = self._response_content(profile, current_state, exposures)
        decision = parse_event_decision_content(
            response_content,
            agent_id=profile.agent_id,
            day=day,
        )
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(response_content)
        self.usage.record(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            pricing=self.pricing,
        )
        self._audit_records.append(
            _event_audit_record(
                agent_id=profile.agent_id,
                day=day,
                provider=self.provider,
                model=self.model,
                policy_type=self.name,
                prompt=prompt,
                raw_response={"content": response_content},
                decision=decision,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                pricing=self.pricing,
                latency_ms=0.0,
            )
        )
        return decision

    def usage_summary(self) -> dict[str, object]:
        return self.usage.summary(provider=self.provider, model=self.model)

    def audit_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(record) for record in self._audit_records)

    def _response_content(
        self,
        profile: EventAgentProfile,
        current_state: EventAgentState,
        exposures: Sequence[EventExposure],
    ) -> str:
        delta = _keyword_delta(exposures)
        private_stance = _clamp_stance(current_state.private_stance + delta)
        public_stance = _clamp_stance(current_state.public_stance + (delta / 2))
        salience = _clamp_probability(
            current_state.salience + (0.10 if exposures else 0.0),
        )
        confidence = _clamp_probability(current_state.confidence + (0.05 if exposures else 0.0))
        messages: list[dict[str, str | None]] = []
        if self.response_style != "silent":
            channel = _mock_message_channel(profile, self.configured_channels)
            messages.append(
                {
                    "channel": channel,
                    "recipient": None,
                    "text": "I am weighing the benefits and costs before deciding where I land.",
                }
            )
        willingness_to_speak = 0.7 if messages else 0.2
        fairness_concern = 0.6 if delta < 0 else 0.35
        silence_reason = "not_silent" if messages else "Not enough new information to post."

        return json.dumps(
            {
                "private_stance": private_stance,
                "public_stance": public_stance,
                "confidence": confidence,
                "salience": salience,
                "willingness_to_speak": willingness_to_speak,
                "perceived_majority": public_stance,
                "fairness_concern": fairness_concern,
                "trust_in_official_info": profile.political_trust,
                "emotion": "conflicted" if exposures else current_state.emotion,
                "silence_reason": silence_reason,
                "private_reasoning": _mock_private_reasoning(delta, exposures),
                "messages": messages,
                "memory_update": _mock_memory_update(profile, exposures),
            },
            sort_keys=True,
            separators=(",", ":"),
        )


class OpenAICompatiblePersonaPolicy:
    name = "event_persona"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        provider: str = "openai_compatible",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_completion_tokens: int = 32,
        token_limit_parameter: TokenLimitParameter = "max_completion_tokens",
        timeout_seconds: float = 30.0,
        input_cost_per_1m_tokens: float = 0.0,
        output_cost_per_1m_tokens: float = 0.0,
        max_estimated_cost_usd: float | None = None,
        configured_channels: Sequence[str] = (),
        known_agent_ids: Sequence[str] = (),
        transport: JSONTransport = _urllib_json_transport,
    ) -> None:
        if provider != "openai_compatible":
            raise ValueError("unsupported persona provider")
        if not model:
            raise ValueError("model must be a non-empty string")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be between 0 and 2")
        if not isfinite(temperature) or not 0.0 <= float(temperature) <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if token_limit_parameter not in ("max_completion_tokens", "max_tokens"):
            raise ValueError("token_limit_parameter must be max_completion_tokens or max_tokens")

        self.provider = provider
        self.model = model
        self.temperature = float(temperature)
        self.max_completion_tokens = _require_positive_int(
            max_completion_tokens,
            "max_completion_tokens",
        )
        self.token_limit_parameter = token_limit_parameter
        self.configured_channels = _string_tuple(configured_channels, "configured_channels")
        self.known_agent_ids = _string_tuple(known_agent_ids, "known_agent_ids")
        self.pricing = LLMPricing(
            input_cost_per_1m_tokens=input_cost_per_1m_tokens,
            output_cost_per_1m_tokens=output_cost_per_1m_tokens,
        )
        self.max_estimated_cost_usd = (
            None
            if max_estimated_cost_usd is None
            else _require_non_negative_finite_number(
                max_estimated_cost_usd,
                "max_estimated_cost_usd",
            )
        )
        self.usage = LLMUsage()
        self._audit_records: list[dict[str, Any]] = []
        self._sensitive_audit_values = tuple(value for value in (api_key,) if value)
        self._client = OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )

    def decide(
        self,
        profile: EventAgentProfile,
        current_state: EventAgentState,
        exposures: Sequence[EventExposure],
        *,
        day: int,
        retrieved_memories: Sequence[object] = (),
    ) -> EventPolicyDecision:
        messages = [
            {"role": "system", "content": _ROLE_MESSAGE},
            {
                "role": "user",
                "content": _event_user_message(
                    profile,
                    current_state,
                    exposures,
                    configured_channels=self.configured_channels,
                    known_agent_ids=self.known_agent_ids,
                    retrieved_memories=retrieved_memories,
                ),
            },
        ]
        prompt = _messages_text(messages)
        prompt_tokens_estimate = estimate_tokens(prompt)
        self._raise_if_cost_cap_exceeded(
            self.usage.input_cost_usd
            + self.usage.output_cost_usd
            + self.pricing.input_cost(prompt_tokens_estimate)
            + self.pricing.output_cost(self.max_completion_tokens)
        )

        started_at = time.perf_counter()
        response = self._client.create_chat_completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            token_limit_parameter=self.token_limit_parameter,
        )
        try:
            content = _extract_choice_content(response)
            decision = parse_event_decision_content(content, agent_id=profile.agent_id, day=day)
        except ValueError as exc:
            completion_estimate = (
                estimate_tokens(content) if "content" in locals() else self.max_completion_tokens
            )
            prompt_tokens, completion_tokens = _response_usage_tokens(
                response,
                default_prompt_tokens=prompt_tokens_estimate,
                default_completion_tokens=completion_estimate,
            )
            latency_ms = (time.perf_counter() - started_at) * 1000
            self.usage.record(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                pricing=self.pricing,
            )
            self._audit_records.append(
                _event_error_audit_record(
                    agent_id=profile.agent_id,
                    day=day,
                    provider=self.provider,
                    model=self.model,
                    policy_type=self.name,
                    prompt=prompt,
                    raw_response=response,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    pricing=self.pricing,
                    latency_ms=latency_ms,
                    error=str(exc),
                    messages=messages,
                    sensitive_values=self._sensitive_audit_values,
                )
            )
            raise
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
        self._audit_records.append(
            _event_audit_record(
                agent_id=profile.agent_id,
                day=day,
                provider=self.provider,
                model=self.model,
                policy_type=self.name,
                prompt=prompt,
                raw_response=response,
                decision=decision,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                pricing=self.pricing,
                latency_ms=latency_ms,
                messages=messages,
                sensitive_values=self._sensitive_audit_values,
            )
        )
        self._raise_if_cost_cap_exceeded(self.usage.input_cost_usd + self.usage.output_cost_usd)
        return decision

    def usage_summary(self) -> dict[str, object]:
        return self.usage.summary(provider=self.provider, model=self.model)

    def audit_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(record) for record in self._audit_records)

    def _raise_if_cost_cap_exceeded(self, estimated_cost_usd: float) -> None:
        if self.max_estimated_cost_usd is None:
            return
        if estimated_cost_usd > self.max_estimated_cost_usd:
            raise ValueError("llm estimated cost cap exceeded")


def _parse_event_message(message: object, *, agent_id: str, day: int) -> EventMessage:
    if not isinstance(message, dict):
        raise ValueError("messages entries must be objects")
    for field_name in ("channel", "recipient", "text"):
        if field_name not in message:
            raise ValueError(f"message must include {field_name}")
    recipient = message["recipient"]
    if recipient is not None and not isinstance(recipient, str):
        raise ValueError("recipient must be a string or null")
    return EventMessage(
        day=day,
        sender_agent_id=agent_id,
        channel=_require_non_empty_str(message["channel"], "channel"),
        recipient_agent_id=recipient,
        text=_require_non_empty_str(message["text"], "text"),
    )


def _event_prompt(
    profile: EventAgentProfile,
    current_state: EventAgentState,
    exposures: Sequence[EventExposure],
    *,
    retrieved_memories: Sequence[object] = (),
) -> str:
    return _messages_text(
        [
            {"role": "system", "content": _ROLE_MESSAGE},
            {
                "role": "user",
                "content": _event_user_message(
                    profile,
                    current_state,
                    exposures,
                    retrieved_memories=retrieved_memories,
                ),
            },
        ]
    )


def _event_user_message(
    profile: EventAgentProfile,
    current_state: EventAgentState,
    exposures: Sequence[EventExposure],
    *,
    configured_channels: Sequence[str] = (),
    known_agent_ids: Sequence[str] = (),
    retrieved_memories: Sequence[object] = (),
) -> str:
    exposure_rows = [exposure.to_dict() for exposure in exposures]
    allowed_channels = _allowed_message_channels(profile, configured_channels)
    allowed_recipients = tuple(
        agent_id for agent_id in known_agent_ids if agent_id != profile.agent_id
    )
    memory_context = _memory_context(retrieved_memories)
    return (
        "Consider the new information as this local resident and decide how your "
        "private view, public posture, and willingness to speak change.\n"
        "Return only JSON with keys private_stance, public_stance, confidence, salience, "
        "willingness_to_speak, perceived_majority, fairness_concern, "
        "trust_in_official_info, emotion, silence_reason, private_reasoning, "
        "messages, and memory_update.\n"
        "willingness_to_speak, fairness_concern, and trust_in_official_info must be "
        "numbers from 0 to 1. perceived_majority must be a stance estimate from -1 to 1.\n"
        "silence_reason should briefly say why you did not post; use not_silent if you post.\n"
        "Keep private_reasoning under 280 characters and memory_update under 160 characters.\n"
        "messages must be a list of objects with channel, recipient, and text. "
        "At most one message; zero messages is allowed. Keep message text under 180 characters.\n"
        "For messages, channel must be one of allowed_channels; "
        "recipient must be null or one of allowed_recipients. "
        "Use null for group/channel posts. Do not use event ids or source ids as recipients.\n"
        f"allowed_channels={json.dumps(allowed_channels, sort_keys=True)}\n"
        f"allowed_recipients={json.dumps(allowed_recipients, sort_keys=True)}\n"
        f"{memory_context}"
        f"profile={json.dumps(profile.to_dict(), sort_keys=True)}\n"
        f"current_state={json.dumps(current_state.to_dict(), sort_keys=True)}\n"
        f"exposures={json.dumps(exposure_rows, sort_keys=True)}"
    )


def _memory_context(retrieved_memories: Sequence[object]) -> str:
    lines = _memory_context_lines(retrieved_memories)
    if not lines:
        return ""
    return "Things you currently remember:\n" + "\n".join(f"- {line}" for line in lines) + "\n"


def _memory_context_lines(retrieved_memories: Sequence[object]) -> tuple[str, ...]:
    lines: list[str] = []
    for item in retrieved_memories:
        memory = getattr(item, "memory", item)
        text = getattr(memory, "text", None)
        if isinstance(text, str) and text.strip():
            lines.append(text.strip())
    return tuple(lines)


def _event_audit_record(
    *,
    agent_id: str,
    day: int,
    provider: str,
    model: str,
    policy_type: str,
    prompt: str,
    raw_response: dict[str, object],
    decision: EventPolicyDecision,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: LLMPricing,
    latency_ms: float,
    messages: list[dict[str, str]] | None = None,
    sensitive_values: tuple[str, ...] = (),
) -> dict[str, Any]:
    input_cost_usd = pricing.input_cost(prompt_tokens)
    output_cost_usd = pricing.output_cost(completion_tokens)
    state = decision.state
    record: dict[str, Any] = {
        "agent_id": agent_id,
        "day": day,
        "provider": provider,
        "model": model,
        "policy_type": policy_type,
        "prompt": prompt,
        "raw_response": copy.deepcopy(raw_response),
        "parsed_private_stance": state.private_stance,
        "parsed_public_stance": state.public_stance,
        "parsed_confidence": state.confidence,
        "parsed_salience": state.salience,
        "parsed_willingness_to_speak": state.willingness_to_speak,
        "parsed_perceived_majority": state.perceived_majority,
        "parsed_fairness_concern": state.fairness_concern,
        "parsed_trust_in_official_info": state.trust_in_official_info,
        "parsed_emotion": state.emotion,
        "parsed_silence_reason": state.silence_reason,
        "parsed_memory_summary": state.memory_summary,
        "parsed_private_reasoning": state.last_private_reasoning,
        "parsed_messages": [message.to_dict() for message in decision.messages],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "input_cost_usd": input_cost_usd,
        "output_cost_usd": output_cost_usd,
        "total_cost_usd": input_cost_usd + output_cost_usd,
        "latency_ms": latency_ms,
    }
    if messages is not None:
        record["messages"] = copy.deepcopy(messages)
    sanitized_record = _sanitize_audit_value(record, sensitive_values=sensitive_values)
    if not isinstance(sanitized_record, dict):
        raise TypeError("sanitized audit record must be a dictionary")
    return sanitized_record


def _event_error_audit_record(
    *,
    agent_id: str,
    day: int,
    provider: str,
    model: str,
    policy_type: str,
    prompt: str,
    raw_response: dict[str, object],
    prompt_tokens: int,
    completion_tokens: int,
    pricing: LLMPricing,
    latency_ms: float,
    error: str,
    messages: list[dict[str, str]] | None = None,
    sensitive_values: tuple[str, ...] = (),
) -> dict[str, Any]:
    input_cost_usd = pricing.input_cost(prompt_tokens)
    output_cost_usd = pricing.output_cost(completion_tokens)
    record: dict[str, Any] = {
        "agent_id": agent_id,
        "day": day,
        "provider": provider,
        "model": model,
        "policy_type": policy_type,
        "prompt": prompt,
        "raw_response": copy.deepcopy(raw_response),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "input_cost_usd": input_cost_usd,
        "output_cost_usd": output_cost_usd,
        "total_cost_usd": input_cost_usd + output_cost_usd,
        "latency_ms": latency_ms,
        "error": error,
    }
    if messages is not None:
        record["messages"] = copy.deepcopy(messages)
    sanitized_record = _sanitize_audit_value(record, sensitive_values=sensitive_values)
    if not isinstance(sanitized_record, dict):
        raise TypeError("sanitized audit record must be a dictionary")
    return sanitized_record


def _sanitize_audit_value(
    value: object,
    *,
    sensitive_values: tuple[str, ...] = (),
) -> object:
    if isinstance(value, dict):
        sanitized: dict[object, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                sanitized[key] = _sanitize_audit_value(
                    item,
                    sensitive_values=sensitive_values,
                )
                continue
            if _is_sensitive_audit_string(key, sensitive_values=sensitive_values):
                continue
            sanitized[key] = _sanitize_audit_value(
                item,
                sensitive_values=sensitive_values,
            )
        return sanitized
    if isinstance(value, list):
        return [
            _sanitize_audit_value(item, sensitive_values=sensitive_values)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _sanitize_audit_value(item, sensitive_values=sensitive_values)
            for item in value
        )
    if isinstance(value, str):
        if _is_sensitive_audit_string(value, sensitive_values=sensitive_values):
            return _REDACTED
    return copy.deepcopy(value)


def _is_sensitive_audit_string(value: str, *, sensitive_values: tuple[str, ...]) -> bool:
    lowered = value.lower()
    if any(pattern in lowered for pattern in _SENSITIVE_RESPONSE_KEY_PATTERNS):
        return True
    if any(pattern in lowered for pattern in _SENSITIVE_RESPONSE_VALUE_PATTERNS):
        return True
    return any(secret and secret in value for secret in sensitive_values)


def _keyword_delta(exposures: Sequence[EventExposure]) -> float:
    if not exposures:
        return 0.0
    scores = []
    for exposure in exposures:
        text = exposure.content.lower()
        score = 0.0
        if any(keyword in text for keyword in _POSITIVE_KEYWORDS):
            score += 0.10
        if any(keyword in text for keyword in _NEGATIVE_KEYWORDS):
            score -= 0.10
        scores.append(score)
    return sum(scores) / len(scores)


def _mock_private_reasoning(delta: float, exposures: Sequence[EventExposure]) -> str:
    if not exposures:
        return "No new information changed the view."
    if delta > 0:
        return "The information points to public health or traffic benefits."
    if delta < 0:
        return "The information raises cost or livelihood concerns."
    return "The information gives reasons on both sides."


def _mock_memory_update(profile: EventAgentProfile, exposures: Sequence[EventExposure]) -> str:
    if not exposures:
        return f"{profile.name} has no new policy information to add."
    return f"{profile.name} considered new local policy information."


def _mock_message_channel(
    profile: EventAgentProfile,
    configured_channels: Sequence[str],
) -> str:
    for channel in profile.media_habits:
        if channel in configured_channels:
            return channel
    if configured_channels:
        return configured_channels[0]
    for channel in profile.media_habits:
        if "chat" in channel:
            return channel
    return "neighborhood_group_chat"


def _allowed_message_channels(
    profile: EventAgentProfile,
    configured_channels: Sequence[str],
) -> tuple[str, ...]:
    configured = tuple(configured_channels)
    preferred = tuple(channel for channel in profile.media_habits if channel in configured)
    if preferred:
        return preferred
    if configured:
        return configured
    return profile.media_habits


def _string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    return tuple(
        _require_non_empty_str(value, f"{field_name}[{index}]")
        for index, value in enumerate(values)
    )


def _messages_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{message['role']}:{message['content']}" for message in messages)


def _extract_choice_content(response: dict[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("llm response is missing choices[0].message.content")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("llm response is missing choices[0].message.content")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("llm response is missing choices[0].message.content")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("llm response is missing choices[0].message.content")
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


def _clamp_stance(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def _require_non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_non_negative_finite_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{field_name} must be a non-negative finite number")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number")
    return float(value)
