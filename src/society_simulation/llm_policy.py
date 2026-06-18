from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from math import ceil, isfinite
import urllib.error
import urllib.request
from typing import Literal, cast

from society_simulation.models import Action
from society_simulation.network_models import NetworkDecision, NetworkObservation
from society_simulation.policies import confidence_from_belief

MockResponseStyle = Literal["neighbor_majority", "current", "contrarian"]
TokenLimitParameter = Literal["max_completion_tokens", "max_tokens"]
JSONTransport = Callable[
    [str, dict[str, str], dict[str, object], float],
    dict[str, object],
]

MOCK_RESPONSE_STYLES: tuple[str, ...] = ("neighbor_majority", "current", "contrarian")
TOKEN_LIMIT_PARAMETERS: tuple[str, ...] = ("max_completion_tokens", "max_tokens")


def estimate_tokens(text: str) -> int:
    if text == "":
        return 0
    return ceil(len(text) / 4)


def _require_non_negative_finite_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{field_name} must be a non-negative finite number")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number")
    return float(value)


def _require_positive_finite_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{field_name} must be a positive finite number")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return float(value)


def _require_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _validate_probability(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{field_name} must be between 0 and 1")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return float(value)


def _validate_action(action: str) -> Action:
    if action not in ("A", "B"):
        raise ValueError("action must be A or B")
    return cast(Action, action)


def _validate_observation(observation: NetworkObservation) -> None:
    if not (
        len(observation.observed_neighbor_ids)
        == len(observation.observed_neighbor_actions)
        == len(observation.observed_neighbor_beliefs)
    ):
        raise ValueError("neighbor observation fields must have equal length")

    _validate_action(observation.current_action)
    _validate_probability(observation.current_belief_probability, "belief_probability")

    for action in observation.observed_neighbor_actions:
        _validate_action(action)
    for belief in observation.observed_neighbor_beliefs:
        _validate_probability(belief, "belief_probability")


@dataclass(frozen=True)
class LLMPricing:
    input_cost_per_1m_tokens: float = 0.0
    output_cost_per_1m_tokens: float = 0.0

    def __post_init__(self) -> None:
        _require_non_negative_finite_number(
            self.input_cost_per_1m_tokens,
            "input_cost_per_1m_tokens",
        )
        _require_non_negative_finite_number(
            self.output_cost_per_1m_tokens,
            "output_cost_per_1m_tokens",
        )

    def input_cost(self, tokens: int) -> float:
        return tokens * self.input_cost_per_1m_tokens / 1_000_000

    def output_cost(self, tokens: int) -> float:
        return tokens * self.output_cost_per_1m_tokens / 1_000_000


@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0

    def record(self, prompt_tokens: int, completion_tokens: int, pricing: LLMPricing) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.input_cost_usd += pricing.input_cost(prompt_tokens)
        self.output_cost_usd += pricing.output_cost(completion_tokens)

    def summary(self, *, provider: str, model: str) -> dict[str, object]:
        return {
            "provider": provider,
            "model": model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "input_cost_usd": self.input_cost_usd,
            "output_cost_usd": self.output_cost_usd,
            "total_cost_usd": self.input_cost_usd + self.output_cost_usd,
        }


@dataclass(frozen=True)
class MockLLMResponse:
    action: Action
    belief_probability: float
    response_text: str


class MockLLMProvider:
    def __init__(self, response_style: str = "neighbor_majority") -> None:
        if response_style not in MOCK_RESPONSE_STYLES:
            raise ValueError("unsupported mock llm response_style")
        self.response_style = response_style

    def complete(self, prompt: str, observation: NetworkObservation) -> MockLLMResponse:
        del prompt
        action, belief = self._decide(observation)
        response_text = json.dumps(
            {"action": action, "belief_probability": belief},
            sort_keys=True,
            separators=(",", ":"),
        )
        return MockLLMResponse(action=action, belief_probability=belief, response_text=response_text)

    def _decide(self, observation: NetworkObservation) -> tuple[Action, float]:
        if self.response_style == "current":
            return (
                _validate_action(observation.current_action),
                _validate_probability(observation.current_belief_probability, "belief_probability"),
            )

        if not observation.observed_neighbor_actions:
            return (
                _validate_action(observation.current_action),
                _validate_probability(observation.current_belief_probability, "belief_probability"),
            )

        a_fraction = observation.observed_neighbor_actions.count("A") / len(
            observation.observed_neighbor_actions
        )

        if self.response_style == "neighbor_majority":
            return self._neighbor_majority_decision(observation, a_fraction)
        if self.response_style == "contrarian":
            return self._contrarian_decision(observation, a_fraction)

        raise ValueError("unsupported mock llm response_style")

    @staticmethod
    def _neighbor_majority_decision(
        observation: NetworkObservation,
        a_fraction: float,
    ) -> tuple[Action, float]:
        if a_fraction > 0.5:
            return "A", a_fraction
        if a_fraction < 0.5:
            return "B", a_fraction
        return _validate_action(observation.current_action), a_fraction

    @staticmethod
    def _contrarian_decision(
        observation: NetworkObservation,
        a_fraction: float,
    ) -> tuple[Action, float]:
        if a_fraction > 0.5:
            return "B", 1.0 - a_fraction
        if a_fraction < 0.5:
            return "A", 1.0 - a_fraction
        return _validate_action(observation.current_action), a_fraction


class MockLLMPolicy:
    name = "mock_llm"

    def __init__(
        self,
        *,
        provider: str = "mock",
        model: str | None = None,
        response_style: str = "neighbor_majority",
        input_cost_per_1m_tokens: float = 0.0,
        output_cost_per_1m_tokens: float = 0.0,
    ) -> None:
        if provider != "mock":
            raise ValueError("unsupported llm provider")
        if response_style not in MOCK_RESPONSE_STYLES:
            raise ValueError("unsupported mock llm response_style")

        self.provider = provider
        self.model = model or f"mock-{response_style.replace('_', '-')}"
        self.response_style = response_style
        self.pricing = LLMPricing(
            input_cost_per_1m_tokens=input_cost_per_1m_tokens,
            output_cost_per_1m_tokens=output_cost_per_1m_tokens,
        )
        self.usage = LLMUsage()
        self._provider = MockLLMProvider(response_style=response_style)

    def decide(self, observation: NetworkObservation) -> NetworkDecision:
        _validate_observation(observation)
        prompt = self._prompt_from_observation(observation)
        response = self._provider.complete(prompt, observation)

        belief = _validate_probability(response.belief_probability, "belief_probability")
        action = _validate_action(response.action)
        self.usage.record(
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(response.response_text),
            pricing=self.pricing,
        )
        return NetworkDecision(
            belief_probability=belief,
            confidence=confidence_from_belief(belief),
            action=action,
        )

    def usage_summary(self) -> dict[str, object]:
        return self.usage.summary(provider=self.provider, model=self.model)

    def _prompt_from_observation(self, observation: NetworkObservation) -> str:
        neighbor_rows = [
            f"id={agent_id},action={action},belief={belief:.6f}"
            for agent_id, action, belief in zip(
                observation.observed_neighbor_ids,
                observation.observed_neighbor_actions,
                observation.observed_neighbor_beliefs,
                strict=True,
            )
        ]
        neighbors = ";".join(neighbor_rows) if neighbor_rows else "none"
        return (
            f"model={self.model}\n"
            f"response_style={self.response_style}\n"
            f"agent_id={observation.agent_id}\n"
            f"round_index={observation.round_index}\n"
            f"current_action={observation.current_action}\n"
            f"current_belief={observation.current_belief_probability:.6f}\n"
            f"neighbors={neighbors}\n"
            "Return JSON with action A or B and belief_probability in [0,1]."
        )


def _urllib_json_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
    timeout_seconds: float,
) -> dict[str, object]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"llm provider request failed with HTTP {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"llm provider request failed: {exc.reason}") from exc

    try:
        data = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise ValueError("llm provider returned malformed JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("llm provider response must be a JSON object")
    return data


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        transport: JSONTransport = _urllib_json_transport,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be a non-empty string")
        if not api_key:
            raise ValueError("api key is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = _require_positive_finite_number(
            timeout_seconds,
            "timeout_seconds",
        )
        self.transport = transport

    def create_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_completion_tokens: int,
        token_limit_parameter: str,
    ) -> dict[str, object]:
        if not model:
            raise ValueError("model must be a non-empty string")
        if token_limit_parameter not in TOKEN_LIMIT_PARAMETERS:
            raise ValueError("token_limit_parameter must be max_completion_tokens or max_tokens")

        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            token_limit_parameter: max_completion_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return self.transport(
            f"{self.base_url}/chat/completions",
            headers,
            payload,
            self.timeout_seconds,
        )


def _llm_system_message() -> str:
    return (
        "You are simulating one agent in a network herding experiment. "
        "Return only compact JSON with keys action and belief_probability. "
        "action must be A or B. belief_probability must be between 0 and 1."
    )


def _llm_user_message(observation: NetworkObservation) -> str:
    neighbor_rows = [
        f"id={agent_id},action={action},belief={belief:.6f}"
        for agent_id, action, belief in zip(
            observation.observed_neighbor_ids,
            observation.observed_neighbor_actions,
            observation.observed_neighbor_beliefs,
            strict=True,
        )
    ]
    neighbors = ";".join(neighbor_rows) if neighbor_rows else "none"
    return (
        f"agent_id={observation.agent_id}\n"
        f"round_index={observation.round_index}\n"
        f"current_action={observation.current_action}\n"
        f"current_belief={observation.current_belief_probability:.6f}\n"
        f"neighbors={neighbors}\n"
        "Decide the next action and belief_probability."
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


def _parse_llm_decision_content(content: str) -> tuple[Action, float]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("llm response content must be JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("llm response content must be a JSON object")
    if "action" not in data:
        raise ValueError("llm response content must include action")
    if "belief_probability" not in data:
        raise ValueError("llm response content must include belief_probability")
    action_value = data["action"]
    if not isinstance(action_value, str):
        raise ValueError("action must be A or B")
    belief_value = data["belief_probability"]
    if not isinstance(belief_value, (int, float)):
        raise ValueError("belief_probability must be between 0 and 1")
    return (
        _validate_action(action_value),
        _validate_probability(float(belief_value), "belief_probability"),
    )


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


class OpenAICompatibleLLMPolicy:
    name = "llm"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        provider: str = "openai_compatible",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_completion_tokens: int = 32,
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
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be between 0 and 2")
        if not isfinite(temperature) or not 0.0 <= float(temperature) <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if token_limit_parameter not in TOKEN_LIMIT_PARAMETERS:
            raise ValueError("token_limit_parameter must be max_completion_tokens or max_tokens")

        self.provider = provider
        self.model = model
        self.temperature = float(temperature)
        self.max_completion_tokens = _require_positive_int(
            max_completion_tokens,
            "max_completion_tokens",
        )
        self.token_limit_parameter = token_limit_parameter
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
        self._client = OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )

    def decide(self, observation: NetworkObservation) -> NetworkDecision:
        _validate_observation(observation)
        messages = [
            {"role": "system", "content": _llm_system_message()},
            {"role": "user", "content": _llm_user_message(observation)},
        ]
        prompt_tokens_estimate = estimate_tokens(_messages_text(messages))
        self._raise_if_cost_cap_exceeded(
            self.usage.input_cost_usd
            + self.usage.output_cost_usd
            + self.pricing.input_cost(prompt_tokens_estimate)
        )

        response = self._client.create_chat_completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            token_limit_parameter=self.token_limit_parameter,
        )
        content = _extract_choice_content(response)
        action, belief = _parse_llm_decision_content(content)
        prompt_tokens, completion_tokens = _response_usage_tokens(
            response,
            default_prompt_tokens=prompt_tokens_estimate,
            default_completion_tokens=estimate_tokens(content),
        )
        self.usage.record(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            pricing=self.pricing,
        )
        self._raise_if_cost_cap_exceeded(self.usage.input_cost_usd + self.usage.output_cost_usd)
        return NetworkDecision(
            belief_probability=belief,
            confidence=confidence_from_belief(belief),
            action=action,
        )

    def usage_summary(self) -> dict[str, object]:
        return self.usage.summary(provider=self.provider, model=self.model)

    def _raise_if_cost_cap_exceeded(self, estimated_cost_usd: float) -> None:
        if self.max_estimated_cost_usd is None:
            return
        if estimated_cost_usd > self.max_estimated_cost_usd:
            raise ValueError("llm estimated cost cap exceeded")
