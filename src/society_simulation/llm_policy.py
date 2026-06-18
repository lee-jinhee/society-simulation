from __future__ import annotations

from dataclasses import dataclass
import json
from math import ceil, isfinite
from typing import Literal, cast

from society_simulation.models import Action
from society_simulation.network_models import NetworkDecision, NetworkObservation
from society_simulation.policies import confidence_from_belief

MockResponseStyle = Literal["neighbor_majority", "current", "contrarian"]

MOCK_RESPONSE_STYLES: tuple[str, ...] = ("neighbor_majority", "current", "contrarian")


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
