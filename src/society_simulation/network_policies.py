from __future__ import annotations

from math import isfinite
import os
from statistics import mean

from society_simulation.config import NetworkUpdatePolicyConfig
from society_simulation.llm_policy import MockLLMPolicy, OpenAICompatibleLLMPolicy
from society_simulation.models import Action
from society_simulation.network_models import NetworkDecision, NetworkObservation
from society_simulation.policies import confidence_from_belief


def _validate_probability(
    value: float,
    field_name: str,
    min_value: float,
    max_value: float,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{field_name} must be between {min_value} and {max_value}")
    if not min_value <= value <= max_value:
        raise ValueError(f"{field_name} must be between {min_value} and {max_value}")


def _validate_action(action: str) -> None:
    if action not in ("A", "B"):
        raise ValueError("action must be A or B")


def _validate_observation(observation: NetworkObservation) -> None:
    if not (
        len(observation.observed_neighbor_ids)
        == len(observation.observed_neighbor_actions)
        == len(observation.observed_neighbor_beliefs)
    ):
        raise ValueError("neighbor observation fields must have equal length")

    _validate_action(observation.current_action)
    _validate_probability(
        observation.current_belief_probability,
        "belief_probability",
        0.0,
        1.0,
    )

    for action in observation.observed_neighbor_actions:
        _validate_action(action)
    for belief in observation.observed_neighbor_beliefs:
        _validate_probability(
            belief,
            "belief_probability",
            0.0,
            1.0,
        )


class MajorityRulePolicy:
    name = "majority_rule"

    def decide(self, observation: NetworkObservation) -> NetworkDecision:
        _validate_observation(observation)
        if not observation.observed_neighbor_actions:
            belief = observation.current_belief_probability
            return NetworkDecision(
                belief_probability=belief,
                confidence=confidence_from_belief(belief),
                action=observation.current_action,
            )

        a_fraction = observation.observed_neighbor_actions.count("A") / len(
            observation.observed_neighbor_actions
        )
        if a_fraction > 0.5:
            action: Action = "A"
        elif a_fraction < 0.5:
            action = "B"
        else:
            action = observation.current_action
        return NetworkDecision(
            belief_probability=a_fraction,
            confidence=confidence_from_belief(a_fraction),
            action=action,
        )


class ThresholdPolicy:
    name = "threshold"

    def __init__(self, adoption_threshold: float) -> None:
        _validate_probability(
            adoption_threshold,
            "adoption_threshold",
            0.5,
            1.0,
        )
        self.adoption_threshold = adoption_threshold

    def decide(self, observation: NetworkObservation) -> NetworkDecision:
        _validate_observation(observation)
        if not observation.observed_neighbor_actions:
            belief = observation.current_belief_probability
            return NetworkDecision(
                belief_probability=belief,
                confidence=confidence_from_belief(belief),
                action=observation.current_action,
            )

        a_fraction = observation.observed_neighbor_actions.count("A") / len(
            observation.observed_neighbor_actions
        )
        b_fraction = 1.0 - a_fraction

        if (
            a_fraction >= self.adoption_threshold
            and b_fraction >= self.adoption_threshold
        ):
            action = observation.current_action
        elif a_fraction >= self.adoption_threshold:
            action: Action = "A"
        elif b_fraction >= self.adoption_threshold:
            action = "B"
        else:
            action = observation.current_action

        return NetworkDecision(
            belief_probability=a_fraction,
            confidence=confidence_from_belief(a_fraction),
            action=action,
        )


class DeGrootPolicy:
    name = "degroot"

    def __init__(self, self_weight: float) -> None:
        _validate_probability(self_weight, "self_weight", 0.0, 1.0)
        self.self_weight = self_weight

    def decide(self, observation: NetworkObservation) -> NetworkDecision:
        _validate_observation(observation)
        if observation.observed_neighbor_beliefs:
            neighbor_mean = mean(observation.observed_neighbor_beliefs)
        else:
            neighbor_mean = observation.current_belief_probability

        belief = (
            self.self_weight * observation.current_belief_probability
            + (1.0 - self.self_weight) * neighbor_mean
        )
        action: Action = "A" if belief >= 0.5 else "B"

        return NetworkDecision(
            belief_probability=belief,
            confidence=confidence_from_belief(belief),
            action=action,
        )


NetworkUpdatePolicy = (
    MajorityRulePolicy | ThresholdPolicy | DeGrootPolicy | MockLLMPolicy | OpenAICompatibleLLMPolicy
)


def build_network_update_policy(config: NetworkUpdatePolicyConfig) -> NetworkUpdatePolicy:
    if config.type == "majority_rule":
        return MajorityRulePolicy()
    if config.type == "threshold":
        if config.adoption_threshold is None:
            raise ValueError("adoption_threshold must be between 0.5 and 1.0")
        return ThresholdPolicy(adoption_threshold=config.adoption_threshold)
    if config.type == "degroot":
        if config.self_weight is None:
            raise ValueError("self_weight must be between 0 and 1")
        return DeGrootPolicy(self_weight=config.self_weight)
    if config.type == "mock_llm":
        return MockLLMPolicy(
            provider=config.provider or "mock",
            model=config.model,
            response_style=config.response_style or "neighbor_majority",
            input_cost_per_1m_tokens=config.input_cost_per_1m_tokens or 0.0,
            output_cost_per_1m_tokens=config.output_cost_per_1m_tokens or 0.0,
        )
    if config.type == "llm":
        if config.model is None:
            raise ValueError("model must be a non-empty string")
        api_key_env = config.api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} is required for llm provider")
        return OpenAICompatibleLLMPolicy(
            provider=config.provider or "openai_compatible",
            model=config.model,
            api_key=api_key,
            base_url=config.base_url or "https://api.openai.com/v1",
            temperature=config.temperature if config.temperature is not None else 0.0,
            max_completion_tokens=(
                config.max_completion_tokens if config.max_completion_tokens is not None else 32
            ),
            token_limit_parameter=config.token_limit_parameter or "max_completion_tokens",
            timeout_seconds=config.timeout_seconds if config.timeout_seconds is not None else 30.0,
            input_cost_per_1m_tokens=config.input_cost_per_1m_tokens or 0.0,
            output_cost_per_1m_tokens=config.output_cost_per_1m_tokens or 0.0,
            max_estimated_cost_usd=config.max_estimated_cost_usd,
        )
    raise ValueError("unsupported network update_policy type")
