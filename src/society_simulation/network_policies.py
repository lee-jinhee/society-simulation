from __future__ import annotations

from statistics import mean

from society_simulation.config import NetworkUpdatePolicyConfig
from society_simulation.models import Action
from society_simulation.network_models import NetworkDecision, NetworkObservation
from society_simulation.policies import confidence_from_belief


class MajorityRulePolicy:
    name = "majority_rule"

    def decide(self, observation: NetworkObservation) -> NetworkDecision:
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
        if not 0.5 <= adoption_threshold <= 1.0:
            raise ValueError("adoption_threshold must be between 0.5 and 1.0")
        self.adoption_threshold = adoption_threshold

    def decide(self, observation: NetworkObservation) -> NetworkDecision:
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

        if a_fraction >= self.adoption_threshold:
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
        if not 0.0 <= self_weight <= 1.0:
            raise ValueError("self_weight must be between 0 and 1")
        self.self_weight = self_weight

    def decide(self, observation: NetworkObservation) -> NetworkDecision:
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


NetworkUpdatePolicy = MajorityRulePolicy | ThresholdPolicy | DeGrootPolicy


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
    raise ValueError("unsupported network update_policy type")
