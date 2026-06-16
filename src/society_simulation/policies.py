from __future__ import annotations

from society_simulation.models import Action, Decision, Observation


def confidence_from_belief(belief_probability: float) -> float:
    return abs(belief_probability - 0.5) * 2.0


def _signal_probability(signal: Action, true_state: Action, signal_accuracy: float) -> float:
    if signal == true_state:
        return signal_accuracy
    return 1.0 - signal_accuracy


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


class BayesianCascadePolicy:
    name = "bayesian_cascade"

    def __init__(self, signal_accuracy: float, prior_probability: float) -> None:
        if not 0.5 <= signal_accuracy <= 1.0:
            raise ValueError("signal_accuracy must be between 0.5 and 1.0")
        if not 0.0 < prior_probability < 1.0:
            raise ValueError("prior_probability must be greater than 0 and less than 1")
        self.signal_accuracy = signal_accuracy
        self.prior_probability = prior_probability

    def decide(self, observation: Observation) -> Decision:
        belief = self._posterior_probability(
            tuple(observation.observed_actions),
            observation.private_signal,
        )
        action: Action = "A" if belief >= 0.5 else "B"
        return Decision(
            belief_probability=belief,
            confidence=confidence_from_belief(belief),
            action=action,
        )

    def _posterior_probability(
        self,
        observed_actions: tuple[Action, ...],
        private_signal: Action,
    ) -> float:
        likelihood_history_a, likelihood_history_b = self._history_likelihoods(observed_actions)
        return self._posterior_from_likelihoods(
            likelihood_history_a,
            likelihood_history_b,
            private_signal,
        )

    def _posterior_from_likelihoods(
        self,
        likelihood_history_a: float,
        likelihood_history_b: float,
        private_signal: Action,
    ) -> float:
        likelihood_signal_a = _signal_probability(private_signal, "A", self.signal_accuracy)
        likelihood_signal_b = _signal_probability(private_signal, "B", self.signal_accuracy)

        numerator = self.prior_probability * likelihood_history_a * likelihood_signal_a
        denominator = numerator + (
            (1.0 - self.prior_probability) * likelihood_history_b * likelihood_signal_b
        )
        if denominator == 0.0:
            # Out-of-model public histories fall back to the private-signal posterior.
            numerator = self.prior_probability * likelihood_signal_a
            denominator = numerator + ((1.0 - self.prior_probability) * likelihood_signal_b)
        return numerator / denominator

    def _history_likelihoods(self, observed_actions: tuple[Action, ...]) -> tuple[float, float]:
        likelihood_history_a = 1.0
        likelihood_history_b = 1.0

        for public_action in observed_actions:
            matching_probability_a = 0.0
            matching_probability_b = 0.0

            for private_signal in ("A", "B"):
                predicted = self._action_from_likelihoods(
                    likelihood_history_a,
                    likelihood_history_b,
                    private_signal,
                )
                if predicted == public_action:
                    matching_probability_a += _signal_probability(
                        private_signal,
                        "A",
                        self.signal_accuracy,
                    )
                    matching_probability_b += _signal_probability(
                        private_signal,
                        "B",
                        self.signal_accuracy,
                    )

            likelihood_history_a *= matching_probability_a
            likelihood_history_b *= matching_probability_b

        return likelihood_history_a, likelihood_history_b

    def _action_from_likelihoods(
        self,
        likelihood_history_a: float,
        likelihood_history_b: float,
        private_signal: Action,
    ) -> Action:
        belief = self._posterior_from_likelihoods(
            likelihood_history_a,
            likelihood_history_b,
            private_signal,
        )
        return "A" if belief >= 0.5 else "B"


class SimpleHeuristicPolicy:
    name = "simple_heuristic"

    def decide(self, observation: Observation) -> Decision:
        score = 1 if observation.private_signal == "A" else -1
        score += observation.observed_actions.count("A")
        score -= observation.observed_actions.count("B")

        if score > 0:
            action: Action = "A"
        elif score < 0:
            action = "B"
        else:
            action = observation.private_signal

        scale = len(observation.observed_actions) + 1
        belief = _clamp_probability(0.5 + 0.5 * (score / scale))
        return Decision(
            belief_probability=belief,
            confidence=confidence_from_belief(belief),
            action=action,
        )


def build_update_policy(
    name: str,
    signal_accuracy: float,
    prior_probability: float,
) -> BayesianCascadePolicy | SimpleHeuristicPolicy:
    if name == "bayesian_cascade":
        return BayesianCascadePolicy(
            signal_accuracy=signal_accuracy,
            prior_probability=prior_probability,
        )
    if name == "simple_heuristic":
        return SimpleHeuristicPolicy()
    raise ValueError("unsupported update_policy")
