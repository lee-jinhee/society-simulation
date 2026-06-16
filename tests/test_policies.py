import pytest

from society_simulation.models import Observation
from society_simulation.policies import BayesianCascadePolicy, SimpleHeuristicPolicy


def test_bayesian_policy_follows_private_signal_without_history() -> None:
    policy = BayesianCascadePolicy(signal_accuracy=0.7, prior_probability=0.5)

    decision_a = policy.decide(Observation(agent_id=0, private_signal="A", observed_actions=[]))
    decision_b = policy.decide(Observation(agent_id=1, private_signal="B", observed_actions=[]))

    assert decision_a.action == "A"
    assert decision_a.belief_probability == pytest.approx(0.7)
    assert decision_a.confidence == pytest.approx(0.4)
    assert decision_b.action == "B"
    assert decision_b.belief_probability == pytest.approx(0.3)


def test_bayesian_policy_uses_public_action_likelihood_not_hidden_signals() -> None:
    policy = BayesianCascadePolicy(signal_accuracy=0.7, prior_probability=0.5)

    decision = policy.decide(
        Observation(agent_id=2, private_signal="B", observed_actions=["A", "A"])
    )

    assert decision.action == "A"
    assert decision.belief_probability == pytest.approx(0.5)
    assert decision.confidence == pytest.approx(0.0)


def test_bayesian_policy_can_follow_private_signal_after_split_history() -> None:
    policy = BayesianCascadePolicy(signal_accuracy=0.7, prior_probability=0.5)

    decision = policy.decide(
        Observation(agent_id=2, private_signal="A", observed_actions=["A", "B"])
    )

    assert decision.action == "A"
    assert decision.belief_probability > 0.5


def test_simple_heuristic_combines_private_signal_and_majority_actions() -> None:
    policy = SimpleHeuristicPolicy()

    decision = policy.decide(
        Observation(agent_id=3, private_signal="B", observed_actions=["A", "A", "A"])
    )

    assert decision.action == "A"
    assert decision.belief_probability > 0.5
    assert 0.0 <= decision.confidence <= 1.0


def test_simple_heuristic_breaks_ties_toward_private_signal() -> None:
    policy = SimpleHeuristicPolicy()

    decision = policy.decide(
        Observation(agent_id=2, private_signal="B", observed_actions=["A"])
    )

    assert decision.action == "B"
    assert decision.belief_probability == pytest.approx(0.5)
