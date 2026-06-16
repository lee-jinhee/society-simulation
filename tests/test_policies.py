import pytest

from society_simulation.models import Observation
from society_simulation.policies import (
    BayesianCascadePolicy,
    SimpleHeuristicPolicy,
    build_update_policy,
)


def test_bayesian_policy_follows_private_signal_without_history() -> None:
    policy = BayesianCascadePolicy(signal_accuracy=0.7, prior_probability=0.5)

    decision_a = policy.decide(Observation(agent_id=0, private_signal="A", observed_actions=()))
    decision_b = policy.decide(Observation(agent_id=1, private_signal="B", observed_actions=()))

    assert decision_a.action == "A"
    assert decision_a.belief_probability == pytest.approx(0.7)
    assert decision_a.confidence == pytest.approx(0.4)
    assert decision_b.action == "B"
    assert decision_b.belief_probability == pytest.approx(0.3)


def test_bayesian_policy_uses_public_action_likelihood_not_hidden_signals() -> None:
    policy = BayesianCascadePolicy(signal_accuracy=0.7, prior_probability=0.5)

    decision = policy.decide(
        Observation(agent_id=2, private_signal="B", observed_actions=("A", "A"))
    )

    assert decision.action == "A"
    assert decision.belief_probability == pytest.approx(0.5)
    assert decision.confidence == pytest.approx(0.0)


def test_bayesian_policy_can_follow_private_signal_after_split_history() -> None:
    policy = BayesianCascadePolicy(signal_accuracy=0.7, prior_probability=0.5)

    decision = policy.decide(
        Observation(agent_id=2, private_signal="A", observed_actions=("A", "B"))
    )

    assert decision.action == "A"
    assert decision.belief_probability > 0.5


def test_bayesian_policy_handles_long_public_history_without_recursion_error() -> None:
    policy = BayesianCascadePolicy(signal_accuracy=0.7, prior_probability=0.5)

    decision = policy.decide(
        Observation(agent_id=1200, private_signal="B", observed_actions=("A",) * 1200)
    )

    assert decision.action == "A"
    assert 0.0 <= decision.belief_probability <= 1.0


def test_simple_heuristic_combines_private_signal_and_majority_actions() -> None:
    policy = SimpleHeuristicPolicy()

    decision = policy.decide(
        Observation(agent_id=3, private_signal="B", observed_actions=("A", "A", "A"))
    )

    assert decision.action == "A"
    assert decision.belief_probability > 0.5
    assert 0.0 <= decision.confidence <= 1.0


def test_simple_heuristic_breaks_ties_toward_private_signal() -> None:
    policy = SimpleHeuristicPolicy()

    decision = policy.decide(
        Observation(agent_id=2, private_signal="B", observed_actions=("A",))
    )

    assert decision.action == "B"
    assert decision.belief_probability == pytest.approx(0.5)


def test_build_update_policy_creates_bayesian_policy() -> None:
    policy = build_update_policy(
        "bayesian_cascade",
        signal_accuracy=0.7,
        prior_probability=0.5,
    )

    assert isinstance(policy, BayesianCascadePolicy)


def test_build_update_policy_creates_simple_heuristic_policy() -> None:
    policy = build_update_policy(
        "simple_heuristic",
        signal_accuracy=0.7,
        prior_probability=0.5,
    )

    assert isinstance(policy, SimpleHeuristicPolicy)


def test_build_update_policy_rejects_unknown_policy_name() -> None:
    with pytest.raises(ValueError, match="unsupported update_policy"):
        build_update_policy("unknown", signal_accuracy=0.7, prior_probability=0.5)
