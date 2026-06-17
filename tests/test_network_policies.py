import pytest

from society_simulation.config import NetworkUpdatePolicyConfig
from society_simulation.network_models import NetworkObservation
from society_simulation.network_policies import (
    DeGrootPolicy,
    MajorityRulePolicy,
    ThresholdPolicy,
    build_network_update_policy,
)


def observation(
    actions: tuple[str, ...],
    beliefs: tuple[float, ...],
    current_action: str = "B",
    current_belief: float = 0.25,
) -> NetworkObservation:
    return NetworkObservation(
        agent_id=0,
        round_index=1,
        current_belief_probability=current_belief,
        current_action=current_action,  # type: ignore[arg-type]
        observed_neighbor_ids=tuple(range(len(actions))),
        observed_neighbor_actions=actions,  # type: ignore[arg-type]
        observed_neighbor_beliefs=beliefs,
    )


def test_majority_rule_chooses_neighbor_majority() -> None:
    policy = MajorityRulePolicy()

    decision = policy.decide(observation(("A", "A", "B"), (0.9, 0.8, 0.2)))

    assert decision.action == "A"
    assert decision.belief_probability == pytest.approx(2 / 3)


def test_majority_rule_tie_keeps_current_action() -> None:
    policy = MajorityRulePolicy()

    decision = policy.decide(observation(("A", "B"), (0.9, 0.1), current_action="B"))

    assert decision.action == "B"
    assert decision.belief_probability == pytest.approx(0.5)


def test_majority_rule_empty_neighbors_keep_current_action_and_belief() -> None:
    policy = MajorityRulePolicy()

    decision = policy.decide(observation((), (), current_action="A", current_belief=0.78))

    assert decision.action == "A"
    assert decision.belief_probability == pytest.approx(0.78)


def test_threshold_policy_switches_when_threshold_reached() -> None:
    policy = ThresholdPolicy(adoption_threshold=0.75)

    decision = policy.decide(observation(("A", "A", "A", "B"), (0.9, 0.8, 0.7, 0.1)))

    assert decision.action == "A"
    assert decision.belief_probability == pytest.approx(0.75)


def test_threshold_policy_keeps_current_action_when_no_side_reaches_threshold() -> None:
    policy = ThresholdPolicy(adoption_threshold=0.75)

    decision = policy.decide(observation(("A", "B", "B", "A"), (0.9, 0.1, 0.2, 0.8)))

    assert decision.action == "B"
    assert decision.belief_probability == pytest.approx(0.5)


def test_threshold_policy_switches_to_b_when_b_reaches_threshold() -> None:
    policy = ThresholdPolicy(adoption_threshold=0.8)

    decision = policy.decide(observation(("B", "B", "B", "A"), (0.1, 0.2, 0.3, 0.9)))

    assert decision.action == "B"
    assert decision.belief_probability == pytest.approx(0.25)


def test_threshold_policy_empty_neighbors_keep_current_action_and_belief() -> None:
    policy = ThresholdPolicy(adoption_threshold=0.6)

    decision = policy.decide(observation((), (), current_action="A", current_belief=0.4))

    assert decision.action == "A"
    assert decision.belief_probability == pytest.approx(0.4)


def test_degroot_policy_averages_current_and_neighbor_beliefs() -> None:
    policy = DeGrootPolicy(self_weight=0.25)

    decision = policy.decide(observation(("A", "B"), (0.8, 0.4), current_belief=0.2))

    assert decision.belief_probability == pytest.approx(0.25 * 0.2 + 0.75 * 0.6)
    assert decision.action == "A"


def test_degroot_policy_empty_neighbors_uses_current_belief() -> None:
    policy = DeGrootPolicy(self_weight=0.3)

    decision = policy.decide(observation((), (), current_action="B", current_belief=0.35))

    assert decision.action == "B"
    assert decision.belief_probability == pytest.approx(0.35)


def test_policy_factory_builds_supported_policies() -> None:
    assert isinstance(
        build_network_update_policy(NetworkUpdatePolicyConfig(type="majority_rule")),
        MajorityRulePolicy,
    )
    assert isinstance(
        build_network_update_policy(
            NetworkUpdatePolicyConfig(type="threshold", adoption_threshold=0.6)
        ),
        ThresholdPolicy,
    )
    assert isinstance(
        build_network_update_policy(NetworkUpdatePolicyConfig(type="degroot", self_weight=0.4)),
        DeGrootPolicy,
    )


def test_policy_factory_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="unsupported network update_policy type"):
        build_network_update_policy(NetworkUpdatePolicyConfig(type="unknown"))


def test_threshold_policy_rejects_bad_threshold() -> None:
    with pytest.raises(
        ValueError, match="adoption_threshold must be between 0.5 and 1.0"
    ):
        ThresholdPolicy(adoption_threshold=0.4)


def test_degroot_policy_rejects_bad_self_weight() -> None:
    with pytest.raises(ValueError, match="self_weight must be between 0 and 1"):
        DeGrootPolicy(self_weight=1.2)
