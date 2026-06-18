import pytest

from society_simulation.llm_policy import MockLLMPolicy, estimate_tokens
from society_simulation.network_models import NetworkObservation


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


def test_estimate_tokens_uses_deterministic_character_approximation() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_mock_llm_policy_decides_neighbor_majority_and_tracks_usage() -> None:
    policy = MockLLMPolicy(response_style="neighbor_majority")

    decision = policy.decide(observation(("A", "A", "B"), (1.0, 1.0, 0.0)))

    assert decision.action == "A"
    assert decision.belief_probability == pytest.approx(2 / 3)
    usage = policy.usage_summary()
    assert usage["provider"] == "mock"
    assert usage["calls"] == 1
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_cost_usd"] == 0.0


def test_mock_llm_policy_uses_current_response_style() -> None:
    policy = MockLLMPolicy(response_style="current")

    decision = policy.decide(
        observation(("A", "A", "A"), (1.0, 1.0, 1.0), current_action="B", current_belief=0.2)
    )

    assert decision.action == "B"
    assert decision.belief_probability == pytest.approx(0.2)


def test_mock_llm_policy_uses_contrarian_response_style() -> None:
    policy = MockLLMPolicy(response_style="contrarian")

    decision = policy.decide(observation(("A", "A", "B"), (1.0, 1.0, 0.0)))

    assert decision.action == "B"
    assert decision.belief_probability == pytest.approx(1 / 3)


def test_mock_llm_policy_estimates_cost() -> None:
    policy = MockLLMPolicy(
        response_style="current",
        input_cost_per_1m_tokens=1.0,
        output_cost_per_1m_tokens=2.0,
    )

    policy.decide(observation((), (), current_action="B", current_belief=0.2))

    usage = policy.usage_summary()
    assert usage["input_cost_usd"] > 0
    assert usage["output_cost_usd"] > 0
    assert usage["total_cost_usd"] == pytest.approx(
        usage["input_cost_usd"] + usage["output_cost_usd"]
    )


def test_mock_llm_policy_rejects_unsupported_provider() -> None:
    with pytest.raises(ValueError, match="unsupported llm provider"):
        MockLLMPolicy(provider="openai")


def test_mock_llm_policy_rejects_unsupported_response_style() -> None:
    with pytest.raises(ValueError, match="unsupported mock llm response_style"):
        MockLLMPolicy(response_style="random")
