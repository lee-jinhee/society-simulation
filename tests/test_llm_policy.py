import pytest

from society_simulation.llm_policy import (
    MockLLMPolicy,
    OpenAICompatibleLLMPolicy,
    estimate_tokens,
)
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


def test_openai_compatible_policy_sends_chat_completion_request_and_tracks_provider_usage() -> None:
    captured: dict[str, object] = {}

    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        captured["timeout_seconds"] = timeout_seconds
        return {
            "choices": [
                {"message": {"content": '{"action":"A","belief_probability":0.75}'}}
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }

    policy = OpenAICompatibleLLMPolicy(
        model="cheap-chat",
        api_key="test-key",
        base_url="https://example.test/v1",
        input_cost_per_1m_tokens=1.0,
        output_cost_per_1m_tokens=2.0,
        transport=transport,
    )

    decision = policy.decide(observation(("A", "B"), (1.0, 0.0)))

    assert decision.action == "A"
    assert decision.belief_probability == pytest.approx(0.75)
    assert captured["url"] == "https://example.test/v1/chat/completions"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "cheap-chat"
    assert payload["temperature"] == 0.0
    assert payload["max_completion_tokens"] == 32
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert captured["timeout_seconds"] == 30.0

    usage = policy.usage_summary()
    assert usage["provider"] == "openai_compatible"
    assert usage["model"] == "cheap-chat"
    assert usage["calls"] == 1
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 7
    assert usage["input_cost_usd"] == pytest.approx(11 / 1_000_000)
    assert usage["output_cost_usd"] == pytest.approx(14 / 1_000_000)
    assert usage["total_cost_usd"] == pytest.approx(25 / 1_000_000)


def test_openai_compatible_policy_can_use_legacy_max_tokens_parameter() -> None:
    captured: dict[str, object] = {}

    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        del url, headers, timeout_seconds
        captured["payload"] = payload
        return {
            "choices": [
                {"message": {"content": '{"action":"B","belief_probability":0.25}'}}
            ],
        }

    policy = OpenAICompatibleLLMPolicy(
        model="legacy-compatible",
        api_key="test-key",
        token_limit_parameter="max_tokens",
        max_completion_tokens=12,
        transport=transport,
    )

    decision = policy.decide(observation(("B",), (0.0,)))

    assert decision.action == "B"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["max_tokens"] == 12
    assert "max_completion_tokens" not in payload
    usage = policy.usage_summary()
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0


def test_openai_compatible_policy_rejects_missing_api_key() -> None:
    with pytest.raises(ValueError, match="api key is required"):
        OpenAICompatibleLLMPolicy(model="cheap-chat", api_key="")


def test_openai_compatible_policy_rejects_malformed_model_content() -> None:
    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        del url, headers, payload, timeout_seconds
        return {"choices": [{"message": {"content": "not json"}}]}

    policy = OpenAICompatibleLLMPolicy(
        model="cheap-chat",
        api_key="test-key",
        transport=transport,
    )

    with pytest.raises(ValueError, match="llm response content must be JSON"):
        policy.decide(observation(("A",), (1.0,)))


def test_openai_compatible_policy_rejects_missing_message_content() -> None:
    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        del url, headers, payload, timeout_seconds
        return {"choices": [{"message": {}}]}

    policy = OpenAICompatibleLLMPolicy(
        model="cheap-chat",
        api_key="test-key",
        transport=transport,
    )

    with pytest.raises(ValueError, match="llm response is missing choices"):
        policy.decide(observation(("A",), (1.0,)))


def test_openai_compatible_policy_enforces_prompt_side_cost_cap_before_call() -> None:
    calls = 0

    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        nonlocal calls
        del url, headers, payload, timeout_seconds
        calls += 1
        return {
            "choices": [
                {"message": {"content": '{"action":"A","belief_probability":0.75}'}}
            ],
        }

    policy = OpenAICompatibleLLMPolicy(
        model="cheap-chat",
        api_key="test-key",
        input_cost_per_1m_tokens=1.0,
        max_estimated_cost_usd=0.0,
        transport=transport,
    )

    with pytest.raises(ValueError, match="llm estimated cost cap exceeded"):
        policy.decide(observation(("A",), (1.0,)))
    assert calls == 0
