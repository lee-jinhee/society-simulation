import json
from inspect import signature

import pytest

from society_simulation.event_models import EventAgentProfile, EventAgentState, EventExposure
from society_simulation.event_policy import (
    MockPersonaPolicy,
    OpenAICompatiblePersonaPolicy,
    parse_event_decision_content,
)
from society_simulation.llm_policy import _urllib_json_transport


def profile() -> EventAgentProfile:
    return EventAgentProfile.from_dict(
        {
            "agent_id": "jisoo",
            "name": "Jisoo Park",
            "age": 37,
            "occupation": "hospital nurse",
            "household_context": "single parent",
            "neighborhood": "east side",
            "core_values": ["fairness", "public health"],
            "material_interests": ["commute time"],
            "political_trust": 0.45,
            "media_habits": ["local_news", "neighborhood_group_chat"],
            "communication_style": "careful",
            "susceptibilities": ["coworker stories"],
            "initial_private_stance": -0.1,
            "initial_public_stance": 0.0,
            "initial_confidence": 0.35,
            "initial_salience": 0.45,
        }
    )


def state() -> EventAgentState:
    return profile().initial_state(day=0)


def exposure() -> EventExposure:
    return EventExposure(
        day=1,
        agent_id="jisoo",
        source_type="event",
        source_id="city_announcement",
        channel="news_feed",
        content="City Hall says congestion pricing could reduce commute delays and pollution.",
    )


def llm_decision_content() -> str:
    return json.dumps(
        {
            "private_stance": 0.1,
            "public_stance": 0.0,
            "confidence": 0.5,
            "salience": 0.6,
            "emotion": "conflicted",
            "private_reasoning": "The announcement is plausible but costly.",
            "messages": [],
            "memory_update": "Jisoo remains conflicted.",
        }
    )


def test_parse_event_decision_content_requires_complete_json() -> None:
    decision = parse_event_decision_content(
        json.dumps(
            {
                "private_stance": 0.2,
                "public_stance": 0.1,
                "confidence": 0.6,
                "salience": 0.7,
                "emotion": "conflicted",
                "private_reasoning": "The policy helps traffic but adds costs.",
                "messages": [
                    {
                        "channel": "neighborhood_group_chat",
                        "recipient": None,
                        "text": "I can see both sides here.",
                    }
                ],
                "memory_update": "Jisoo is more conflicted after the city announcement.",
            }
        )
    )

    assert decision.state.private_stance == 0.2
    assert decision.messages[0].text == "I can see both sides here."


def test_parse_event_decision_content_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="event llm response content must be JSON"):
        parse_event_decision_content("{")


def test_mock_persona_policy_returns_audited_decision() -> None:
    policy = MockPersonaPolicy(response_style="balanced")

    decision = policy.decide(profile(), state(), (exposure(),), day=1)

    assert decision.state.agent_id == "jisoo"
    assert decision.state.day == 1
    assert decision.state.salience >= state().salience
    assert decision.messages
    usage = policy.usage_summary()
    assert usage["provider"] == "mock"
    assert usage["calls"] == 1
    assert policy.audit_records()[0]["agent_id"] == "jisoo"
    assert "Stay in character" in policy.audit_records()[0]["prompt"]


def test_mock_persona_policy_silent_style_posts_no_messages() -> None:
    policy = MockPersonaPolicy(response_style="silent")

    decision = policy.decide(profile(), state(), (exposure(),), day=1)

    assert decision.messages == ()


def test_mock_persona_policy_reactive_style_posts_message() -> None:
    policy = MockPersonaPolicy(response_style="reactive")

    decision = policy.decide(profile(), state(), (exposure(),), day=1)

    assert decision.state.private_stance > state().private_stance
    assert decision.messages


def test_openai_compatible_persona_policy_mirrors_llm_constructor_defaults() -> None:
    parameters = signature(OpenAICompatiblePersonaPolicy).parameters

    assert parameters["max_completion_tokens"].default == 32
    assert parameters["transport"].default is _urllib_json_transport


def test_openai_compatible_persona_policy_sends_human_role_prompt_without_experiment_language() -> None:
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
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "private_stance": 0.1,
                                "public_stance": 0.0,
                                "confidence": 0.5,
                                "salience": 0.6,
                                "emotion": "conflicted",
                                "private_reasoning": "The announcement is plausible but costly.",
                                "messages": [],
                                "memory_update": "Jisoo remains conflicted.",
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40},
        }

    policy = OpenAICompatiblePersonaPolicy(
        model="cheap-chat",
        api_key="secret-key",
        base_url="https://example.test/v1",
        input_cost_per_1m_tokens=1.0,
        output_cost_per_1m_tokens=2.0,
        transport=transport,
    )

    decision = policy.decide(profile(), state(), (exposure(),), day=1)

    assert decision.state.private_stance == 0.1
    payload = captured["payload"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert isinstance(messages, list)
    prompt_text = json.dumps(messages)
    assert "Stay in character" in prompt_text
    assert "social network experiment" not in prompt_text
    assert "secret-key" not in json.dumps(policy.audit_records(), sort_keys=True)


def test_openai_compatible_persona_policy_redacts_secret_keys_and_auth_patterns() -> None:
    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        del url, headers, payload, timeout_seconds
        return {
            "choices": [{"message": {"content": llm_decision_content()}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "safe_note": "keep this audit detail",
            "secret-key": "debug",
            "lowercase_auth": "authorization: bearer sk-other",
            "basic_auth": "Authorization: Basic abc",
            "debug_api-key_hint": "redact this",
        }

    policy = OpenAICompatiblePersonaPolicy(
        model="cheap-chat",
        api_key="secret-key",
        base_url="https://example.test/v1",
        transport=transport,
    )

    policy.decide(profile(), state(), (exposure(),), day=1)

    audit_json = json.dumps(policy.audit_records(), sort_keys=True)
    assert "keep this audit detail" in audit_json
    assert "secret-key" not in audit_json
    assert "authorization" not in audit_json.lower()
    assert "bearer" not in audit_json.lower()
    assert "basic abc" not in audit_json.lower()
    assert "api-key" not in audit_json.lower()


def test_openai_compatible_persona_policy_preflight_includes_worst_case_output_cost() -> None:
    called = False

    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        nonlocal called
        del url, headers, payload, timeout_seconds
        called = True
        return {"choices": [{"message": {"content": llm_decision_content()}}]}

    policy = OpenAICompatiblePersonaPolicy(
        model="cheap-chat",
        api_key="secret-key",
        base_url="https://example.test/v1",
        max_completion_tokens=32,
        output_cost_per_1m_tokens=1_000_000.0,
        max_estimated_cost_usd=31.0,
        transport=transport,
    )

    with pytest.raises(ValueError, match="llm estimated cost cap exceeded"):
        policy.decide(profile(), state(), (exposure(),), day=1)

    assert called is False


def test_openai_compatible_persona_policy_audits_paid_response_before_post_call_cap_error() -> None:
    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        del url, headers, payload, timeout_seconds
        return {
            "choices": [{"message": {"content": llm_decision_content()}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }

    policy = OpenAICompatiblePersonaPolicy(
        model="cheap-chat",
        api_key="secret-key",
        base_url="https://example.test/v1",
        max_completion_tokens=1,
        output_cost_per_1m_tokens=1.0,
        max_estimated_cost_usd=0.0000015,
        transport=transport,
    )

    with pytest.raises(ValueError, match="llm estimated cost cap exceeded"):
        policy.decide(profile(), state(), (exposure(),), day=1)

    records = policy.audit_records()
    assert len(records) == 1
    assert records[0]["completion_tokens"] == 2


def test_openai_compatible_persona_policy_redacts_echoed_secrets_from_audit() -> None:
    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        del url, payload, timeout_seconds
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "private_stance": 0.1,
                                "public_stance": 0.0,
                                "confidence": 0.5,
                                "salience": 0.6,
                                "emotion": "conflicted",
                                "private_reasoning": "The announcement is plausible but costly.",
                                "messages": [],
                                "memory_update": "Jisoo remains conflicted.",
                            }
                        )
                    }
                }
            ],
            "request_headers": headers,
            "metadata": {
                "api_key": "secret-key",
                "debug": "Authorization: Bearer secret-key",
            },
        }

    policy = OpenAICompatiblePersonaPolicy(
        model="cheap-chat",
        api_key="secret-key",
        base_url="https://example.test/v1",
        transport=transport,
    )

    policy.decide(profile(), state(), (exposure(),), day=1)

    audit_json = json.dumps(policy.audit_records(), sort_keys=True)
    assert "secret-key" not in audit_json
    assert "Authorization" not in audit_json
    assert "Bearer" not in audit_json


def test_openai_compatible_persona_policy_redacts_api_key_echo_values_from_audit() -> None:
    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        del url, headers, payload, timeout_seconds
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "private_stance": 0.1,
                                "public_stance": 0.0,
                                "confidence": 0.5,
                                "salience": 0.6,
                                "emotion": "conflicted",
                                "private_reasoning": "The announcement is plausible but costly.",
                                "messages": [],
                                "memory_update": "Jisoo remains conflicted.",
                            }
                        )
                    }
                }
            ],
            "debug_echo": "secret-key",
        }

    policy = OpenAICompatiblePersonaPolicy(
        model="cheap-chat",
        api_key="secret-key",
        base_url="https://example.test/v1",
        transport=transport,
    )

    policy.decide(profile(), state(), (exposure(),), day=1)

    assert "secret-key" not in json.dumps(policy.audit_records(), sort_keys=True)


def test_openai_compatible_persona_policy_redacts_parsed_output_secrets_from_audit() -> None:
    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        del url, headers, payload, timeout_seconds
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "private_stance": 0.1,
                                "public_stance": 0.0,
                                "confidence": 0.5,
                                "salience": 0.6,
                                "emotion": "conflicted",
                                "private_reasoning": (
                                    "Authorization: Bearer secret-key shaped my view."
                                ),
                                "messages": [
                                    {
                                        "channel": "neighborhood_group_chat",
                                        "recipient": None,
                                        "text": "api_key secret-key should not be stored.",
                                    }
                                ],
                                "memory_update": "Remembered api_key secret-key from the note.",
                            }
                        )
                    }
                }
            ],
        }

    policy = OpenAICompatiblePersonaPolicy(
        model="cheap-chat",
        api_key="secret-key",
        base_url="https://example.test/v1",
        transport=transport,
    )

    policy.decide(profile(), state(), (exposure(),), day=1)

    audit_json = json.dumps(policy.audit_records(), sort_keys=True)
    assert "secret-key" not in audit_json
    assert "Authorization" not in audit_json
    assert "Bearer" not in audit_json
    assert "api_key" not in audit_json


def test_openai_compatible_persona_policy_redacts_prompt_secrets_from_audit() -> None:
    secret_exposure = EventExposure(
        day=1,
        agent_id="jisoo",
        source_type="event",
        source_id="debug_echo",
        channel="news_feed",
        content="Authorization: Bearer secret-key and api_key should not persist.",
    )

    def transport(
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        del url, headers, payload, timeout_seconds
        return {"choices": [{"message": {"content": llm_decision_content()}}]}

    policy = OpenAICompatiblePersonaPolicy(
        model="cheap-chat",
        api_key="secret-key",
        base_url="https://example.test/v1",
        transport=transport,
    )

    policy.decide(profile(), state(), (secret_exposure,), day=1)

    audit_json = json.dumps(policy.audit_records(), sort_keys=True)
    assert "secret-key" not in audit_json
    assert "Authorization" not in audit_json
    assert "Bearer" not in audit_json
    assert "api_key" not in audit_json
