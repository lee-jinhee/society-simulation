import json
from pathlib import Path

import pytest

from society_simulation.config import load_config
from society_simulation.event_config import EventDrivenOpinionConfig


def valid_event_config(tmp_path: Path) -> dict[str, object]:
    return {
        "experiment_name": "event_driven_opinion_dynamics",
        "seed": 11,
        "scenario_name": "congestion_pricing",
        "days": 3,
        "agents": [
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
            },
            {
                "agent_id": "minho",
                "name": "Minho Lee",
                "age": 42,
                "occupation": "delivery driver",
                "household_context": "married with two children",
                "neighborhood": "west side",
                "core_values": ["work", "family budget"],
                "material_interests": ["fuel cost", "delivery time"],
                "political_trust": 0.25,
                "media_habits": ["local_news", "neighborhood_group_chat"],
                "communication_style": "direct",
                "susceptibilities": ["small business stories"],
                "initial_private_stance": -0.45,
                "initial_public_stance": -0.25,
                "initial_confidence": 0.55,
                "initial_salience": 0.7,
            },
        ],
        "relationships": [
            {
                "source_agent_id": "jisoo",
                "target_agent_id": "minho",
                "relationship_type": "neighbor",
                "trust": 0.6,
                "conversation_frequency": "medium",
                "conflict_sensitivity": 0.5,
                "channels": ["neighborhood_group_chat"],
            }
        ],
        "events": [
            {
                "event_id": "city_announcement",
                "day": 1,
                "title": "City announces congestion charge",
                "source": "City Hall",
                "source_type": "official",
                "content": "The city proposes a weekday downtown congestion charge.",
                "policy_stance": 0.35,
                "credibility": 0.8,
                "emotional_intensity": 0.2,
                "affected_interests": ["commute time", "public health"],
                "audience_filter": {"media_habits_any": ["local_news"]},
            }
        ],
        "channels": [{"channel_id": "neighborhood_group_chat", "type": "group_chat"}],
        "update_policy": {"type": "mock_persona", "response_style": "balanced"},
        "output_dir": str(tmp_path / "event-run"),
    }


def test_load_event_driven_config(tmp_path: Path) -> None:
    path = tmp_path / "event.json"
    path.write_text(json.dumps(valid_event_config(tmp_path)), encoding="utf-8")

    config = load_config(path)

    assert isinstance(config, EventDrivenOpinionConfig)
    assert config.experiment_name == "event_driven_opinion_dynamics"
    assert config.scenario_name == "congestion_pricing"
    assert len(config.agents) == 2
    assert config.update_policy["type"] == "mock_persona"


def test_event_config_to_dict_round_trips(tmp_path: Path) -> None:
    config = EventDrivenOpinionConfig.from_dict(valid_event_config(tmp_path))
    payload = config.to_dict()

    round_tripped = EventDrivenOpinionConfig.from_dict(payload)

    assert round_tripped.to_dict() == payload


def test_event_config_freezes_update_policy_after_construction(tmp_path: Path) -> None:
    config = EventDrivenOpinionConfig.from_dict(valid_event_config(tmp_path))

    with pytest.raises(TypeError):
        config.update_policy["response_style"] = "silent"


def test_event_config_freezes_channel_mappings_after_construction(tmp_path: Path) -> None:
    config = EventDrivenOpinionConfig.from_dict(valid_event_config(tmp_path))

    with pytest.raises(TypeError):
        config.channels[0]["type"] = "direct_message"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("channels", 0, "metadata"), {"local_news"}),
    ],
)
def test_event_config_rejects_unsupported_free_form_values(
    tmp_path: Path,
    path: tuple[object, ...],
    value: object,
) -> None:
    data = valid_event_config(tmp_path)
    target = data
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValueError, match="must contain only JSON-compatible values"):
        EventDrivenOpinionConfig.from_dict(data)


@pytest.mark.parametrize(
    "path",
    [
        ("channels", 0),
        ("update_policy",),
    ],
)
def test_event_config_rejects_non_string_free_form_dict_keys(
    tmp_path: Path,
    path: tuple[object, ...],
) -> None:
    data = valid_event_config(tmp_path)
    target = data
    for key in path:
        target = target[key]  # type: ignore[index,assignment]
    target[1] = "bad"  # type: ignore[index]

    with pytest.raises(ValueError, match="object keys must be strings"):
        EventDrivenOpinionConfig.from_dict(data)


def test_event_config_rejects_duplicate_agent_ids(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["agents"] = [data["agents"][0], data["agents"][0]]

    config = EventDrivenOpinionConfig.from_dict(data)

    with pytest.raises(ValueError, match="agent ids must be unique"):
        config.validate()


def test_event_config_rejects_duplicate_channel_ids(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["channels"] = [
        {"channel_id": "neighborhood_group_chat", "type": "group_chat"},
        {"channel_id": "neighborhood_group_chat", "type": "group_chat"},
    ]

    config = EventDrivenOpinionConfig.from_dict(data)

    with pytest.raises(ValueError, match="channel ids must be unique"):
        config.validate()


def test_event_config_rejects_relationship_to_missing_agent(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["relationships"][0]["target_agent_id"] = "missing"  # type: ignore[index]

    config = EventDrivenOpinionConfig.from_dict(data)

    with pytest.raises(ValueError, match="relationship target_agent_id is not in agents"):
        config.validate()


def test_event_config_rejects_relationship_to_missing_channel(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["relationships"][0]["channels"] = ["missing_channel"]  # type: ignore[index]

    config = EventDrivenOpinionConfig.from_dict(data)

    with pytest.raises(ValueError, match="relationship channel is not in channels"):
        config.validate()


def test_event_config_rejects_duplicate_event_ids(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["events"] = [data["events"][0], data["events"][0]]

    config = EventDrivenOpinionConfig.from_dict(data)

    with pytest.raises(ValueError, match="event ids must be unique"):
        config.validate()


def test_event_config_rejects_event_outside_day_range(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["events"][0]["day"] = 9  # type: ignore[index]

    config = EventDrivenOpinionConfig.from_dict(data)

    with pytest.raises(ValueError, match="event day must be between 1 and days"):
        config.validate()


def test_event_config_rejects_day_zero_event(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["events"][0]["day"] = 0  # type: ignore[index]

    config = EventDrivenOpinionConfig.from_dict(data)

    with pytest.raises(ValueError, match="event day must be between 1 and days"):
        config.validate()


@pytest.mark.parametrize(
    ("update_policy", "message"),
    [
        ({"type": "unsupported"}, "unsupported event update_policy type"),
        (
            {"type": "mock_persona", "response_style": "aggressive"},
            "unsupported mock_persona response_style",
        ),
    ],
)
def test_event_config_rejects_unsupported_update_policy(
    tmp_path: Path,
    update_policy: dict[str, object],
    message: str,
) -> None:
    data = valid_event_config(tmp_path)
    data["update_policy"] = update_policy

    config = EventDrivenOpinionConfig.from_dict(data)

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_event_config_rejects_empty_llm_model(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["update_policy"] = {"type": "llm", "model": ""}

    with pytest.raises(ValueError, match="llm.model must be a non-empty string"):
        EventDrivenOpinionConfig.from_dict(data)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("api_key", "secret-bearing update_policy key"),
        ("Authorization", "secret-bearing update_policy key"),
        ("api-key", "secret-bearing update_policy key"),
        ("headers", "secret-bearing update_policy key"),
    ],
)
def test_event_config_rejects_direct_secret_update_policy_keys_before_replay(
    tmp_path: Path,
    key: str,
    message: str,
) -> None:
    data = valid_event_config(tmp_path)
    data["update_policy"] = {"type": "llm", "model": "cheap-chat", key: "secret"}

    with pytest.raises(ValueError, match=message):
        EventDrivenOpinionConfig.from_dict(data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_completion_tokens", True, "update_policy.max_completion_tokens must be an integer"),
        ("temperature", "0.7", "update_policy.temperature must be a number"),
        ("unknown", 1, "unsupported update_policy key"),
    ],
)
def test_event_config_rejects_invalid_llm_update_policy_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    data = valid_event_config(tmp_path)
    update_policy: dict[str, object] = {"type": "llm", "model": "cheap-chat"}
    update_policy[field] = value
    data["update_policy"] = update_policy

    with pytest.raises(ValueError, match=message):
        EventDrivenOpinionConfig.from_dict(data)


def test_event_config_rejects_mock_persona_unknown_update_policy_key(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["update_policy"] = {"type": "mock_persona", "response_style": "balanced", "extra": 1}

    with pytest.raises(ValueError, match="unsupported update_policy key"):
        EventDrivenOpinionConfig.from_dict(data)


def test_event_config_accepts_typed_llm_update_policy_fields(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["update_policy"] = {
        "type": "llm",
        "provider": "openai_compatible",
        "model": "cheap-chat",
        "api_key_env": "EVENT_LLM_API_KEY",
        "base_url": "https://example.test/v1",
        "temperature": 0.7,
        "max_completion_tokens": 64,
        "token_limit_parameter": "max_completion_tokens",
        "timeout_seconds": 10.5,
        "input_cost_per_1m_tokens": 0.0,
        "output_cost_per_1m_tokens": 1.25,
        "max_estimated_cost_usd": 0.5,
    }

    config = EventDrivenOpinionConfig.from_dict(data)

    config.validate()
    assert config.update_policy["max_completion_tokens"] == 64


def test_event_config_to_dict_is_json_serializable(tmp_path: Path) -> None:
    config = EventDrivenOpinionConfig.from_dict(valid_event_config(tmp_path))

    json.dumps(config.to_dict(), allow_nan=False)
