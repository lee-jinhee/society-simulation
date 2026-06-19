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


def test_event_config_rejects_duplicate_agent_ids(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["agents"] = [data["agents"][0], data["agents"][0]]

    config = EventDrivenOpinionConfig.from_dict(data)

    with pytest.raises(ValueError, match="agent ids must be unique"):
        config.validate()


def test_event_config_rejects_relationship_to_missing_agent(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["relationships"][0]["target_agent_id"] = "missing"  # type: ignore[index]

    config = EventDrivenOpinionConfig.from_dict(data)

    with pytest.raises(ValueError, match="relationship target_agent_id is not in agents"):
        config.validate()


def test_event_config_rejects_event_outside_day_range(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["events"][0]["day"] = 9  # type: ignore[index]

    config = EventDrivenOpinionConfig.from_dict(data)

    with pytest.raises(ValueError, match="event day must be between 0 and days"):
        config.validate()
