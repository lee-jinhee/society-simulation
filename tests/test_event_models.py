import pytest

from society_simulation.event_models import (
    EventAgentProfile,
    EventAgentState,
    EventExposure,
    EventMessage,
    EventRelationship,
    OpinionEvent,
    validate_probability,
    validate_stance,
)


def test_validate_stance_accepts_minus_one_to_one() -> None:
    assert validate_stance(-1.0, "private_stance") == -1.0
    assert validate_stance(0, "private_stance") == 0.0
    assert validate_stance(1.0, "private_stance") == 1.0


@pytest.mark.parametrize(
    "value",
    [-1.1, 1.1, "0.3", True, float("nan"), float("inf"), float("-inf")],
)
def test_validate_stance_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="private_stance must be a number between -1 and 1"):
        validate_stance(value, "private_stance")


def test_validate_probability_accepts_zero_to_one() -> None:
    assert validate_probability(0, "confidence") == 0.0
    assert validate_probability(0.5, "confidence") == 0.5
    assert validate_probability(1.0, "confidence") == 1.0


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, "0.3", True, float("nan"), float("inf"), float("-inf")],
)
def test_validate_probability_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="confidence must be a number between 0 and 1"):
        validate_probability(value, "confidence")


def _valid_agent_state_kwargs() -> dict[str, object]:
    return {
        "agent_id": "jisoo",
        "day": 2,
        "private_stance": -0.25,
        "public_stance": 0.0,
        "confidence": 0.52,
        "salience": 0.77,
        "emotion": "worried",
        "memory_summary": "Jisoo is worried about commute costs.",
        "last_private_reasoning": "The taxi story felt concrete.",
    }


def test_agent_profile_from_dict_validates_required_fields() -> None:
    profile = EventAgentProfile.from_dict(
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

    assert profile.agent_id == "jisoo"
    initial_state = profile.initial_state(day=0)
    assert initial_state.private_stance == -0.1
    assert initial_state.emotion == "calm"
    assert initial_state.memory_summary
    assert initial_state.last_private_reasoning
    assert profile.to_dict()["name"] == "Jisoo Park"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("private_stance", 1.1, "private_stance must be a number between -1 and 1"),
        ("public_stance", -1.1, "public_stance must be a number between -1 and 1"),
        ("confidence", -0.1, "confidence must be a number between 0 and 1"),
        ("salience", float("inf"), "salience must be a number between 0 and 1"),
    ],
)
def test_agent_state_rejects_invalid_stance_and_probability_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    state_kwargs = _valid_agent_state_kwargs()
    state_kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        EventAgentState(**state_kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("agent_id", "agent_id must be a non-empty string"),
        ("emotion", "emotion must be a non-empty string"),
        ("memory_summary", "memory_summary must be a non-empty string"),
        ("last_private_reasoning", "last_private_reasoning must be a non-empty string"),
    ],
)
def test_agent_state_rejects_empty_strings(field: str, message: str) -> None:
    state_kwargs = _valid_agent_state_kwargs()
    state_kwargs[field] = ""

    with pytest.raises(ValueError, match=message):
        EventAgentState(**state_kwargs)  # type: ignore[arg-type]


def test_relationship_from_dict_round_trips() -> None:
    relationship = EventRelationship.from_dict(
        {
            "source_agent_id": "jisoo",
            "target_agent_id": "minho",
            "relationship_type": "coworker",
            "trust": 0.72,
            "conversation_frequency": "high",
            "conflict_sensitivity": 0.4,
            "channels": ["hospital_group_chat"],
        }
    )

    assert relationship.trust == 0.72
    assert relationship.channels == ("hospital_group_chat",)
    assert relationship.to_dict()["target_agent_id"] == "minho"


def test_event_from_dict_supports_audience_filter() -> None:
    event = OpinionEvent.from_dict(
        {
            "event_id": "city_announcement",
            "day": 1,
            "title": "City announces downtown congestion charge",
            "source": "City Hall",
            "source_type": "official",
            "content": "The city proposes a weekday downtown congestion charge.",
            "policy_stance": 0.35,
            "credibility": 0.8,
            "emotional_intensity": 0.2,
            "affected_interests": ["commute time", "public health"],
            "audience_filter": {"media_habits_any": ["local_news"]},
        }
    )

    assert event.day == 1
    assert event.audience_filter == {"media_habits_any": ["local_news"]}


def test_exposure_message_and_state_to_dict_are_json_ready() -> None:
    exposure = EventExposure(
        day=2,
        agent_id="jisoo",
        source_type="event",
        source_id="taxi_story",
        channel="news_feed",
        content="A taxi driver says the fee threatens her income.",
    )
    message = EventMessage(
        day=2,
        sender_agent_id="minho",
        channel="neighborhood_group_chat",
        recipient_agent_id=None,
        text="This fee sounds rough for shift workers.",
    )
    state = EventAgentState(**_valid_agent_state_kwargs())  # type: ignore[arg-type]

    assert exposure.to_dict()["source_id"] == "taxi_story"
    assert message.to_dict()["recipient_agent_id"] is None
    assert state.to_dict()["emotion"] == "worried"
