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


def _valid_event_kwargs() -> dict[str, object]:
    return {
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
    assert profile.core_values == ("fairness", "public health")
    assert profile.material_interests == ("commute time",)
    assert profile.media_habits == ("local_news", "neighborhood_group_chat")
    assert profile.susceptibilities == ("coworker stories",)
    initial_state = profile.initial_state(day=0)
    assert initial_state.private_stance == -0.1
    assert initial_state.emotion == "calm"
    assert initial_state.memory_summary
    assert initial_state.last_private_reasoning
    profile_data = profile.to_dict()
    assert profile_data["name"] == "Jisoo Park"
    assert profile_data["core_values"] == ["fairness", "public health"]
    assert profile_data["material_interests"] == ["commute time"]
    assert profile_data["media_habits"] == ["local_news", "neighborhood_group_chat"]
    assert profile_data["susceptibilities"] == ["coworker stories"]


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
    relationship_data = relationship.to_dict()
    assert relationship_data["target_agent_id"] == "minho"
    assert relationship_data["channels"] == ["hospital_group_chat"]


def test_event_from_dict_supports_audience_filter() -> None:
    event_data = _valid_event_kwargs()
    event_data["audience_filter"] = {"media_habits_any": ["local_news"]}
    event = OpinionEvent.from_dict(event_data)

    assert event.day == 1
    assert event.audience_filter == {"media_habits_any": ("local_news",)}
    assert event.to_dict()["audience_filter"] == {"media_habits_any": ["local_news"]}
    assert event.to_dict()["affected_interests"] == ["commute time", "public health"]


def test_event_from_dict_defaults_missing_audience_filter_to_empty_dict() -> None:
    event = OpinionEvent.from_dict(_valid_event_kwargs())

    assert event.audience_filter == {}
    assert event.to_dict()["audience_filter"] == {}


@pytest.mark.parametrize(
    "audience_filter",
    [
        "local_news",
        [("media_habits_any", ["local_news"])],
    ],
)
def test_event_rejects_non_dict_audience_filter(audience_filter: object) -> None:
    event_kwargs = _valid_event_kwargs()
    event_kwargs["audience_filter"] = audience_filter

    with pytest.raises(ValueError, match="audience_filter must be an object"):
        OpinionEvent(**event_kwargs)  # type: ignore[arg-type]


def test_event_from_dict_rejects_non_dict_audience_filter() -> None:
    event_data = _valid_event_kwargs()
    event_data["audience_filter"] = "local_news"

    with pytest.raises(ValueError, match="audience_filter must be an object"):
        OpinionEvent.from_dict(event_data)


@pytest.mark.parametrize(
    "audience_filter",
    [
        {"bad_set": {"local_news"}},
        {"bad_object": object()},
    ],
)
def test_event_rejects_unsupported_audience_filter_values(
    audience_filter: dict[str, object],
) -> None:
    event_kwargs = _valid_event_kwargs()
    event_kwargs["audience_filter"] = audience_filter

    with pytest.raises(ValueError, match="audience_filter must contain only JSON-compatible values"):
        OpinionEvent(**event_kwargs)  # type: ignore[arg-type]


def test_event_rejects_non_string_audience_filter_keys() -> None:
    event_kwargs = _valid_event_kwargs()
    event_kwargs["audience_filter"] = {1: "local_news"}

    with pytest.raises(ValueError, match="audience_filter keys must be strings"):
        OpinionEvent(**event_kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_event_rejects_non_finite_audience_filter_float(value: float) -> None:
    event_kwargs = _valid_event_kwargs()
    event_kwargs["audience_filter"] = {"score": value}

    with pytest.raises(ValueError, match="audience_filter must contain only JSON-compatible values"):
        OpinionEvent(**event_kwargs)  # type: ignore[arg-type]


def test_event_audience_filter_scalar_values_round_trip_through_to_dict() -> None:
    event_kwargs = _valid_event_kwargs()
    event_kwargs["audience_filter"] = {
        "source": "local_news",
        "count": 2,
        "weight": 0.75,
        "trusted": True,
        "optional": None,
    }
    event = OpinionEvent(**event_kwargs)  # type: ignore[arg-type]

    assert event.to_dict()["audience_filter"] == {
        "source": "local_news",
        "count": 2,
        "weight": 0.75,
        "trusted": True,
        "optional": None,
    }


def test_event_defensively_copies_audience_filter() -> None:
    audience_filter = {
        "media_habits_any": ["local_news"],
        "nested": {"channels": ["group_chat"]},
    }
    event_kwargs = _valid_event_kwargs()
    event_kwargs["audience_filter"] = audience_filter
    event = OpinionEvent(**event_kwargs)  # type: ignore[arg-type]

    audience_filter["media_habits_any"].append("cable_news")
    audience_filter["nested"]["channels"].append("newspaper")

    assert event.audience_filter == {
        "media_habits_any": ("local_news",),
        "nested": {"channels": ("group_chat",)},
    }
    assert event.to_dict()["audience_filter"] == {
        "media_habits_any": ["local_news"],
        "nested": {"channels": ["group_chat"]},
    }


def test_event_audience_filter_rejects_direct_mutation() -> None:
    event_kwargs = _valid_event_kwargs()
    event_kwargs["audience_filter"] = {"media_habits_any": ["local_news"]}
    event = OpinionEvent(**event_kwargs)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        event.audience_filter["new"] = "value"


def test_event_audience_filter_rejects_nested_mapping_mutation() -> None:
    event_kwargs = _valid_event_kwargs()
    event_kwargs["audience_filter"] = {"nested": {"channels": ["group_chat"]}}
    event = OpinionEvent(**event_kwargs)  # type: ignore[arg-type]

    nested_filter = event.audience_filter["nested"]

    with pytest.raises(TypeError):
        nested_filter["channels"] = ["newspaper"]  # type: ignore[index]


def test_event_to_dict_returns_plain_audience_filter_containers() -> None:
    event_kwargs = _valid_event_kwargs()
    event_kwargs["audience_filter"] = {"nested": {"channels": ["group_chat"]}}
    event = OpinionEvent(**event_kwargs)  # type: ignore[arg-type]

    serialized = event.to_dict()
    audience_filter = serialized["audience_filter"]

    assert isinstance(audience_filter, dict)
    assert isinstance(audience_filter["nested"], dict)  # type: ignore[index]
    assert isinstance(audience_filter["nested"]["channels"], list)  # type: ignore[index]
    assert audience_filter == {"nested": {"channels": ["group_chat"]}}


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
