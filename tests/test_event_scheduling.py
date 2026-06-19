from society_simulation.event_models import EventAgentProfile, EventMessage, OpinionEvent
from society_simulation.event_scheduling import build_day_exposures


def agent(agent_id: str, media_habits: list[str]) -> EventAgentProfile:
    return EventAgentProfile.from_dict(
        {
            "agent_id": agent_id,
            "name": agent_id.title(),
            "age": 30,
            "occupation": "resident",
            "household_context": "apartment",
            "neighborhood": "central",
            "core_values": ["fairness"],
            "material_interests": ["commute time"],
            "political_trust": 0.5,
            "media_habits": media_habits,
            "communication_style": "plain",
            "susceptibilities": ["friends"],
            "initial_private_stance": 0.0,
            "initial_public_stance": 0.0,
            "initial_confidence": 0.4,
            "initial_salience": 0.4,
        }
    )


def test_build_day_exposures_delivers_events_by_media_habit() -> None:
    agents = (agent("a", ["local_news"]), agent("b", ["podcast"]))
    event = OpinionEvent.from_dict(
        {
            "event_id": "news",
            "day": 1,
            "title": "City proposal",
            "source": "Local News",
            "source_type": "news",
            "content": "The fee may reduce traffic.",
            "policy_stance": 0.2,
            "credibility": 0.7,
            "emotional_intensity": 0.3,
            "affected_interests": ["commute time"],
            "audience_filter": {"media_habits_any": ["local_news"]},
        }
    )

    exposures = build_day_exposures(
        day=1,
        agents=agents,
        events=(event,),
        previous_messages=(),
        channel_members={"neighborhood": {"a", "b"}},
    )

    assert [exposure.agent_id for exposure in exposures] == ["a"]
    assert exposures[0].source_id == "news"


def test_build_day_exposures_delivers_previous_group_chat_messages_to_channel_members_except_sender() -> None:
    agents = (agent("a", ["local_news"]), agent("b", ["local_news"]))
    message = EventMessage(
        day=1,
        sender_agent_id="a",
        channel="neighborhood",
        recipient_agent_id=None,
        text="I worry this fee hurts workers.",
    )

    exposures = build_day_exposures(
        day=2,
        agents=agents,
        events=(),
        previous_messages=(message,),
        channel_members={"neighborhood": {"a", "b"}},
    )

    assert len(exposures) == 1
    assert exposures[0].agent_id == "b"
    assert exposures[0].source_type == "message"


def test_build_day_exposures_delivers_private_dm_only_to_recipient() -> None:
    agents = (
        agent("a", ["local_news"]),
        agent("b", ["local_news"]),
        agent("c", ["local_news"]),
    )
    message = EventMessage(
        day=1,
        sender_agent_id="a",
        channel="private_dm",
        recipient_agent_id="c",
        text="Can we talk about the charge?",
    )

    exposures = build_day_exposures(
        day=2,
        agents=agents,
        events=(),
        previous_messages=(message,),
        channel_members={},
    )

    assert [exposure.agent_id for exposure in exposures] == ["c"]


def test_build_day_exposures_delivers_events_by_agent_id_filter() -> None:
    agents = (agent("a", ["local_news"]), agent("b", ["local_news"]))
    event = OpinionEvent.from_dict(
        {
            "event_id": "targeted",
            "day": 1,
            "title": "Targeted update",
            "source": "City Hall",
            "source_type": "official",
            "content": "A charge exemption is under review.",
            "policy_stance": 0.1,
            "credibility": 0.8,
            "emotional_intensity": 0.2,
            "affected_interests": ["commute time"],
            "audience_filter": {"agent_ids": ["b"]},
        }
    )

    exposures = build_day_exposures(
        day=1,
        agents=agents,
        events=(event,),
        previous_messages=(),
        channel_members={},
    )

    assert [exposure.agent_id for exposure in exposures] == ["b"]
