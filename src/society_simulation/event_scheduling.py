from __future__ import annotations

from collections.abc import Mapping

from society_simulation.event_models import (
    EventAgentProfile,
    EventExposure,
    EventMessage,
    OpinionEvent,
)


def build_day_exposures(
    *,
    day: int,
    agents: tuple[EventAgentProfile, ...],
    events: tuple[OpinionEvent, ...],
    previous_messages: tuple[EventMessage, ...],
    channel_members: Mapping[str, set[str]],
) -> tuple[EventExposure, ...]:
    agent_by_id = {agent.agent_id: agent for agent in agents}
    exposures: list[EventExposure] = []
    for event in events:
        if event.day != day:
            continue
        for agent in agents:
            if _event_visible_to_agent(event, agent):
                exposures.append(
                    EventExposure(
                        day=day,
                        agent_id=agent.agent_id,
                        source_type="event",
                        source_id=event.event_id,
                        channel="news_feed",
                        content=f"{event.title}: {event.content}",
                    )
                )
    for message in previous_messages:
        if message.day >= day:
            continue
        if message.recipient_agent_id is not None:
            if message.recipient_agent_id in agent_by_id:
                exposures.append(
                    EventExposure(
                        day=day,
                        agent_id=message.recipient_agent_id,
                        source_type="message",
                        source_id=f"{message.sender_agent_id}:{message.day}:{message.channel}",
                        channel=f"private_dm:{message.channel}",
                        content=f"{message.sender_agent_id}: {message.text}",
                    )
                )
            continue
        for agent_id in sorted(channel_members.get(message.channel, set())):
            if agent_id == message.sender_agent_id:
                continue
            if agent_id not in agent_by_id:
                continue
            exposures.append(
                EventExposure(
                    day=day,
                    agent_id=agent_id,
                    source_type="message",
                    source_id=f"{message.sender_agent_id}:{message.day}:{message.channel}",
                    channel=message.channel,
                    content=f"{message.sender_agent_id}: {message.text}",
                )
            )
    return tuple(exposures)


def _event_visible_to_agent(event: OpinionEvent, agent: EventAgentProfile) -> bool:
    filters: list[bool] = []

    if "media_habits_any" in event.audience_filter:
        media_any = _require_str_sequence(
            event.audience_filter["media_habits_any"],
            "media_habits_any",
        )
        filters.append(any(item in agent.media_habits for item in media_any))

    if "agent_ids" in event.audience_filter:
        agent_ids = _require_str_sequence(event.audience_filter["agent_ids"], "agent_ids")
        filters.append(agent.agent_id in set(agent_ids))

    if filters:
        return any(filters)
    return True


def _require_str_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return tuple(value)
