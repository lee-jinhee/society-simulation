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
                        channel=message.channel,
                        content=f"{message.sender_agent_id}: {message.text}",
                    )
                )
            continue
        for agent_id in sorted(channel_members.get(message.channel, set())):
            if agent_id == message.sender_agent_id:
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
    media_any = event.audience_filter.get("media_habits_any")
    if isinstance(media_any, (list, tuple)):
        return any(item in agent.media_habits for item in media_any if isinstance(item, str))
    agent_ids = event.audience_filter.get("agent_ids")
    if isinstance(agent_ids, (list, tuple)):
        return agent.agent_id in {item for item in agent_ids if isinstance(item, str)}
    return True
