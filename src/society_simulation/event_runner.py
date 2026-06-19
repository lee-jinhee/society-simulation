from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
import os
from pathlib import Path
import re
from typing import Any

from society_simulation.event_config import EventDrivenOpinionConfig
from society_simulation.event_memory import SocialMemory, build_memory_query, retrieve_memories
from society_simulation.event_metrics import compute_event_metrics
from society_simulation.event_models import (
    EventAgentProfile,
    EventAgentState,
    EventExposure,
    EventMessage,
    EventRelationship,
    OpinionEvent,
)
from society_simulation.event_policy import MockPersonaPolicy, OpenAICompatiblePersonaPolicy
from society_simulation.event_replay import EventReplayWriter
from society_simulation.event_scheduling import build_day_exposures

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class EventRunResult:
    states_by_day: tuple[tuple[EventAgentState, ...], ...]
    exposures: tuple[EventExposure, ...]
    messages: tuple[EventMessage, ...]
    memories: tuple[SocialMemory, ...]
    retrievals: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    output_dir: Path


def run_event_driven_opinion_dynamics(config: EventDrivenOpinionConfig) -> EventRunResult:
    config.validate()
    policy = _build_event_policy(config)
    channel_members = _channel_members(config)
    profile_by_id = {agent.agent_id: agent for agent in config.agents}
    event_by_id = {event.event_id: event for event in config.events}
    relationship_trust = _relationship_trust_by_pair(config.relationships)
    known_agent_ids = {agent.agent_id for agent in config.agents}
    known_channel_ids = _channel_ids(config)
    memory_enabled = bool(config.memory_retrieval["enabled"])
    memory_limit = int(config.memory_retrieval["limit"])
    states_by_day: list[tuple[EventAgentState, ...]] = [
        tuple(agent.initial_state(day=0) for agent in config.agents)
    ]
    all_exposures: list[EventExposure] = []
    all_messages: list[EventMessage] = []
    all_memories: list[SocialMemory] = []
    all_retrievals: list[dict[str, Any]] = []
    try:
        for day in range(1, config.days + 1):
            previous_states = {state.agent_id: state for state in states_by_day[-1]}
            previous_day_messages = tuple(
                message for message in all_messages if message.day == day - 1
            )
            day_exposures = build_day_exposures(
                day=day,
                agents=config.agents,
                events=config.events,
                previous_messages=previous_day_messages,
                channel_members=channel_members,
            )
            all_exposures.extend(day_exposures)
            if memory_enabled:
                for exposure in day_exposures:
                    all_memories.append(
                        _memory_from_exposure(
                            exposure=exposure,
                            agent_profile=profile_by_id[exposure.agent_id],
                            event_by_id=event_by_id,
                            relationship_trust=relationship_trust,
                            sequence=len(all_memories),
                        )
                    )
            exposures_by_agent: dict[str, list[EventExposure]] = {}
            for exposure in day_exposures:
                exposures_by_agent.setdefault(exposure.agent_id, []).append(exposure)
            next_states: list[EventAgentState] = []
            for profile in config.agents:
                agent_exposures = tuple(exposures_by_agent.get(profile.agent_id, []))
                retrieved_memories = ()
                if memory_enabled:
                    query = build_memory_query(
                        agent_id=profile.agent_id,
                        day=day,
                        exposures=agent_exposures,
                        affected_interests=profile.material_interests + profile.core_values,
                    )
                    retrieved_memories = retrieve_memories(
                        tuple(memory for memory in all_memories if memory.day < day),
                        query,
                        limit=memory_limit,
                    )
                    all_retrievals.append(
                        {
                            "agent_id": profile.agent_id,
                            "day": day,
                            "query": query.to_dict(),
                            "retrieved": [item.to_dict() for item in retrieved_memories],
                        }
                    )
                decision = _decide(
                    policy,
                    profile,
                    previous_states[profile.agent_id],
                    agent_exposures,
                    day=day,
                    retrieved_memories=retrieved_memories,
                )
                _validate_generated_messages(
                    decision.messages,
                    profile_agent_id=profile.agent_id,
                    known_agent_ids=known_agent_ids,
                    known_channel_ids=known_channel_ids,
                    day=day,
                )
                _validate_generated_state(
                    decision.state,
                    profile_agent_id=profile.agent_id,
                    day=day,
                )
                next_states.append(decision.state)
                all_messages.extend(decision.messages)
                if memory_enabled:
                    all_memories.append(
                        _memory_from_private_reasoning(
                            profile=profile,
                            state=decision.state,
                            day=day,
                        )
                    )
                    for message in decision.messages:
                        all_memories.append(
                            _memory_from_self_message(
                                message=message,
                                state=decision.state,
                                sequence=len(all_memories),
                            )
                        )
            states_by_day.append(tuple(next_states))
    except Exception as exc:
        _write_partial_replay(
            config=config,
            policy=policy,
            states_by_day=tuple(states_by_day),
            exposures=tuple(all_exposures),
            messages=tuple(all_messages),
            memories=tuple(all_memories),
            retrievals=tuple(all_retrievals),
            error=exc,
        )
        raise
    frozen_states = tuple(states_by_day)
    frozen_exposures = tuple(all_exposures)
    frozen_messages = tuple(all_messages)
    frozen_memories = tuple(all_memories)
    frozen_retrievals = tuple(all_retrievals)
    metrics = compute_event_metrics(
        frozen_states,
        frozen_messages,
        memories=frozen_memories,
        retrievals=frozen_retrievals,
    )
    usage_summary = getattr(policy, "usage_summary", None)
    if callable(usage_summary):
        metrics["llm_usage"] = usage_summary()
    audit_records = getattr(policy, "audit_records", None)
    llm_decisions = audit_records() if callable(audit_records) else ()
    output_dir = EventReplayWriter(config).write(
        states_by_day=frozen_states,
        exposures=frozen_exposures,
        messages=frozen_messages,
        metrics=metrics,
        llm_decisions=llm_decisions,
        memories=frozen_memories,
        retrievals=frozen_retrievals,
    )
    return EventRunResult(
        states_by_day=frozen_states,
        exposures=frozen_exposures,
        messages=frozen_messages,
        memories=frozen_memories,
        retrievals=frozen_retrievals,
        metrics=metrics,
        output_dir=output_dir,
    )


def _validate_generated_messages(
    messages: tuple[EventMessage, ...],
    *,
    profile_agent_id: str,
    known_agent_ids: set[str],
    known_channel_ids: set[str],
    day: int,
) -> None:
    for message in messages:
        if message.sender_agent_id != profile_agent_id:
            raise ValueError("generated message sender_agent_id must match profile")
        if message.sender_agent_id not in known_agent_ids:
            raise ValueError("generated message sender_agent_id is not in agents")
        if message.channel not in known_channel_ids:
            raise ValueError("generated message channel is not in channels")
        if (
            message.recipient_agent_id is not None
            and message.recipient_agent_id not in known_agent_ids
        ):
            raise ValueError("generated message recipient_agent_id is not in agents")
        if message.day != day:
            raise ValueError("generated message day must match simulation day")


def _validate_generated_state(
    state: EventAgentState,
    *,
    profile_agent_id: str,
    day: int,
) -> None:
    if state.agent_id != profile_agent_id:
        raise ValueError("generated state agent_id must match profile")
    if state.day != day:
        raise ValueError("generated state day must match simulation day")


def _write_partial_replay(
    *,
    config: EventDrivenOpinionConfig,
    policy: object,
    states_by_day: tuple[tuple[EventAgentState, ...], ...],
    exposures: tuple[EventExposure, ...],
    messages: tuple[EventMessage, ...],
    memories: tuple[SocialMemory, ...],
    retrievals: tuple[dict[str, Any], ...],
    error: Exception,
) -> None:
    metrics = compute_event_metrics(
        states_by_day,
        messages,
        memories=memories,
        retrievals=retrievals,
    )
    metrics["error"] = str(error)
    usage_summary = getattr(policy, "usage_summary", None)
    if callable(usage_summary):
        metrics["llm_usage"] = usage_summary()
    audit_records = getattr(policy, "audit_records", None)
    llm_decisions = audit_records() if callable(audit_records) else ()
    EventReplayWriter(config).write(
        states_by_day=states_by_day,
        exposures=exposures,
        messages=messages,
        metrics=metrics,
        llm_decisions=llm_decisions,
        memories=memories,
        retrievals=retrievals,
    )


def _channel_members(config: EventDrivenOpinionConfig) -> dict[str, set[str]]:
    members: dict[str, set[str]] = {}
    for relationship in config.relationships:
        for channel in relationship.channels:
            members.setdefault(channel, set()).add(relationship.source_agent_id)
            members.setdefault(channel, set()).add(relationship.target_agent_id)
    return members


def _channel_ids(config: EventDrivenOpinionConfig) -> set[str]:
    return set(_configured_channel_ids(config))


def _configured_channel_ids(config: EventDrivenOpinionConfig) -> tuple[str, ...]:
    return tuple(
        channel_id
        for channel in config.channels
        if isinstance((channel_id := channel.get("channel_id")), str)
    )


def _relationship_trust_by_pair(
    relationships: tuple[EventRelationship, ...],
) -> dict[tuple[str, str], float]:
    trust: dict[tuple[str, str], float] = {}
    for relationship in relationships:
        trust[(relationship.source_agent_id, relationship.target_agent_id)] = relationship.trust
        trust.setdefault(
            (relationship.target_agent_id, relationship.source_agent_id),
            relationship.trust,
        )
    return trust


def _decide(
    policy: object,
    profile: EventAgentProfile,
    current_state: EventAgentState,
    exposures: tuple[EventExposure, ...],
    *,
    day: int,
    retrieved_memories: tuple[object, ...],
) -> object:
    decide = getattr(policy, "decide")
    if _accepts_retrieved_memories(policy):
        return decide(
            profile,
            current_state,
            exposures,
            day=day,
            retrieved_memories=retrieved_memories,
        )
    return decide(profile, current_state, exposures, day=day)


def _accepts_retrieved_memories(policy: object) -> bool:
    try:
        return "retrieved_memories" in signature(getattr(policy, "decide")).parameters
    except (TypeError, ValueError):
        return True


def _memory_from_exposure(
    *,
    exposure: EventExposure,
    agent_profile: EventAgentProfile,
    event_by_id: dict[str, OpinionEvent],
    relationship_trust: dict[tuple[str, str], float],
    sequence: int,
) -> SocialMemory:
    event = event_by_id.get(exposure.source_id) if exposure.source_type == "event" else None
    sender_agent_id = _sender_agent_id_from_message_exposure(exposure)
    identity_relevance = _interest_overlap_score(
        exposure.content,
        agent_profile.material_interests + agent_profile.core_values,
    )
    emotional_intensity = event.emotional_intensity if event is not None else 0.5
    source_trust = (
        event.credibility
        if event is not None
        else relationship_trust.get((sender_agent_id or "", exposure.agent_id), 0.5)
    )
    return SocialMemory(
        memory_id=f"{exposure.agent_id}:{exposure.day}:exposure:{sequence}",
        agent_id=exposure.agent_id,
        day=exposure.day,
        kind="social_message" if exposure.source_type == "message" else "event_exposure",
        text=exposure.content,
        source_id=exposure.source_id,
        source_type=exposure.source_type,
        channel=exposure.channel,
        related_agent_ids=(sender_agent_id,) if sender_agent_id is not None else (),
        related_event_ids=(exposure.source_id,) if event is not None else (),
        stance_signal=event.policy_stance if event is not None else 0.0,
        emotional_intensity=emotional_intensity,
        source_trust=source_trust,
        identity_relevance=identity_relevance,
        importance=max(0.3, min(1.0, (identity_relevance + emotional_intensity + source_trust) / 3)),
        private=False,
    )


def _memory_from_private_reasoning(
    *,
    profile: EventAgentProfile,
    state: EventAgentState,
    day: int,
) -> SocialMemory:
    return SocialMemory(
        memory_id=f"{profile.agent_id}:{day}:self_reasoning",
        agent_id=profile.agent_id,
        day=day,
        kind="self_reasoning",
        text=state.last_private_reasoning,
        source_id=f"{profile.agent_id}:{day}:decision",
        source_type="self",
        channel="internal",
        related_agent_ids=(),
        related_event_ids=(),
        stance_signal=state.private_stance,
        emotional_intensity=state.salience,
        source_trust=1.0,
        identity_relevance=state.salience,
        importance=state.confidence,
        private=True,
    )


def _memory_from_self_message(
    *,
    message: EventMessage,
    state: EventAgentState,
    sequence: int,
) -> SocialMemory:
    return SocialMemory(
        memory_id=f"{message.sender_agent_id}:{message.day}:self_message:{sequence}",
        agent_id=message.sender_agent_id,
        day=message.day,
        kind="self_message",
        text=message.text,
        source_id=f"{message.sender_agent_id}:{message.day}:{message.channel}",
        source_type="self",
        channel=message.channel,
        related_agent_ids=(
            (message.recipient_agent_id,) if message.recipient_agent_id is not None else ()
        ),
        related_event_ids=(),
        stance_signal=state.public_stance,
        emotional_intensity=state.salience,
        source_trust=1.0,
        identity_relevance=state.salience,
        importance=state.confidence,
        private=message.recipient_agent_id is not None,
    )


def _sender_agent_id_from_message_exposure(exposure: EventExposure) -> str | None:
    if exposure.source_type != "message" or ":" not in exposure.source_id:
        return None
    sender_agent_id = exposure.source_id.split(":", 1)[0]
    return sender_agent_id if sender_agent_id else None


def _interest_overlap_score(text: str, interests: tuple[str, ...]) -> float:
    interest_tokens: set[str] = set()
    for interest in interests:
        interest_tokens.update(_tokens(interest))
    if not interest_tokens:
        return 0.0
    text_tokens = _tokens(text)
    if not text_tokens:
        return 0.0
    return min(1.0, len(text_tokens & interest_tokens) / len(interest_tokens))


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _build_event_policy(config: EventDrivenOpinionConfig) -> object:
    policy = config.update_policy
    if policy["type"] == "mock_persona":
        return MockPersonaPolicy(
            response_style=policy.get("response_style", "balanced"),  # type: ignore[arg-type]
            configured_channels=_configured_channel_ids(config),
            input_cost_per_1m_tokens=policy.get("input_cost_per_1m_tokens", 0.0),  # type: ignore[arg-type]
            output_cost_per_1m_tokens=policy.get("output_cost_per_1m_tokens", 0.0),  # type: ignore[arg-type]
        )

    api_key_env = policy.get("api_key_env", "SOCIETY_SIM_LLM_API_KEY")
    if not isinstance(api_key_env, str):
        raise ValueError("update_policy.api_key_env must be a non-empty string")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise ValueError(f"{api_key_env} environment variable is required for llm policy")
    return OpenAICompatiblePersonaPolicy(
        model=policy["model"],  # type: ignore[arg-type]
        api_key=api_key,
        provider=policy.get("provider", "openai_compatible"),  # type: ignore[arg-type]
        base_url=policy.get("base_url", "https://api.openai.com/v1"),  # type: ignore[arg-type]
        temperature=policy.get("temperature", 0.0),  # type: ignore[arg-type]
        max_completion_tokens=policy.get("max_completion_tokens", 160),  # type: ignore[arg-type]
        token_limit_parameter=policy.get("token_limit_parameter", "max_completion_tokens"),  # type: ignore[arg-type]
        timeout_seconds=policy.get("timeout_seconds", 30.0),  # type: ignore[arg-type]
        input_cost_per_1m_tokens=policy.get("input_cost_per_1m_tokens", 0.0),  # type: ignore[arg-type]
        output_cost_per_1m_tokens=policy.get("output_cost_per_1m_tokens", 0.0),  # type: ignore[arg-type]
        configured_channels=_configured_channel_ids(config),
        known_agent_ids=tuple(agent.agent_id for agent in config.agents),
        max_estimated_cost_usd=(
            None
            if "max_estimated_cost_usd" not in policy
            else policy["max_estimated_cost_usd"]  # type: ignore[arg-type]
        ),
    )
