from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from society_simulation.event_config import EventDrivenOpinionConfig
from society_simulation.event_metrics import compute_event_metrics
from society_simulation.event_models import EventAgentState, EventExposure, EventMessage
from society_simulation.event_policy import MockPersonaPolicy, OpenAICompatiblePersonaPolicy
from society_simulation.event_replay import EventReplayWriter
from society_simulation.event_scheduling import build_day_exposures


@dataclass(frozen=True)
class EventRunResult:
    states_by_day: tuple[tuple[EventAgentState, ...], ...]
    exposures: tuple[EventExposure, ...]
    messages: tuple[EventMessage, ...]
    metrics: dict[str, Any]
    output_dir: Path


def run_event_driven_opinion_dynamics(config: EventDrivenOpinionConfig) -> EventRunResult:
    config.validate()
    policy = _build_event_policy(config)
    channel_members = _channel_members(config)
    known_agent_ids = {agent.agent_id for agent in config.agents}
    known_channel_ids = _channel_ids(config)
    states_by_day: list[tuple[EventAgentState, ...]] = [
        tuple(agent.initial_state(day=0) for agent in config.agents)
    ]
    all_exposures: list[EventExposure] = []
    all_messages: list[EventMessage] = []
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
            exposures_by_agent: dict[str, list[EventExposure]] = {}
            for exposure in day_exposures:
                exposures_by_agent.setdefault(exposure.agent_id, []).append(exposure)
            next_states: list[EventAgentState] = []
            for profile in config.agents:
                decision = policy.decide(
                    profile,
                    previous_states[profile.agent_id],
                    tuple(exposures_by_agent.get(profile.agent_id, [])),
                    day=day,
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
            states_by_day.append(tuple(next_states))
    except Exception as exc:
        _write_partial_replay(
            config=config,
            policy=policy,
            states_by_day=tuple(states_by_day),
            exposures=tuple(all_exposures),
            messages=tuple(all_messages),
            error=exc,
        )
        raise
    frozen_states = tuple(states_by_day)
    frozen_exposures = tuple(all_exposures)
    frozen_messages = tuple(all_messages)
    metrics = compute_event_metrics(frozen_states, frozen_messages)
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
    )
    return EventRunResult(
        states_by_day=frozen_states,
        exposures=frozen_exposures,
        messages=frozen_messages,
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
    error: Exception,
) -> None:
    metrics = compute_event_metrics(states_by_day, messages)
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
        max_estimated_cost_usd=(
            None
            if "max_estimated_cost_usd" not in policy
            else policy["max_estimated_cost_usd"]  # type: ignore[arg-type]
        ),
    )
