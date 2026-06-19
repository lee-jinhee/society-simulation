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
    states_by_day: list[tuple[EventAgentState, ...]] = [
        tuple(agent.initial_state(day=0) for agent in config.agents)
    ]
    all_exposures: list[EventExposure] = []
    all_messages: list[EventMessage] = []
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
            next_states.append(decision.state)
            all_messages.extend(decision.messages)
        states_by_day.append(tuple(next_states))
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


def _channel_members(config: EventDrivenOpinionConfig) -> dict[str, set[str]]:
    members: dict[str, set[str]] = {}
    for relationship in config.relationships:
        for channel in relationship.channels:
            members.setdefault(channel, set()).add(relationship.source_agent_id)
            members.setdefault(channel, set()).add(relationship.target_agent_id)
    return members


def _build_event_policy(config: EventDrivenOpinionConfig) -> object:
    policy = config.update_policy
    if policy["type"] == "mock_persona":
        return MockPersonaPolicy(
            response_style=str(policy.get("response_style", "balanced")),
            input_cost_per_1m_tokens=float(policy.get("input_cost_per_1m_tokens", 0.0)),
            output_cost_per_1m_tokens=float(policy.get("output_cost_per_1m_tokens", 0.0)),
        )

    api_key_env = str(policy.get("api_key_env", "SOCIETY_SIM_LLM_API_KEY"))
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise ValueError(f"{api_key_env} environment variable is required for llm policy")
    return OpenAICompatiblePersonaPolicy(
        model=str(policy["model"]),
        api_key=api_key,
        base_url=str(policy.get("base_url", "https://api.openai.com/v1")),
        temperature=float(policy.get("temperature", 0.0)),
        max_completion_tokens=int(policy.get("max_completion_tokens", 160)),
        token_limit_parameter=str(policy.get("token_limit_parameter", "max_completion_tokens")),
        timeout_seconds=float(policy.get("timeout_seconds", 30.0)),
        input_cost_per_1m_tokens=float(policy.get("input_cost_per_1m_tokens", 0.0)),
        output_cost_per_1m_tokens=float(policy.get("output_cost_per_1m_tokens", 0.0)),
        max_estimated_cost_usd=(
            None
            if "max_estimated_cost_usd" not in policy
            else float(policy["max_estimated_cost_usd"])
        ),
    )
