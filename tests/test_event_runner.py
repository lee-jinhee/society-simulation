import json
from pathlib import Path

import pytest

from tests.test_event_config import valid_event_config

import society_simulation.event_runner as event_runner
from society_simulation.config import load_config
from society_simulation.event_config import EventDrivenOpinionConfig
from society_simulation.event_models import EventAgentState, EventMessage
from society_simulation.event_policy import EventPolicyDecision
from society_simulation.runner import run_experiment


def test_run_experiment_dispatches_event_driven_opinion_dynamics(tmp_path: Path) -> None:
    path = tmp_path / "event.json"
    path.write_text(json.dumps(valid_event_config(tmp_path)), encoding="utf-8")
    config = load_config(path)
    assert isinstance(config, EventDrivenOpinionConfig)

    result = run_experiment(config)

    assert result.output_dir == tmp_path / "event-run"
    assert len(result.states_by_day) == config.days + 1
    assert len(result.states_by_day[0]) == len(config.agents)
    assert result.metrics["agent_count"] == len(config.agents)
    assert "llm_usage" in result.metrics
    assert (result.output_dir / "agent_states.jsonl").exists()
    assert (result.output_dir / "messages.jsonl").exists()
    assert (result.output_dir / "llm_decisions.jsonl").exists()


def test_event_runner_is_deterministic_for_mock_policy(tmp_path: Path) -> None:
    first_data = valid_event_config(tmp_path)
    second_data = valid_event_config(tmp_path)
    first_data["output_dir"] = str(tmp_path / "first")
    second_data["output_dir"] = str(tmp_path / "second")

    first = run_experiment(EventDrivenOpinionConfig.from_dict(first_data))
    second = run_experiment(EventDrivenOpinionConfig.from_dict(second_data))

    assert first.metrics == second.metrics
    assert (first.output_dir / "agent_states.jsonl").read_text(encoding="utf-8") == (
        second.output_dir / "agent_states.jsonl"
    ).read_text(encoding="utf-8")
    assert (first.output_dir / "metrics.json").read_text(encoding="utf-8") == (
        second.output_dir / "metrics.json"
    ).read_text(encoding="utf-8")
    assert (first.output_dir / "exposures.jsonl").read_text(encoding="utf-8") == (
        second.output_dir / "exposures.jsonl"
    ).read_text(encoding="utf-8")
    assert (first.output_dir / "llm_decisions.jsonl").read_text(encoding="utf-8") == (
        second.output_dir / "llm_decisions.jsonl"
    ).read_text(encoding="utf-8")


def test_mock_policy_uses_configured_non_chat_channel(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["output_dir"] = str(tmp_path / "renamed-channel")
    data["channels"] = [{"channel_id": "community_forum", "type": "group_chat"}]
    for agent in data["agents"]:  # type: ignore[index]
        agent["media_habits"] = ["local_news", "community_forum"]  # type: ignore[index]
    for relationship in data["relationships"]:  # type: ignore[index]
        relationship["channels"] = ["community_forum"]  # type: ignore[index]

    result = run_experiment(EventDrivenOpinionConfig.from_dict(data))

    assert result.messages
    assert {message.channel for message in result.messages} == {"community_forum"}


def test_event_runner_exposes_only_previous_day_messages(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["output_dir"] = str(tmp_path / "previous-day")

    result = run_experiment(EventDrivenOpinionConfig.from_dict(data))

    message_exposures = [
        exposure for exposure in result.exposures if exposure.source_type == "message"
    ]
    day2_sources = {exposure.source_id for exposure in message_exposures if exposure.day == 2}
    day3_sources = {exposure.source_id for exposure in message_exposures if exposure.day == 3}

    assert day2_sources
    assert all(":1:" in source_id for source_id in day2_sources)
    assert all(":2:" in source_id for source_id in day3_sources)
    assert not any(":1:" in source_id for source_id in day3_sources)
    assert not any(f":{exposure.day}:" in exposure.source_id for exposure in message_exposures)


def test_event_runner_requires_configured_llm_api_key_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = valid_event_config(tmp_path)
    data["update_policy"] = {
        "type": "llm",
        "model": "gpt-test",
        "api_key_env": "MISSING_EVENT_KEY",
    }
    monkeypatch.delenv("MISSING_EVENT_KEY", raising=False)

    with pytest.raises(ValueError, match="MISSING_EVENT_KEY.*required"):
        run_experiment(EventDrivenOpinionConfig.from_dict(data))


def test_event_runner_passes_configured_social_context_to_llm_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class CapturingPolicy:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)

    data = valid_event_config(tmp_path)
    data["update_policy"] = {
        "type": "llm",
        "model": "cheap-chat",
        "api_key_env": "EVENT_TEST_KEY",
    }
    monkeypatch.setenv("EVENT_TEST_KEY", "secret-key")
    monkeypatch.setattr(event_runner, "OpenAICompatiblePersonaPolicy", CapturingPolicy)

    event_runner._build_event_policy(EventDrivenOpinionConfig.from_dict(data))

    assert captured["configured_channels"] == ("neighborhood_group_chat",)
    assert captured["known_agent_ids"] == ("jisoo", "minho")


def _copy_state(
    current_state: EventAgentState,
    *,
    agent_id: str | None = None,
    day: int | None = None,
) -> EventAgentState:
    return EventAgentState(
        agent_id=current_state.agent_id if agent_id is None else agent_id,
        day=current_state.day if day is None else day,
        private_stance=current_state.private_stance,
        public_stance=current_state.public_stance,
        confidence=current_state.confidence,
        salience=current_state.salience,
        emotion=current_state.emotion,
        memory_summary=current_state.memory_summary,
        last_private_reasoning=current_state.last_private_reasoning,
    )


def test_event_runner_rejects_policy_state_for_wrong_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = valid_event_config(tmp_path)
    data["days"] = 1
    data["output_dir"] = str(tmp_path / "wrong-state-agent")

    class WrongAgentStatePolicy:
        def decide(self, profile, current_state, exposures, *, day):  # type: ignore[no-untyped-def]
            del profile, exposures
            return EventPolicyDecision(
                state=_copy_state(current_state, agent_id="minho", day=day),
                messages=(),
            )

    monkeypatch.setattr(event_runner, "_build_event_policy", lambda config: WrongAgentStatePolicy())

    with pytest.raises(ValueError, match="generated state agent_id must match profile"):
        run_experiment(EventDrivenOpinionConfig.from_dict(data))

    state_rows = [
        json.loads(line)
        for line in (Path(data["output_dir"]) / "agent_states.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(state_rows) == 2
    assert {row["day"] for row in state_rows} == {0}


def test_event_runner_rejects_policy_state_for_wrong_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = valid_event_config(tmp_path)
    data["days"] = 1
    data["output_dir"] = str(tmp_path / "wrong-state-day")

    class WrongDayStatePolicy:
        def decide(self, profile, current_state, exposures, *, day):  # type: ignore[no-untyped-def]
            del profile, exposures
            return EventPolicyDecision(
                state=_copy_state(current_state, day=day + 1),
                messages=(),
            )

    monkeypatch.setattr(event_runner, "_build_event_policy", lambda config: WrongDayStatePolicy())

    with pytest.raises(ValueError, match="generated state day must match simulation day"):
        run_experiment(EventDrivenOpinionConfig.from_dict(data))

    state_rows = [
        json.loads(line)
        for line in (Path(data["output_dir"]) / "agent_states.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(state_rows) == 2
    assert {row["day"] for row in state_rows} == {0}


def test_event_runner_rejects_generated_message_for_wrong_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = valid_event_config(tmp_path)
    data["days"] = 1
    data["output_dir"] = str(tmp_path / "wrong-message-day")

    class WrongDayMessagePolicy:
        def decide(self, profile, current_state, exposures, *, day):  # type: ignore[no-untyped-def]
            del profile, exposures
            return EventPolicyDecision(
                state=_copy_state(current_state, day=day),
                messages=(
                    EventMessage(
                        day=day + 1,
                        sender_agent_id=current_state.agent_id,
                        channel="neighborhood_group_chat",
                        recipient_agent_id=None,
                        text="This should not be recorded.",
                    ),
                ),
            )

    monkeypatch.setattr(event_runner, "_build_event_policy", lambda config: WrongDayMessagePolicy())

    with pytest.raises(ValueError, match="generated message day must match simulation day"):
        run_experiment(EventDrivenOpinionConfig.from_dict(data))

    assert (Path(data["output_dir"]) / "messages.jsonl").read_text(encoding="utf-8") == ""


def test_event_runner_rejects_generated_message_for_unknown_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidMessagePolicy:
        def decide(self, profile, current_state, exposures, *, day):  # type: ignore[no-untyped-def]
            del exposures
            return EventPolicyDecision(
                state=EventAgentState(
                    agent_id=profile.agent_id,
                    day=day,
                    private_stance=current_state.private_stance,
                    public_stance=current_state.public_stance,
                    confidence=current_state.confidence,
                    salience=current_state.salience,
                    emotion=current_state.emotion,
                    memory_summary=current_state.memory_summary,
                    last_private_reasoning=current_state.last_private_reasoning,
                ),
                messages=(
                    EventMessage(
                        day=day,
                        sender_agent_id=profile.agent_id,
                        channel="missing_channel",
                        recipient_agent_id=None,
                        text="This should not be recorded.",
                    ),
                ),
            )

    monkeypatch.setattr(event_runner, "_build_event_policy", lambda config: InvalidMessagePolicy())

    with pytest.raises(ValueError, match="generated message channel is not in channels"):
        run_experiment(EventDrivenOpinionConfig.from_dict(valid_event_config(tmp_path)))


def test_event_runner_rejects_generated_message_for_wrong_sender(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidSenderPolicy:
        def decide(self, profile, current_state, exposures, *, day):  # type: ignore[no-untyped-def]
            del profile, exposures
            return EventPolicyDecision(
                state=EventAgentState(
                    agent_id=current_state.agent_id,
                    day=day,
                    private_stance=current_state.private_stance,
                    public_stance=current_state.public_stance,
                    confidence=current_state.confidence,
                    salience=current_state.salience,
                    emotion=current_state.emotion,
                    memory_summary=current_state.memory_summary,
                    last_private_reasoning=current_state.last_private_reasoning,
                ),
                messages=(
                    EventMessage(
                        day=day,
                        sender_agent_id="minho",
                        channel="neighborhood_group_chat",
                        recipient_agent_id=None,
                        text="This should not be recorded.",
                    ),
                ),
            )

    monkeypatch.setattr(event_runner, "_build_event_policy", lambda config: InvalidSenderPolicy())

    with pytest.raises(ValueError, match="generated message sender_agent_id must match profile"):
        run_experiment(EventDrivenOpinionConfig.from_dict(valid_event_config(tmp_path)))


def test_event_runner_persists_partial_replay_with_audited_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = valid_event_config(tmp_path)
    data["output_dir"] = str(tmp_path / "partial")
    audit_row = {
        "agent_id": "jisoo",
        "day": 1,
        "provider": "openai_compatible",
        "model": "cheap-chat",
        "policy_type": "event_persona",
        "prompt": "prompt",
        "raw_response": {"choices": [{"message": {"content": "malformed"}}]},
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "input_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "total_cost_usd": 0.0,
        "latency_ms": 1.0,
        "error": "event llm response content must be JSON",
    }

    class AuditedFailurePolicy:
        def decide(self, profile, current_state, exposures, *, day):  # type: ignore[no-untyped-def]
            del profile, current_state, exposures, day
            raise ValueError("event llm response content must be JSON")

        def usage_summary(self) -> dict[str, object]:
            return {"provider": "openai_compatible", "model": "cheap-chat", "calls": 1}

        def audit_records(self) -> tuple[dict[str, object], ...]:
            return (audit_row,)

    monkeypatch.setattr(event_runner, "_build_event_policy", lambda config: AuditedFailurePolicy())

    with pytest.raises(ValueError, match="event llm response content must be JSON"):
        run_experiment(EventDrivenOpinionConfig.from_dict(data))

    output_dir = Path(data["output_dir"])
    assert (output_dir / "agent_states.jsonl").exists()
    assert (output_dir / "exposures.jsonl").exists()
    assert (output_dir / "messages.jsonl").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["error"] == "event llm response content must be JSON"
    rows = [
        json.loads(line)
        for line in (output_dir / "llm_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [audit_row]
