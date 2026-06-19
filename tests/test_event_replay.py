import json
from pathlib import Path

import pytest

from tests.test_event_config import valid_event_config

from society_simulation.event_config import EventDrivenOpinionConfig
from society_simulation.event_memory import RetrievedMemory, SocialMemory
from society_simulation.event_models import EventExposure, EventMessage
from society_simulation.event_replay import EventReplayWriter


def _replay_inputs(tmp_path: Path) -> dict[str, object]:
    config = EventDrivenOpinionConfig.from_dict(valid_event_config(tmp_path))
    states_by_day = (
        tuple(agent.initial_state(day=0) for agent in config.agents),
        tuple(agent.initial_state(day=1) for agent in config.agents),
    )
    exposures = (
        EventExposure(
            day=1,
            agent_id="jisoo",
            source_type="event",
            source_id="city_announcement",
            channel="news_feed",
            content="City announcement.",
        ),
    )
    messages = (
        EventMessage(
            day=1,
            sender_agent_id="jisoo",
            channel="neighborhood_group_chat",
            recipient_agent_id=None,
            text="I am thinking about the fee.",
        ),
    )
    metrics = {"agent_count": 2, "day_count": 2, "final_private_stance_mean": 0.0}
    llm_decisions = (
        {"agent_id": "jisoo", "day": 1, "prompt": "prompt", "raw_response": {"content": "{}"}},
    )
    return {
        "config": config,
        "states_by_day": states_by_day,
        "exposures": exposures,
        "messages": messages,
        "metrics": metrics,
        "llm_decisions": llm_decisions,
    }


def test_event_replay_writer_writes_required_artifacts(tmp_path: Path) -> None:
    inputs = _replay_inputs(tmp_path)
    config = inputs["config"]
    assert isinstance(config, EventDrivenOpinionConfig)

    output_dir = EventReplayWriter(config).write(
        states_by_day=inputs["states_by_day"],
        exposures=inputs["exposures"],
        messages=inputs["messages"],
        metrics=inputs["metrics"],
        llm_decisions=inputs["llm_decisions"],
    )

    assert output_dir == Path(config.output_dir)
    for name in [
        "config.json",
        "agents.json",
        "relationships.json",
        "events.json",
        "exposures.jsonl",
        "messages.jsonl",
        "agent_states.jsonl",
        "metrics.json",
        "summary.md",
        "llm_decisions.jsonl",
    ]:
        assert (output_dir / name).exists()
    state_rows = [
        json.loads(line)
        for line in (output_dir / "agent_states.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert state_rows[0]["agent_id"] == "jisoo"

    config_payload = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config_payload["experiment_name"] == "event_driven_opinion_dynamics"
    agents_payload = json.loads((output_dir / "agents.json").read_text(encoding="utf-8"))
    assert len(agents_payload["agents"]) == 2
    relationships_payload = json.loads(
        (output_dir / "relationships.json").read_text(encoding="utf-8")
    )
    assert "relationships" in relationships_payload
    events_payload = json.loads((output_dir / "events.json").read_text(encoding="utf-8"))
    assert "events" in events_payload
    exposure_rows = [
        json.loads(line)
        for line in (output_dir / "exposures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exposure_rows[0]["source_id"] == "city_announcement"
    message_rows = [
        json.loads(line)
        for line in (output_dir / "messages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert message_rows[0]["text"] == "I am thinking about the fee."
    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["agent_count"] == 2
    decision_rows = [
        json.loads(line)
        for line in (output_dir / "llm_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert decision_rows[0]["agent_id"] == "jisoo"
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "congestion_pricing" in summary
    assert "final_private_stance_mean" in summary


def test_event_replay_writer_writes_empty_llm_decisions_file(tmp_path: Path) -> None:
    inputs = _replay_inputs(tmp_path)
    config = inputs["config"]
    assert isinstance(config, EventDrivenOpinionConfig)

    output_dir = EventReplayWriter(config).write(
        states_by_day=inputs["states_by_day"],
        exposures=inputs["exposures"],
        messages=inputs["messages"],
        metrics=inputs["metrics"],
        llm_decisions=(),
    )

    decisions_path = output_dir / "llm_decisions.jsonl"
    assert decisions_path.exists()
    assert decisions_path.read_text(encoding="utf-8") == ""


def test_event_replay_writer_writes_memory_artifacts(tmp_path: Path) -> None:
    inputs = _replay_inputs(tmp_path)
    config = inputs["config"]
    assert isinstance(config, EventDrivenOpinionConfig)
    social_memory = SocialMemory(
        memory_id="jisoo:1:event:city",
        agent_id="jisoo",
        day=1,
        kind="event_exposure",
        text="City announcement",
        source_id="city",
        source_type="event",
        channel="news_feed",
        related_agent_ids=(),
        related_event_ids=("city",),
        stance_signal=0.2,
        emotional_intensity=0.3,
        source_trust=0.8,
        identity_relevance=0.6,
        importance=0.7,
        private=False,
    )

    output_dir = EventReplayWriter(config).write(
        states_by_day=inputs["states_by_day"],
        exposures=inputs["exposures"],
        messages=inputs["messages"],
        metrics=inputs["metrics"],
        llm_decisions=(),
        memories=(social_memory,),
        retrievals=(
            {
                "agent_id": "jisoo",
                "day": 1,
                "query": {"agent_id": "jisoo"},
                "retrieved": [
                    RetrievedMemory(social_memory, 1, 1, 1, 1, 1, 1, 1).to_dict()
                ],
            },
        ),
    )

    memory_rows = [
        json.loads(line)
        for line in (output_dir / "memories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    retrieval_rows = [
        json.loads(line)
        for line in (output_dir / "retrievals.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert memory_rows[0]["memory_id"] == "jisoo:1:event:city"
    assert retrieval_rows[0]["retrieved"][0]["memory"]["memory_id"] == "jisoo:1:event:city"


def test_event_replay_writer_overwrites_stale_llm_decisions(tmp_path: Path) -> None:
    inputs = _replay_inputs(tmp_path)
    config = inputs["config"]
    assert isinstance(config, EventDrivenOpinionConfig)
    writer = EventReplayWriter(config)

    writer.write(
        states_by_day=inputs["states_by_day"],
        exposures=inputs["exposures"],
        messages=inputs["messages"],
        metrics=inputs["metrics"],
        llm_decisions=inputs["llm_decisions"],
    )
    output_dir = writer.write(
        states_by_day=inputs["states_by_day"],
        exposures=inputs["exposures"],
        messages=inputs["messages"],
        metrics=inputs["metrics"],
        llm_decisions=(),
    )

    assert (output_dir / "llm_decisions.jsonl").read_text(encoding="utf-8") == ""


def test_event_replay_writer_rejects_non_strict_json(tmp_path: Path) -> None:
    inputs = _replay_inputs(tmp_path)
    config = inputs["config"]
    assert isinstance(config, EventDrivenOpinionConfig)

    with pytest.raises(ValueError):
        EventReplayWriter(config).write(
            states_by_day=inputs["states_by_day"],
            exposures=inputs["exposures"],
            messages=inputs["messages"],
            metrics={"bad": float("nan")},
            llm_decisions=inputs["llm_decisions"],
        )
