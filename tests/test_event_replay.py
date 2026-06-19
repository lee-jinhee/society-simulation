import json
from pathlib import Path

from tests.test_event_config import valid_event_config

from society_simulation.event_config import EventDrivenOpinionConfig
from society_simulation.event_models import EventExposure, EventMessage
from society_simulation.event_replay import EventReplayWriter


def test_event_replay_writer_writes_required_artifacts(tmp_path: Path) -> None:
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

    output_dir = EventReplayWriter(config).write(
        states_by_day=states_by_day,
        exposures=exposures,
        messages=messages,
        metrics=metrics,
        llm_decisions=llm_decisions,
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
