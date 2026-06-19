import json
from pathlib import Path

import pytest

from tests.test_event_config import valid_event_config

from society_simulation.config import load_config
from society_simulation.event_config import EventDrivenOpinionConfig
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
