import json
from pathlib import Path

from society_simulation.config import ExperimentConfig
from society_simulation.models import AgentState
from society_simulation.replay import ReplayWriter


def test_replay_writer_outputs_config_steps_metrics_and_summary(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_name="sequential_information_cascade",
        seed=11,
        num_agents=1,
        true_state="A",
        signal_accuracy=0.7,
        prior_probability=0.5,
        scheduler="sequential",
        observation_policy="previous_actions",
        update_policy="simple_heuristic",
        output_dir=str(tmp_path / "run"),
    )
    state = AgentState(
        agent_id=0,
        private_signal="A",
        belief_probability=1.0,
        confidence=1.0,
        action="A",
        step_index=0,
        observed_actions=(),
    )
    metrics = {
        "final_accuracy": 1.0,
        "correct_cascade": False,
        "wrong_cascade": False,
        "cascade_start_step": None,
        "private_signal_ignored_count": 0,
        "action_counts": {"A": 1, "B": 0},
        "belief_summary": {"min": 1.0, "max": 1.0, "mean": 1.0},
    }

    output_path = ReplayWriter(config).write(
        true_state="A",
        states=[state],
        metrics=metrics,
    )

    assert output_path == tmp_path / "run"
    assert json.loads((output_path / "config.json").read_text(encoding="utf-8"))["seed"] == 11
    assert json.loads((output_path / "metrics.json").read_text(encoding="utf-8"))["final_accuracy"] == 1.0
    step_lines = (output_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(step_lines) == 1
    step = json.loads(step_lines[0])
    assert step["agent_id"] == 0
    assert step["true_state"] == "A"
    assert step["random_seed"] == 11
    assert step["update_policy"] == "simple_heuristic"
    assert "experiment_name=sequential_information_cascade" in (output_path / "summary.txt").read_text(
        encoding="utf-8"
    )
