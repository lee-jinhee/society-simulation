import json
from pathlib import Path

import pytest
from society_simulation import cli


def test_cli_run_writes_replay_and_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "sequential_information_cascade"
    config_path.write_text(
        json.dumps(
            {
                "experiment_name": "sequential_information_cascade",
                "seed": 5,
                "num_agents": 4,
                "true_state": "A",
                "signal_accuracy": 0.7,
                "prior_probability": 0.5,
                "scheduler": "sequential",
                "observation_policy": "previous_actions",
                "update_policy": "simple_heuristic",
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["run", str(config_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "experiment=sequential_information_cascade" in output
    assert "true_state=A" in output
    assert "output_dir=" in output
    assert "action_counts=" in output
    assert "correct_cascade=" in output
    assert "wrong_cascade=" in output
    assert (output_dir / "metrics.json").exists()
