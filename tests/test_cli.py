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
    assert output.splitlines() == [
        "experiment=sequential_information_cascade",
        "true_state=A",
        "action_counts={'A': 1, 'B': 3}",
        "correct_cascade=False",
        "wrong_cascade=False",
        f"output_dir={output_dir}",
    ]
    assert (output_dir / "metrics.json").exists()


def test_cli_run_bad_config_path_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", str(missing)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr().err
    assert "No such file or directory" in captured
    assert "Unable to read config file" in captured
    assert "Traceback" not in captured


def test_cli_run_malformed_json_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", str(bad_path)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr().err
    assert "Invalid config file" in captured
    assert "Expecting" in captured  # JSON decode detail
    assert "Traceback" not in captured


def test_cli_run_invalid_config_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "bad_config.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_name": "sequential_information_cascade",
                "seed": 5,
                "num_agents": 0,
                "true_state": "A",
                "signal_accuracy": 0.7,
                "prior_probability": 0.5,
                "scheduler": "sequential",
                "observation_policy": "previous_actions",
                "update_policy": "simple_heuristic",
                "output_dir": str(tmp_path / "bad-run"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", str(config_path)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr().err
    assert "Invalid config file" in captured
    assert "num_agents must be positive" in captured
    assert "Traceback" not in captured
