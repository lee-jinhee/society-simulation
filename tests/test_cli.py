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


def test_cli_runs_network_herding_config_and_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "network.json"
    output_dir = tmp_path / "network-run"
    config_path.write_text(
        json.dumps(
            {
                "experiment_name": "network_herding",
                "seed": 5,
                "num_agents": 6,
                "initial_opinion": {"type": "bernoulli", "probability_a": 0.5},
                "topology": {"type": "cycle"},
                "scheduler": {"type": "synchronous_rounds", "rounds": 2},
                "observation_policy": {"type": "neighbor_actions"},
                "update_policy": {"type": "majority_rule"},
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["run", str(config_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "experiment=network_herding" in output
    assert "action_counts=" in output
    assert "output_dir=" in output
    assert (output_dir / "graph.json").exists()
    assert (output_dir / "timeseries.jsonl").exists()


def test_example_network_config_exists_and_is_valid() -> None:
    from society_simulation.config import NetworkHerdingConfig, load_config

    config = load_config("examples/network_herding.json")

    assert isinstance(config, NetworkHerdingConfig)
    assert config.experiment_name == "network_herding"
