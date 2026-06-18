import json
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("error", [OSError("disk full"), ValueError("bad run")])
def test_cli_run_runtime_failures_report_experiment_run_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    error: OSError | ValueError,
) -> None:
    config_path = tmp_path / "config.json"
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
                "output_dir": str(tmp_path / "run"),
            }
        ),
        encoding="utf-8",
    )

    def raise_runtime_error(_config: object) -> object:
        raise error

    monkeypatch.setattr(cli, "run_experiment", raise_runtime_error)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", str(config_path)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr().err
    assert "Experiment run failed" in captured
    assert str(error) in captured
    assert "Invalid config file" not in captured
    assert "Traceback" not in captured


def test_cli_run_missing_action_counts_reports_experiment_run_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
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
                "output_dir": str(tmp_path / "network-run"),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cli,
        "run_experiment",
        lambda _config: SimpleNamespace(metrics={}, output_dir=tmp_path / "network-run"),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", str(config_path)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Experiment run failed" in captured.err
    assert "metrics must include action_counts or final_action_counts" in captured.err
    assert "Invalid config file" not in captured.err
    assert "Traceback" not in captured.err
    assert "action_counts=None" not in captured.out


def test_cli_sweep_runs_config_and_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sweep_path = tmp_path / "sweep.json"
    output_dir = tmp_path / "sweep-output"
    sweep_path.write_text(
        json.dumps(
            {
                "sweep_name": "cli_sweep",
                "base_config": {
                    "experiment_name": "network_herding",
                    "seed": 1,
                    "num_agents": 6,
                    "initial_opinion": {"type": "bernoulli", "probability_a": 0.5},
                    "topology": {"type": "cycle"},
                    "scheduler": {"type": "synchronous_rounds", "rounds": 2},
                    "observation_policy": {"type": "neighbor_actions"},
                    "update_policy": {"type": "majority_rule"},
                    "output_dir": str(tmp_path / "ignored"),
                },
                "factors": [{"name": "seed", "path": "seed", "values": [1, 2]}],
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["sweep", str(sweep_path)])

    assert exit_code == 0
    output = capsys.readouterr().out.splitlines()
    assert output == [
        "sweep=cli_sweep",
        "runs=2",
        "completed=2",
        "failed=0",
        f"output_dir={output_dir}",
        f"summary_csv={output_dir / 'summary.csv'}",
    ]
    assert (output_dir / "manifest.jsonl").exists()
    assert (output_dir / "summary.csv").exists()


def test_cli_sweep_invalid_config_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sweep_path = tmp_path / "bad_sweep.json"
    sweep_path.write_text(json.dumps({"sweep_name": ""}), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["sweep", str(sweep_path)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr().err
    assert "Invalid sweep config file" in captured
    assert "sweep_name must be a non-empty string" in captured
    assert "Traceback" not in captured


def test_cli_sweep_missing_path_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing_sweep.json"

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["sweep", str(missing)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr().err
    assert "Unable to read sweep config file" in captured
    assert "Traceback" not in captured


def test_cli_sweep_returns_one_when_any_run_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep_path = tmp_path / "sweep.json"
    output_dir = tmp_path / "sweep-output"
    summary_csv = output_dir / "summary.csv"
    sweep_path.write_text(
        json.dumps(
            {
                "sweep_name": "cli_sweep",
                "base_config": {
                    "experiment_name": "network_herding",
                    "seed": 1,
                    "num_agents": 6,
                    "initial_opinion": {"type": "bernoulli", "probability_a": 0.5},
                    "topology": {"type": "cycle"},
                    "scheduler": {"type": "synchronous_rounds", "rounds": 2},
                    "observation_policy": {"type": "neighbor_actions"},
                    "update_policy": {"type": "majority_rule"},
                    "output_dir": str(tmp_path / "ignored"),
                },
                "factors": [{"name": "seed", "path": "seed", "values": [1, 2]}],
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cli,
        "run_sweep",
        lambda _sweep: SimpleNamespace(
            sweep_name="cli_sweep",
            runs=2,
            completed=1,
            failed=1,
            output_dir=output_dir,
            summary_csv_path=summary_csv,
        ),
        raising=False,
    )

    exit_code = cli.main(["sweep", str(sweep_path)])

    assert exit_code == 1
    output = capsys.readouterr().out.splitlines()
    assert output == [
        "sweep=cli_sweep",
        "runs=2",
        "completed=1",
        "failed=1",
        f"output_dir={output_dir}",
        f"summary_csv={summary_csv}",
    ]


def test_cli_sweep_runtime_failures_report_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep_path = tmp_path / "sweep.json"
    output_dir = tmp_path / "sweep-output"
    sweep_path.write_text(
        json.dumps(
            {
                "sweep_name": "cli_sweep",
                "base_config": {
                    "experiment_name": "network_herding",
                    "seed": 1,
                    "num_agents": 6,
                    "initial_opinion": {"type": "bernoulli", "probability_a": 0.5},
                    "topology": {"type": "cycle"},
                    "scheduler": {"type": "synchronous_rounds", "rounds": 2},
                    "observation_policy": {"type": "neighbor_actions"},
                    "update_policy": {"type": "majority_rule"},
                    "output_dir": str(tmp_path / "ignored"),
                },
                "factors": [{"name": "seed", "path": "seed", "values": [1, 2]}],
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    def raise_runtime_error(_sweep: object) -> object:
        raise OSError("disk full")

    monkeypatch.setattr(cli, "run_sweep", raise_runtime_error)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["sweep", str(sweep_path)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Sweep run failed for" in captured.err
    assert "disk full" in captured.err
    assert "Traceback" not in captured.err
    assert "sweep=" not in captured.out
    assert "summary_csv=" not in captured.out


def test_cli_analyze_writes_report_and_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.test_sweep_analysis import write_analysis_fixture

    sweep_dir = write_analysis_fixture(tmp_path)

    exit_code = cli.main(["analyze", str(sweep_dir)])

    assert exit_code == 0
    output = capsys.readouterr().out.splitlines()
    assert output == [
        "analysis=network_topology_sweep",
        "runs=5",
        "completed=3",
        "failed=1",
        f"output_dir={sweep_dir / 'analysis'}",
        f"report={sweep_dir / 'analysis' / 'report.md'}",
    ]
    assert (sweep_dir / "analysis" / "report.md").exists()
    assert (sweep_dir / "analysis" / "group_summary.csv").exists()
    assert (sweep_dir / "analysis" / "group_summary.json").exists()
    assert (sweep_dir / "analysis" / "failure_summary.csv").exists()


def test_cli_analyze_invalid_output_dir_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing-sweep-output"

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["analyze", str(missing)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr().err
    assert "Analyze failed for" in captured
    assert "sweep output directory does not exist" in captured
    assert "Traceback" not in captured


def test_cli_analyze_missing_artifact_reports_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.test_sweep_analysis import write_analysis_fixture

    sweep_dir = write_analysis_fixture(tmp_path)
    (sweep_dir / "manifest.jsonl").unlink()

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["analyze", str(sweep_dir)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr().err
    assert "Analyze failed for" in captured
    assert "missing required sweep artifact: manifest.jsonl" in captured
    assert "Traceback" not in captured


def test_example_network_topology_sweep_exists_and_is_valid() -> None:
    from society_simulation.sweep_config import expand_sweep, load_sweep_config

    sweep = load_sweep_config("experiments/network_topology_sweep.json")
    runs = expand_sweep(sweep)

    assert sweep.sweep_name == "network_topology_sweep"
    assert len(runs) == 48
