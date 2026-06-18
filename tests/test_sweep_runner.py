import csv
from pathlib import Path
from types import SimpleNamespace

from society_simulation.sweep_config import parse_sweep_config
from society_simulation.sweep_runner import run_sweep
from tests.test_sweep_config import valid_sweep_dict


def test_run_sweep_executes_all_materialized_runs(tmp_path: Path) -> None:
    data = valid_sweep_dict(tmp_path)
    data["factors"] = [
        {"name": "seed", "path": "seed", "values": [1, 2]},
        {
            "name": "topology",
            "values": [
                {"label": "cycle", "overrides": {"topology": {"type": "cycle"}}}
            ],
        },
    ]
    sweep = parse_sweep_config(data)

    result = run_sweep(sweep)

    assert result.sweep_name == "network_topology_sweep"
    assert result.runs == 2
    assert result.completed == 2
    assert result.failed == 0
    assert result.output_dir == Path(sweep.output_dir)
    assert result.summary_csv_path.exists()
    assert result.summary_json_path.exists()
    assert result.manifest_path.exists()
    for record in result.records:
        assert Path(str(record.output_dir)).exists()
        assert (Path(str(record.output_dir)) / "metrics.json").exists()

    with result.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["status"] == "completed"


def test_run_sweep_records_runtime_failures_and_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data = valid_sweep_dict(tmp_path)
    data["factors"] = [{"name": "seed", "path": "seed", "values": [1, 2]}]
    sweep = parse_sweep_config(data)
    calls: list[int] = []

    def fake_run_experiment(config: object) -> object:
        seed = config.seed
        calls.append(seed)
        if seed == 1:
            raise ValueError("seed one failed")
        return SimpleNamespace(
            metrics={
                "final_action_counts": {"A": 1, "B": 5},
                "final_a_fraction": 1 / 6,
                "consensus_reached": False,
                "consensus_action": None,
                "time_to_consensus": None,
                "polarization_index": 0.0,
                "opinion_variance": 0.0,
                "mean_belief": 0.4,
                "edge_disagreement_rate": 0.0,
                "component_count": 1,
            },
            output_dir=Path(config.output_dir),
        )

    monkeypatch.setattr(
        "society_simulation.sweep_runner.run_experiment",
        fake_run_experiment,
    )

    result = run_sweep(sweep)

    assert calls == [1, 2]
    assert result.runs == 2
    assert result.completed == 1
    assert result.failed == 1
    assert result.records[0].status == "failed"
    assert result.records[0].labels == {"seed": "1"}
    assert result.records[0].error == "seed one failed"
    assert result.records[1].status == "completed"
    assert result.manifest_path.exists()
    assert result.summary_csv_path.exists()

    with result.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["failed", "completed"]
    assert [row["seed"] for row in rows] == ["1", "2"]
