import csv
import json
from pathlib import Path

from society_simulation.sweep_artifacts import SweepRunRecord, write_sweep_artifacts
from society_simulation.sweep_config import expand_sweep, parse_sweep_config
from tests.test_sweep_config import valid_sweep_dict


def test_write_sweep_artifacts_writes_manifest_csv_and_summary_json(
    tmp_path: Path,
) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)
    records = (
        SweepRunRecord(
            run_id=planned_runs[0].run_id,
            labels=planned_runs[0].labels,
            experiment_name="network_herding",
            output_dir=planned_runs[0].config["output_dir"],
            status="completed",
            error=None,
            metrics={
                "final_action_counts": {"A": 4, "B": 2},
                "final_a_fraction": 0.6666666667,
                "consensus_reached": False,
                "consensus_action": None,
                "time_to_consensus": None,
                "polarization_index": 0.2,
                "opinion_variance": 0.1,
                "mean_belief": 0.62,
                "edge_disagreement_rate": 0.25,
                "component_count": 1,
            },
        ),
        SweepRunRecord(
            run_id=planned_runs[1].run_id,
            labels=planned_runs[1].labels,
            experiment_name="network_herding",
            output_dir=planned_runs[1].config["output_dir"],
            status="failed",
            error="boom",
            metrics={},
        ),
    )

    paths = write_sweep_artifacts(sweep, planned_runs, records)

    assert paths.output_dir == Path(sweep.output_dir)
    assert paths.manifest_path.exists()
    assert paths.summary_csv_path.exists()
    assert paths.summary_json_path.exists()
    assert paths.sweep_config_path.exists()

    manifest_lines = paths.manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == len(planned_runs)
    assert json.loads(manifest_lines[0])["status"] == "completed"
    assert json.loads(manifest_lines[1])["error"] == "boom"
    assert json.loads(manifest_lines[2])["status"] == "pending"

    with paths.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == [
        "run_id",
        "seed",
        "initial_a",
        "topology",
        "experiment_name",
        "output_dir",
        "status",
        "error",
        "final_action_counts_A",
        "final_action_counts_B",
        "final_a_fraction",
        "consensus_reached",
        "consensus_action",
        "time_to_consensus",
        "polarization_index",
        "opinion_variance",
        "mean_belief",
        "edge_disagreement_rate",
        "component_count",
    ]
    assert rows[0]["run_id"] == planned_runs[0].run_id
    assert rows[0]["status"] == "completed"
    assert rows[0]["final_action_counts_A"] == "4"
    assert rows[1]["status"] == "failed"
    assert rows[1]["error"] == "boom"
    assert rows[1]["final_a_fraction"] == ""

    summary = json.loads(paths.summary_json_path.read_text(encoding="utf-8"))
    assert summary["sweep_name"] == "network_topology_sweep"
    assert summary["runs"] == len(planned_runs)
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["metric_means"]["final_a_fraction"] == 0.6666666667
    assert summary["groups"]["seed"]["1"]["completed"] == 1


def test_write_sweep_artifacts_includes_all_planned_runs_when_records_missing(
    tmp_path: Path,
) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)

    paths = write_sweep_artifacts(sweep, planned_runs, records=())

    with paths.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(planned_runs)
    assert rows[0]["status"] == "pending"


def test_write_sweep_artifacts_writes_sweep_config_json(tmp_path: Path) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)

    paths = write_sweep_artifacts(sweep, planned_runs, records=())

    text = paths.sweep_config_path.read_text(encoding="utf-8")
    assert json.loads(text) == sweep.to_dict()
    assert text.splitlines()[1] == '  "base_config": {'


def test_write_sweep_artifacts_serializes_record_output_dir_paths(
    tmp_path: Path,
) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)
    output_dir = Path(planned_runs[0].config["output_dir"])

    paths = write_sweep_artifacts(
        sweep,
        planned_runs,
        records=(
            SweepRunRecord(
                run_id=planned_runs[0].run_id,
                labels=planned_runs[0].labels,
                experiment_name="network_herding",
                output_dir=output_dir,
                status="completed",
                error=None,
                metrics={},
            ),
        ),
    )

    manifest_row = json.loads(paths.manifest_path.read_text(encoding="utf-8").splitlines()[0])
    assert manifest_row["output_dir"] == str(output_dir)
    assert isinstance(manifest_row["output_dir"], str)
    with paths.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["output_dir"] == str(output_dir)


def test_summary_metric_means_use_completed_rows_and_continuous_fields_only(
    tmp_path: Path,
) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)
    records = (
        SweepRunRecord(
            run_id=planned_runs[0].run_id,
            labels=planned_runs[0].labels,
            experiment_name="network_herding",
            output_dir=planned_runs[0].config["output_dir"],
            status="completed",
            error=None,
            metrics={
                "final_action_counts": {"A": 4, "B": 2},
                "final_a_fraction": 0.75,
                "consensus_reached": True,
                "component_count": 1,
            },
        ),
        SweepRunRecord(
            run_id=planned_runs[1].run_id,
            labels=planned_runs[1].labels,
            experiment_name="network_herding",
            output_dir=planned_runs[1].config["output_dir"],
            status="failed",
            error="boom",
            metrics={
                "final_action_counts": {"A": 100, "B": 100},
                "final_a_fraction": 0.0,
                "consensus_reached": False,
                "component_count": 9,
            },
        ),
    )

    paths = write_sweep_artifacts(sweep, planned_runs, records)

    summary = json.loads(paths.summary_json_path.read_text(encoding="utf-8"))
    assert summary["metric_means"]["final_a_fraction"] == 0.75
    assert summary["metric_means"]["component_count"] == 1.0
    assert "consensus_reached" not in summary["metric_means"]
    assert "final_action_counts_A" not in summary["metric_means"]
    assert summary["groups"]["seed"]["1"]["metric_means"]["final_a_fraction"] == 0.75
    assert summary["groups"]["seed"]["1"]["metric_means"]["component_count"] == 1.0


def test_write_sweep_artifacts_accepts_action_counts_metric_fallback(
    tmp_path: Path,
) -> None:
    sweep = parse_sweep_config(valid_sweep_dict(tmp_path))
    planned_runs = expand_sweep(sweep)
    records = (
        SweepRunRecord(
            run_id=planned_runs[0].run_id,
            labels=planned_runs[0].labels,
            experiment_name="network_herding",
            output_dir=planned_runs[0].config["output_dir"],
            status="completed",
            error=None,
            metrics={"action_counts": {"A": 1, "B": 2}},
        ),
    )

    paths = write_sweep_artifacts(sweep, planned_runs, records)

    with paths.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["final_action_counts_A"] == "1"
    assert rows[0]["final_action_counts_B"] == "2"
