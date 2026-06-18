import csv
from dataclasses import fields
import json
from pathlib import Path

from society_simulation.sweep_analysis import analyze_sweep
from society_simulation.sweep_analysis_artifacts import (
    FAILURE_FIELDS,
    GROUP_SUMMARY_FIELDS,
    SweepAnalysisArtifactPaths,
    write_analysis_artifacts,
)
from tests.test_sweep_analysis import write_analysis_fixture


def field_names(dataclass_type) -> tuple[str, ...]:
    return tuple(field.name for field in fields(dataclass_type))


def test_sweep_analysis_artifact_paths_dataclass_matches_task_api() -> None:
    assert field_names(SweepAnalysisArtifactPaths) == (
        "output_dir",
        "report_path",
        "group_summary_csv_path",
        "group_summary_json_path",
        "failure_summary_csv_path",
    )


def test_write_analysis_artifacts_returns_paths_and_writes_files(tmp_path: Path) -> None:
    result = analyze_sweep(write_analysis_fixture(tmp_path))

    paths = write_analysis_artifacts(result)

    assert paths.output_dir == result.analysis_dir
    assert paths.report_path == result.analysis_dir / "report.md"
    assert paths.group_summary_csv_path == result.analysis_dir / "group_summary.csv"
    assert paths.group_summary_json_path == result.analysis_dir / "group_summary.json"
    assert paths.failure_summary_csv_path == result.analysis_dir / "failure_summary.csv"
    assert paths.output_dir.is_dir()
    assert paths.report_path.exists()
    assert paths.group_summary_csv_path.exists()
    assert paths.group_summary_json_path.exists()
    assert paths.failure_summary_csv_path.exists()


def test_report_includes_overview_toplines_factor_tables_and_failures(
    tmp_path: Path,
) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    paths = write_analysis_artifacts(analyze_sweep(sweep_dir))

    report = paths.report_path.read_text(encoding="utf-8")

    assert "# Sweep Analysis: network_topology_sweep" in report
    assert "## Overview" in report
    assert "- Runs: 5" in report
    assert "- Completed: 3" in report
    assert "- Failed: 1" in report
    assert "## Topline" in report
    assert "- Highest consensus rate: topology=complete (1.0000)" in report
    assert "- Highest polarization: topology=cycle (0.6000)" in report
    assert "- Highest edge disagreement: topology=cycle (0.4000)" in report
    assert "## Factor Summaries" in report
    assert "### seed" in report
    assert "### topology" in report
    assert "### threshold" in report
    assert (
        "| value | runs | completed | failed | consensus_rate | "
        "mean_final_a_fraction | mean_polarization_index | "
        "mean_edge_disagreement_rate |"
    ) in report
    assert (
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    ) in report
    assert (
        "| complete | 2 | 2 | 0 | 1.0000 | 0.9500 | 0.0500 | 0.0250 |"
    ) in report
    assert "| cycle | 3 | 1 | 1 | 0.0000 | 0.5000 | 0.6000 | 0.4000 |" in report
    assert "## Failures" in report
    assert "| run_id | status | error | output_dir |" in report
    assert "| --- | --- | --- | --- |" in report
    assert (
        "| seed-2__topology-cycle__threshold-0_7 | failed | boom | "
        f"{sweep_dir / 'runs' / 'run-4'} |"
    ) in report
    assert (
        "| seed-1__topology-cycle__threshold-0_7 | pending |  | "
        f"{sweep_dir / 'runs' / 'run-5'} |"
    ) in report


def test_group_summary_csv_fields_order_and_numeric_formatting(
    tmp_path: Path,
) -> None:
    paths = write_analysis_artifacts(analyze_sweep(write_analysis_fixture(tmp_path)))

    with paths.group_summary_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == list(GROUP_SUMMARY_FIELDS)
    assert [(row["factor"], row["value"]) for row in rows] == [
        ("seed", "1"),
        ("seed", "2"),
        ("topology", "complete"),
        ("topology", "cycle"),
        ("threshold", "0_55"),
        ("threshold", "0_7"),
    ]
    assert rows[0] == {
        "factor": "seed",
        "value": "1",
        "runs": "3",
        "completed": "2",
        "failed": "0",
        "consensus_rate": "0.5000",
        "mean_final_a_fraction": "0.7500",
        "mean_polarization_index": "0.3000",
        "mean_edge_disagreement_rate": "0.2000",
        "mean_time_to_consensus": "1.0000",
        "mean_opinion_variance": "0.1250",
        "mean_component_count": "1.0000",
    }
    assert rows[3] == {
        "factor": "topology",
        "value": "cycle",
        "runs": "3",
        "completed": "1",
        "failed": "1",
        "consensus_rate": "0.0000",
        "mean_final_a_fraction": "0.5000",
        "mean_polarization_index": "0.6000",
        "mean_edge_disagreement_rate": "0.4000",
        "mean_time_to_consensus": "",
        "mean_opinion_variance": "0.2500",
        "mean_component_count": "1.0000",
    }


def test_group_summary_json_contains_native_values_from_analysis_result(
    tmp_path: Path,
) -> None:
    paths = write_analysis_artifacts(analyze_sweep(write_analysis_fixture(tmp_path)))

    text = paths.group_summary_json_path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert text == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert set(payload) == {
        "sweep_name",
        "runs",
        "completed",
        "failed",
        "groups",
        "toplines",
        "incomplete_runs",
    }
    assert payload["sweep_name"] == "network_topology_sweep"
    assert payload["runs"] == 5
    assert payload["completed"] == 3
    assert payload["failed"] == 1
    assert payload["groups"][0]["factor_name"] == "seed"
    assert payload["groups"][0]["value"] == "1"
    assert payload["groups"][0]["consensus_rate"] == 0.5
    assert payload["groups"][3]["factor_name"] == "topology"
    assert payload["groups"][3]["mean_time_to_consensus"] is None
    assert payload["toplines"]["highest_consensus_rate"] == {
        "name": "highest_consensus_rate",
        "factor_name": "topology",
        "value": "complete",
        "metric_value": 1.0,
    }
    assert payload["toplines"]["highest_polarization"]["metric_value"] == 0.6
    assert [run["status"] for run in payload["incomplete_runs"]] == ["failed", "pending"]
    assert payload["incomplete_runs"][0]["error"] == "boom"


def test_failure_summary_csv_includes_failed_and_pending_rows_in_order(
    tmp_path: Path,
) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    paths = write_analysis_artifacts(analyze_sweep(sweep_dir))

    with paths.failure_summary_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == list(FAILURE_FIELDS)
    assert rows == [
        {
            "run_id": "seed-2__topology-cycle__threshold-0_7",
            "status": "failed",
            "error": "boom",
            "output_dir": str(sweep_dir / "runs" / "run-4"),
        },
        {
            "run_id": "seed-1__topology-cycle__threshold-0_7",
            "status": "pending",
            "error": "",
            "output_dir": str(sweep_dir / "runs" / "run-5"),
        },
    ]


def test_write_analysis_artifacts_rewrites_report_and_json_deterministically(
    tmp_path: Path,
) -> None:
    result = analyze_sweep(write_analysis_fixture(tmp_path))
    paths = write_analysis_artifacts(result)
    report_before = paths.report_path.read_text(encoding="utf-8")
    json_before = paths.group_summary_json_path.read_text(encoding="utf-8")

    rewritten_paths = write_analysis_artifacts(result)

    assert rewritten_paths == paths
    assert paths.report_path.read_text(encoding="utf-8") == report_before
    assert paths.group_summary_json_path.read_text(encoding="utf-8") == json_before
