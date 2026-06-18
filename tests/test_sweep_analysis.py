import csv
from dataclasses import fields
import json
from pathlib import Path

import pytest

from society_simulation.sweep_analysis import (
    GroupSummary,
    IncompleteRun,
    SweepAnalysisResult,
    ToplineEntry,
    analyze_sweep,
)


SUMMARY_FIELDS = [
    "run_id",
    "seed",
    "topology",
    "threshold",
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


def write_analysis_fixture(tmp_path: Path) -> Path:
    sweep_dir = tmp_path / "network_topology_sweep"
    sweep_dir.mkdir()
    (sweep_dir / "sweep_config.json").write_text(
        json.dumps(
            {
                "sweep_name": "network_topology_sweep",
                "base_config": {"experiment_name": "network_herding"},
                "factors": [
                    {"name": "seed", "path": "seed", "values": [1, 2]},
                    {
                        "name": "topology",
                        "values": [
                            {"label": "complete", "overrides": {"topology": {"type": "complete"}}},
                            {"label": "cycle", "overrides": {"topology": {"type": "cycle"}}},
                        ],
                    },
                    {
                        "name": "threshold",
                        "path": "update_policy.adoption_threshold",
                        "values": [0.55, 0.7],
                    },
                ],
                "output_dir": str(sweep_dir),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [
        {
            "run_id": "seed-1__topology-complete__threshold-0_55",
            "seed": "1",
            "topology": "complete",
            "threshold": "0_55",
            "experiment_name": "network_herding",
            "output_dir": str(sweep_dir / "runs" / "run-1"),
            "status": "completed",
            "error": "",
            "final_action_counts_A": "30",
            "final_action_counts_B": "0",
            "final_a_fraction": "1.0",
            "consensus_reached": "True",
            "consensus_action": "A",
            "time_to_consensus": "1",
            "polarization_index": "0.0",
            "opinion_variance": "0.0",
            "mean_belief": "1.0",
            "edge_disagreement_rate": "0.0",
            "component_count": "1",
        },
        {
            "run_id": "seed-2__topology-complete__threshold-0_7",
            "seed": "2",
            "topology": "complete",
            "threshold": "0_7",
            "experiment_name": "network_herding",
            "output_dir": str(sweep_dir / "runs" / "run-2"),
            "status": "completed",
            "error": "",
            "final_action_counts_A": "27",
            "final_action_counts_B": "3",
            "final_a_fraction": "0.9",
            "consensus_reached": "True",
            "consensus_action": "A",
            "time_to_consensus": "2",
            "polarization_index": "0.1",
            "opinion_variance": "0.02",
            "mean_belief": "0.9",
            "edge_disagreement_rate": "0.05",
            "component_count": "1",
        },
        {
            "run_id": "seed-1__topology-cycle__threshold-0_55",
            "seed": "1",
            "topology": "cycle",
            "threshold": "0_55",
            "experiment_name": "network_herding",
            "output_dir": str(sweep_dir / "runs" / "run-3"),
            "status": "completed",
            "error": "",
            "final_action_counts_A": "15",
            "final_action_counts_B": "15",
            "final_a_fraction": "0.5",
            "consensus_reached": "False",
            "consensus_action": "",
            "time_to_consensus": "",
            "polarization_index": "0.6",
            "opinion_variance": "0.25",
            "mean_belief": "0.5",
            "edge_disagreement_rate": "0.4",
            "component_count": "1",
        },
        {
            "run_id": "seed-2__topology-cycle__threshold-0_7",
            "seed": "2",
            "topology": "cycle",
            "threshold": "0_7",
            "experiment_name": "network_herding",
            "output_dir": str(sweep_dir / "runs" / "run-4"),
            "status": "failed",
            "error": "boom",
            "final_action_counts_A": "",
            "final_action_counts_B": "",
            "final_a_fraction": "",
            "consensus_reached": "",
            "consensus_action": "",
            "time_to_consensus": "",
            "polarization_index": "",
            "opinion_variance": "",
            "mean_belief": "",
            "edge_disagreement_rate": "",
            "component_count": "",
        },
        {
            "run_id": "seed-1__topology-cycle__threshold-0_7",
            "seed": "1",
            "topology": "cycle",
            "threshold": "0_7",
            "experiment_name": "network_herding",
            "output_dir": str(sweep_dir / "runs" / "run-5"),
            "status": "pending",
            "error": "",
            "final_action_counts_A": "",
            "final_action_counts_B": "",
            "final_a_fraction": "",
            "consensus_reached": "",
            "consensus_action": "",
            "time_to_consensus": "",
            "polarization_index": "",
            "opinion_variance": "",
            "mean_belief": "",
            "edge_disagreement_rate": "",
            "component_count": "",
        },
    ]
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (sweep_dir / "summary.json").write_text(
        json.dumps(
            {
                "sweep_name": "network_topology_sweep",
                "runs": 5,
                "completed": 3,
                "failed": 1,
                "metric_means": {},
                "groups": {},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (sweep_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return sweep_dir


def group_by(result, factor_name: str, value: str):
    for group in result.group_summaries:
        if group.factor_name == factor_name and group.value == value:
            return group
    raise AssertionError(f"group not found: {factor_name}={value}")


def field_names(dataclass_type) -> tuple[str, ...]:
    return tuple(field.name for field in fields(dataclass_type))


def test_sweep_analysis_dataclasses_match_task_api() -> None:
    assert field_names(GroupSummary) == (
        "factor_name",
        "value",
        "runs",
        "completed",
        "failed",
        "consensus_rate",
        "mean_final_a_fraction",
        "mean_polarization_index",
        "mean_edge_disagreement_rate",
        "mean_time_to_consensus",
        "mean_opinion_variance",
        "mean_component_count",
    )
    assert field_names(IncompleteRun) == ("run_id", "status", "error", "output_dir")
    assert field_names(ToplineEntry) == ("name", "factor_name", "value", "metric_value")
    assert field_names(SweepAnalysisResult) == (
        "sweep_name",
        "source_dir",
        "analysis_dir",
        "runs",
        "completed",
        "failed",
        "factor_names",
        "group_summaries",
        "toplines",
        "incomplete_runs",
    )


def test_analyze_sweep_computes_grouped_metrics_and_toplines(tmp_path: Path) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)

    result = analyze_sweep(sweep_dir)

    assert result.sweep_name == "network_topology_sweep"
    assert result.source_dir == sweep_dir
    assert result.runs == 5
    assert result.completed == 3
    assert result.failed == 1
    assert result.analysis_dir == sweep_dir / "analysis"
    assert result.factor_names == ("seed", "topology", "threshold")

    complete = group_by(result, "topology", "complete")
    assert complete.runs == 2
    assert complete.completed == 2
    assert complete.failed == 0
    assert complete.consensus_rate == 1.0
    assert complete.mean_final_a_fraction == pytest.approx(0.95)
    assert complete.mean_polarization_index == pytest.approx(0.05)
    assert complete.mean_edge_disagreement_rate == pytest.approx(0.025)
    assert complete.mean_time_to_consensus == pytest.approx(1.5)
    assert complete.metric("mean_final_a_fraction") == pytest.approx(0.95)
    assert complete.to_dict()["factor_name"] == "topology"

    cycle = group_by(result, "topology", "cycle")
    assert cycle.runs == 3
    assert cycle.completed == 1
    assert cycle.failed == 1
    assert cycle.consensus_rate == 0.0
    assert cycle.mean_final_a_fraction == 0.5
    assert cycle.mean_polarization_index == 0.6
    assert cycle.mean_time_to_consensus is None

    assert [group.factor_name for group in result.group_summaries[:2]] == [
        "seed",
        "seed",
    ]
    assert [group.value for group in result.group_summaries[:2]] == ["1", "2"]
    assert result.toplines["highest_consensus_rate"].factor_name == "topology"
    assert result.toplines["highest_consensus_rate"].value == "complete"
    assert result.toplines["highest_polarization"].factor_name == "topology"
    assert result.toplines["highest_polarization"].value == "cycle"
    assert "highest_edge_disagreement" in result.toplines
    assert result.toplines["highest_edge_disagreement"].name == "highest_edge_disagreement"
    assert result.toplines["highest_edge_disagreement"].value == "cycle"
    assert [run.status for run in result.incomplete_runs] == ["failed", "pending"]
    assert result.incomplete_runs[0].to_dict()["error"] == "boom"


def test_analyze_sweep_rejects_missing_required_file(tmp_path: Path) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    (sweep_dir / "summary.csv").unlink()

    with pytest.raises(ValueError, match="missing required sweep artifact: summary.csv"):
        analyze_sweep(sweep_dir)


def test_analyze_sweep_rejects_summary_count_mismatch(tmp_path: Path) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    (sweep_dir / "summary.json").write_text(
        json.dumps({"sweep_name": "network_topology_sweep", "runs": 99, "completed": 3, "failed": 1}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="summary.csv row count 5 does not match summary.json runs 99"):
        analyze_sweep(sweep_dir)


def test_analyze_sweep_rejects_missing_factor_column(tmp_path: Path) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    rows = list(csv.DictReader((sweep_dir / "summary.csv").open(newline="", encoding="utf-8")))
    fieldnames = [field for field in SUMMARY_FIELDS if field != "threshold"]
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != "threshold"} for row in rows)

    with pytest.raises(ValueError, match="summary.csv is missing required column: threshold"):
        analyze_sweep(sweep_dir)
