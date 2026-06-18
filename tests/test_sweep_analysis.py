import csv
from dataclasses import fields
import json
import math
from pathlib import Path
import re

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
                "base_config": {
                    "experiment_name": "network_herding",
                    "seed": 1,
                    "num_agents": 30,
                    "initial_opinion": {"type": "bernoulli", "probability_a": 0.5},
                    "topology": {"type": "cycle"},
                    "scheduler": {"type": "synchronous_rounds", "rounds": 2},
                    "observation_policy": {"type": "neighbor_actions"},
                    "update_policy": {"type": "threshold", "adoption_threshold": 0.6},
                    "output_dir": str(sweep_dir / "ignored"),
                },
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


def read_manifest_entries(sweep_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (sweep_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_manifest_entries(sweep_dir: Path, entries: list[dict[str, object]]) -> None:
    with (sweep_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


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


def test_analyze_sweep_rejects_missing_output_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing-sweep"

    with pytest.raises(
        ValueError,
        match=re.escape(f"sweep output directory does not exist: {missing_dir}"),
    ):
        analyze_sweep(missing_dir)


@pytest.mark.parametrize("artifact_name", ["sweep_config.json", "summary.json"])
def test_analyze_sweep_wraps_malformed_json_artifacts(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    (sweep_dir / artifact_name).write_text("{not-json", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"malformed JSON artifact {re.escape(artifact_name)}:",
    ):
        analyze_sweep(sweep_dir)


def test_analyze_sweep_wraps_malformed_manifest_json(tmp_path: Path) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    (sweep_dir / "manifest.jsonl").write_text("{not-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed JSON artifact manifest\\.jsonl:"):
        analyze_sweep(sweep_dir)


def test_analyze_sweep_rejects_manifest_count_mismatch(tmp_path: Path) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    entries = read_manifest_entries(sweep_dir)
    write_manifest_entries(sweep_dir, entries[:-1])

    with pytest.raises(
        ValueError,
        match="manifest\\.jsonl entry count 4 does not match summary\\.csv row count 5",
    ):
        analyze_sweep(sweep_dir)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("run_id", "different-run"),
        ("status", "failed"),
        ("error", "changed error"),
        ("output_dir", "/tmp/different-output"),
    ],
)
def test_analyze_sweep_rejects_manifest_row_mismatch(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    entries = read_manifest_entries(sweep_dir)
    run_id = str(entries[0]["run_id"])
    entries[0][field] = replacement
    write_manifest_entries(sweep_dir, entries)

    with pytest.raises(
        ValueError,
        match=rf"manifest\.jsonl row 1 .*{field}.*{re.escape(run_id)}",
    ):
        analyze_sweep(sweep_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda entry: entry.pop("run_id"),
            "manifest.jsonl row 1 is missing required field: run_id",
        ),
        (
            lambda entry: entry.update({"status": 123}),
            "manifest.jsonl row 1 field status must be a string",
        ),
        (
            lambda entry: entry.update({"error": 123}),
            "manifest.jsonl row 1 field error must be a string or null",
        ),
    ],
)
def test_analyze_sweep_rejects_invalid_manifest_required_fields(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    entries = read_manifest_entries(sweep_dir)
    mutation(entries[0])
    write_manifest_entries(sweep_dir, entries)

    with pytest.raises(ValueError, match=re.escape(message)):
        analyze_sweep(sweep_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: config.update(
                {
                    "factors": [
                        {"name": "seed", "path": "seed", "values": [1]},
                        {"name": "seed", "path": "seed", "values": [2]},
                    ]
                }
            ),
            "invalid sweep_config.json: factor names must be unique",
        ),
        (
            lambda config: config["factors"][0].pop("values"),
            "invalid sweep_config.json: factor seed values must be a non-empty list",
        ),
    ],
)
def test_analyze_sweep_rejects_invalid_sweep_config_schema_via_parser(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    config = json.loads((sweep_dir / "sweep_config.json").read_text(encoding="utf-8"))
    mutation(config)
    (sweep_dir / "sweep_config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(message)):
        analyze_sweep(sweep_dir)


def test_analyze_sweep_rejects_malformed_summary_csv(tmp_path: Path) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    (sweep_dir / "summary.csv").write_text('"unterminated\n', encoding="utf-8")

    with pytest.raises(ValueError, match="malformed CSV artifact summary\\.csv:"):
        analyze_sweep(sweep_dir)


@pytest.mark.parametrize(
    ("row_transform", "message"),
    [
        (
            lambda row: row[:-1],
            "malformed CSV artifact summary.csv: row 2 has missing cells",
        ),
        (
            lambda row: [*row, "extra"],
            "malformed CSV artifact summary.csv: row 2 has extra cells",
        ),
    ],
)
def test_analyze_sweep_rejects_summary_csv_row_shape_errors(
    tmp_path: Path,
    row_transform,
    message: str,
) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    with (sweep_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    rows[1] = row_transform(rows[1])
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    with pytest.raises(ValueError, match=re.escape(message)):
        analyze_sweep(sweep_dir)


def test_analyze_sweep_rejects_summary_count_mismatch(tmp_path: Path) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    (sweep_dir / "summary.json").write_text(
        json.dumps({"sweep_name": "network_topology_sweep", "runs": 99, "completed": 3, "failed": 1}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="summary.csv row count 5 does not match summary.json runs 99"):
        analyze_sweep(sweep_dir)


@pytest.mark.parametrize(
    ("status", "expected", "actual"),
    [("completed", 99, 3), ("failed", 99, 1)],
)
def test_analyze_sweep_rejects_summary_status_count_mismatch(
    tmp_path: Path,
    status: str,
    expected: int,
    actual: int,
) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    summary = {"sweep_name": "network_topology_sweep", "runs": 5, "completed": 3, "failed": 1}
    summary[status] = expected
    (sweep_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            f"summary.csv {status} count {actual} "
            f"does not match summary.json {status} {expected}"
        ),
    ):
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


def test_analyze_sweep_allows_analysis_unused_summary_columns_to_be_absent(
    tmp_path: Path,
) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    unused_fields = {
        "final_action_counts_A",
        "final_action_counts_B",
        "consensus_action",
        "mean_belief",
    }
    rows = list(csv.DictReader((sweep_dir / "summary.csv").open(newline="", encoding="utf-8")))
    fieldnames = [field for field in SUMMARY_FIELDS if field not in unused_fields]
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key in fieldnames} for row in rows)

    result = analyze_sweep(sweep_dir)

    assert result.runs == 5
    assert group_by(result, "topology", "complete").mean_final_a_fraction == pytest.approx(0.95)


def test_analyze_sweep_ignores_unparsable_metric_cells_in_completed_rows(
    tmp_path: Path,
) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    rows = list(csv.DictReader((sweep_dir / "summary.csv").open(newline="", encoding="utf-8")))
    rows[0]["final_a_fraction"] = "not-a-number"
    rows[0]["polarization_index"] = "not-a-number"
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    complete = group_by(analyze_sweep(sweep_dir), "topology", "complete")

    assert complete.mean_final_a_fraction == pytest.approx(0.9)
    assert complete.mean_polarization_index == pytest.approx(0.1)


def test_analyze_sweep_ignores_non_finite_metric_cells(tmp_path: Path) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    rows = list(csv.DictReader((sweep_dir / "summary.csv").open(newline="", encoding="utf-8")))
    rows[0]["final_a_fraction"] = "nan"
    rows[0]["polarization_index"] = "inf"
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    result = analyze_sweep(sweep_dir)
    complete = group_by(result, "topology", "complete")

    assert complete.mean_final_a_fraction == pytest.approx(0.9)
    assert complete.mean_polarization_index == pytest.approx(0.1)
    assert result.toplines["highest_polarization"].factor_name == "topology"
    assert result.toplines["highest_polarization"].value == "cycle"
    assert all(math.isfinite(topline.metric_value) for topline in result.toplines.values())


def test_toplines_ignore_seed_when_non_seed_groups_have_metric_data(
    tmp_path: Path,
) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    rows = list(csv.DictReader((sweep_dir / "summary.csv").open(newline="", encoding="utf-8")))
    for row in rows:
        if row["status"] == "completed":
            row["polarization_index"] = "0.01"
    rows[0]["polarization_index"] = "0.99"
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    result = analyze_sweep(sweep_dir)

    assert group_by(result, "seed", "1").mean_polarization_index == pytest.approx(0.5)
    assert result.toplines["highest_polarization"].factor_name == "topology"
    assert result.toplines["highest_polarization"].value == "complete"


@pytest.mark.parametrize("name", ["factor_name", "runs", "does_not_exist"])
def test_group_summary_metric_rejects_unknown_or_nonmetric_names(
    tmp_path: Path,
    name: str,
) -> None:
    sweep_dir = write_analysis_fixture(tmp_path)
    group = group_by(analyze_sweep(sweep_dir), "topology", "complete")

    with pytest.raises(ValueError, match=f"unknown group summary metric: {name}"):
        group.metric(name)
