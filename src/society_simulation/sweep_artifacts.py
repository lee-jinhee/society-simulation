from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path

from society_simulation.sweep_config import MaterializedRun, SweepConfig

AD_METRIC_FIELDS = (
    "paid_impression_count",
    "unique_paid_reach",
    "organic_ad_impression_count",
    "unique_organic_ad_reach",
    "unique_total_ad_reach",
    "relevant_paid_reach",
    "relevant_total_reach",
    "mean_ad_frequency",
    "max_ad_frequency",
    "frequency_cap_hit_count",
    "ad_like_count",
    "advertiser_follow_count",
    "ad_dm_count",
    "ad_generated_post_count",
    "ad_negative_action_count",
    "paid_to_organic_spillover_rate",
    "ad_delivery_exhausted_budget",
    "ad_delivery_remaining_budget",
    "burn_in_action_mean",
    "burn_in_follow_churn",
    "burn_in_exposure_diversity",
)
NUMERIC_AD_MEAN_FIELDS = tuple(
    field for field in AD_METRIC_FIELDS if field != "ad_delivery_exhausted_budget"
)

METRIC_FIELDS = (
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
    "final_private_stance_mean",
    "final_public_stance_mean",
    "final_private_public_gap",
    "final_private_stance_variance",
    "final_public_stance_variance",
    "final_mean_confidence",
    "final_mean_salience",
    "message_count",
    "agent_count",
    "day_count",
    "experiment_family",
    "user_count",
    "post_count",
    "feed_impression_count",
    "action_count",
    "like_count",
    "dm_count",
    "follow_count",
    "unfollow_count",
    "initial_follow_edge_count",
    "final_follow_edge_count",
    "follow_edge_delta",
    "new_follow_edge_count",
    "removed_follow_edge_count",
    "final_stance_mean",
    "final_stance_variance",
    "exposure_diversity",
    "states_recorded",
    *AD_METRIC_FIELDS,
)
NUMERIC_MEAN_FIELDS = (
    "final_a_fraction",
    "time_to_consensus",
    "polarization_index",
    "opinion_variance",
    "mean_belief",
    "edge_disagreement_rate",
    "component_count",
    "final_private_stance_mean",
    "final_public_stance_mean",
    "final_private_public_gap",
    "final_private_stance_variance",
    "final_public_stance_variance",
    "final_mean_confidence",
    "final_mean_salience",
    "message_count",
    "agent_count",
    "day_count",
    "user_count",
    "post_count",
    "feed_impression_count",
    "action_count",
    "like_count",
    "dm_count",
    "follow_count",
    "unfollow_count",
    "initial_follow_edge_count",
    "final_follow_edge_count",
    "follow_edge_delta",
    "new_follow_edge_count",
    "removed_follow_edge_count",
    "final_stance_mean",
    "final_stance_variance",
    "exposure_diversity",
    "states_recorded",
    *NUMERIC_AD_MEAN_FIELDS,
)
MANIFEST_FIELDS = (
    "run_id",
    "labels",
    "experiment_name",
    "output_dir",
    "status",
    "error",
    *METRIC_FIELDS,
)


@dataclass(frozen=True)
class SweepRunRecord:
    run_id: str
    labels: dict[str, str]
    experiment_name: str
    output_dir: str | os.PathLike[str]
    status: str
    error: str | None
    metrics: dict[str, object]


@dataclass(frozen=True)
class SweepArtifactPaths:
    output_dir: Path
    sweep_config_path: Path
    manifest_path: Path
    summary_csv_path: Path
    summary_json_path: Path


def write_sweep_artifacts(
    sweep: SweepConfig,
    planned_runs: Iterable[MaterializedRun],
    records: Iterable[SweepRunRecord],
) -> SweepArtifactPaths:
    output_dir = Path(sweep.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = SweepArtifactPaths(
        output_dir=output_dir,
        sweep_config_path=output_dir / "sweep_config.json",
        manifest_path=output_dir / "manifest.jsonl",
        summary_csv_path=output_dir / "summary.csv",
        summary_json_path=output_dir / "summary.json",
    )

    rows = _artifact_rows(sweep, planned_runs, records)
    paths.sweep_config_path.write_text(
        json.dumps(sweep.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_manifest(paths.manifest_path, rows)
    _write_summary_csv(paths.summary_csv_path, sweep, rows)
    _write_summary_json(paths.summary_json_path, sweep, rows)
    return paths


def _artifact_rows(
    sweep: SweepConfig,
    planned_runs: Iterable[MaterializedRun],
    records: Iterable[SweepRunRecord],
) -> list[dict[str, object]]:
    records_by_id = {record.run_id: record for record in records}
    rows: list[dict[str, object]] = []
    for planned_run in planned_runs:
        record = records_by_id.get(planned_run.run_id)
        labels = dict(record.labels if record is not None else planned_run.labels)
        row: dict[str, object] = {
            "run_id": planned_run.run_id,
            "labels": labels,
            "experiment_name": (
                record.experiment_name
                if record is not None
                else str(planned_run.config["experiment_name"])
            ),
            "output_dir": (
                str(record.output_dir)
                if record is not None
                else str(planned_run.config["output_dir"])
            ),
            "status": record.status if record is not None else "pending",
            "error": record.error if record is not None else None,
        }
        row.update(_flatten_metrics(record.metrics if record is not None else {}))
        rows.append(row)
    return rows


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            manifest_row = {field: row[field] for field in MANIFEST_FIELDS}
            handle.write(json.dumps(manifest_row, sort_keys=True) + "\n")


def _write_summary_csv(
    path: Path,
    sweep: SweepConfig,
    rows: list[dict[str, object]],
) -> None:
    factor_names = [factor.name for factor in sweep.factors]
    fieldnames = [
        "run_id",
        *factor_names,
        "experiment_name",
        "output_dir",
        "status",
        "error",
        *METRIC_FIELDS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_csv_row(sweep, row) for row in rows)


def _write_summary_json(
    path: Path,
    sweep: SweepConfig,
    rows: list[dict[str, object]],
) -> None:
    summary = {
        "sweep_name": sweep.sweep_name,
        "runs": len(rows),
        "completed": _count_status(rows, "completed"),
        "failed": _count_status(rows, "failed"),
        "metric_means": _metric_means(rows),
        "groups": _factor_groups(sweep, rows),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _factor_groups(
    sweep: SweepConfig,
    rows: list[dict[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    groups: dict[str, dict[str, dict[str, object]]] = {}
    for factor in sweep.factors:
        rows_by_label: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            rows_by_label.setdefault(_labels(row)[factor.name], []).append(row)

        factor_groups: dict[str, dict[str, object]] = {}
        for label, group_rows in rows_by_label.items():
            factor_groups[label] = {
                "runs": len(group_rows),
                "completed": _count_status(group_rows, "completed"),
                "failed": _count_status(group_rows, "failed"),
                "metric_means": _metric_means(group_rows),
            }
        groups[factor.name] = factor_groups
    return groups


def _csv_row(sweep: SweepConfig, row: dict[str, object]) -> dict[str, object]:
    labels = _labels(row)
    return {
        "run_id": row["run_id"],
        **{factor.name: labels[factor.name] for factor in sweep.factors},
        "experiment_name": row["experiment_name"],
        "output_dir": row["output_dir"],
        "status": row["status"],
        "error": row["error"],
        **{field: row[field] for field in METRIC_FIELDS},
    }


def _labels(row: dict[str, object]) -> dict[str, str]:
    labels = row["labels"]
    assert isinstance(labels, dict)
    return labels


def _flatten_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    flattened = {field: None for field in METRIC_FIELDS}
    action_counts = metrics.get("final_action_counts")
    fallback_action_counts = metrics.get("action_counts")
    if not isinstance(action_counts, Mapping):
        action_counts = fallback_action_counts
    for action in ("A", "B"):
        value = action_counts.get(action) if isinstance(action_counts, Mapping) else None
        if value is None and isinstance(fallback_action_counts, Mapping):
            value = fallback_action_counts.get(action)
        flattened[f"final_action_counts_{action}"] = value
    for field in METRIC_FIELDS:
        if field.startswith("final_action_counts_"):
            continue
        flattened[field] = metrics.get(field)
    return flattened


def _metric_means(rows: list[dict[str, object]]) -> dict[str, float]:
    means: dict[str, float] = {}
    completed_rows = [row for row in rows if row["status"] == "completed"]
    for field in NUMERIC_MEAN_FIELDS:
        values = [_numeric_metric(row[field]) for row in completed_rows]
        numeric_values = [value for value in values if value is not None]
        if numeric_values:
            means[field] = sum(numeric_values) / len(numeric_values)
    return means


def _numeric_metric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _count_status(rows: list[dict[str, object]], status: str) -> int:
    return sum(1 for row in rows if row["status"] == status)
