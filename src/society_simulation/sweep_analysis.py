from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from society_simulation.sweep_config import SweepConfig, parse_sweep_config


REQUIRED_ARTIFACTS = (
    "sweep_config.json",
    "summary.csv",
    "summary.json",
    "manifest.jsonl",
)
ANALYSIS_METRIC_COLUMNS = (
    "final_a_fraction",
    "consensus_reached",
    "time_to_consensus",
    "polarization_index",
    "opinion_variance",
    "edge_disagreement_rate",
    "component_count",
)
AD_NUMERIC_MEAN_FIELDS = (
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
    "ad_delivery_remaining_budget",
    "burn_in_action_mean",
    "burn_in_follow_churn",
    "burn_in_exposure_diversity",
)
NUMERIC_MEAN_FIELDS = (
    "final_a_fraction",
    "time_to_consensus",
    "polarization_index",
    "opinion_variance",
    "edge_disagreement_rate",
    "component_count",
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
    *AD_NUMERIC_MEAN_FIELDS,
)
AD_GROUP_SUMMARY_METRICS = tuple(f"mean_{field}" for field in AD_NUMERIC_MEAN_FIELDS)
GROUP_SUMMARY_METRICS = (
    "consensus_rate",
    "mean_final_a_fraction",
    "mean_polarization_index",
    "mean_edge_disagreement_rate",
    "mean_time_to_consensus",
    "mean_opinion_variance",
    "mean_component_count",
    "mean_user_count",
    "mean_post_count",
    "mean_feed_impression_count",
    "mean_action_count",
    "mean_like_count",
    "mean_dm_count",
    "mean_follow_count",
    "mean_unfollow_count",
    "mean_initial_follow_edge_count",
    "mean_final_follow_edge_count",
    "mean_follow_edge_delta",
    "mean_new_follow_edge_count",
    "mean_removed_follow_edge_count",
    "mean_final_stance_mean",
    "mean_final_stance_variance",
    "mean_exposure_diversity",
    "mean_states_recorded",
    *AD_GROUP_SUMMARY_METRICS,
)
MANIFEST_COMPARE_FIELDS = ("run_id", "status", "error", "output_dir")


@dataclass(frozen=True)
class GroupSummary:
    factor_name: str
    value: str
    runs: int
    completed: int
    failed: int
    consensus_rate: float | None
    mean_final_a_fraction: float | None
    mean_polarization_index: float | None
    mean_edge_disagreement_rate: float | None
    mean_time_to_consensus: float | None
    mean_opinion_variance: float | None
    mean_component_count: float | None
    mean_user_count: float | None
    mean_post_count: float | None
    mean_feed_impression_count: float | None
    mean_action_count: float | None
    mean_like_count: float | None
    mean_dm_count: float | None
    mean_follow_count: float | None
    mean_unfollow_count: float | None
    mean_initial_follow_edge_count: float | None
    mean_final_follow_edge_count: float | None
    mean_follow_edge_delta: float | None
    mean_new_follow_edge_count: float | None
    mean_removed_follow_edge_count: float | None
    mean_final_stance_mean: float | None
    mean_final_stance_variance: float | None
    mean_exposure_diversity: float | None
    mean_states_recorded: float | None
    mean_paid_impression_count: float | None
    mean_unique_paid_reach: float | None
    mean_organic_ad_impression_count: float | None
    mean_unique_organic_ad_reach: float | None
    mean_unique_total_ad_reach: float | None
    mean_relevant_paid_reach: float | None
    mean_relevant_total_reach: float | None
    mean_mean_ad_frequency: float | None
    mean_max_ad_frequency: float | None
    mean_frequency_cap_hit_count: float | None
    mean_ad_like_count: float | None
    mean_advertiser_follow_count: float | None
    mean_ad_dm_count: float | None
    mean_ad_generated_post_count: float | None
    mean_ad_negative_action_count: float | None
    mean_paid_to_organic_spillover_rate: float | None
    mean_ad_delivery_remaining_budget: float | None
    mean_burn_in_action_mean: float | None
    mean_burn_in_follow_churn: float | None
    mean_burn_in_exposure_diversity: float | None

    def metric(self, name: str) -> float | None:
        if name not in GROUP_SUMMARY_METRICS:
            raise ValueError(f"unknown group summary metric: {name}")
        return getattr(self, name)

    def to_dict(self) -> dict[str, object]:
        return {
            "factor_name": self.factor_name,
            "value": self.value,
            "runs": self.runs,
            "completed": self.completed,
            "failed": self.failed,
            "consensus_rate": self.consensus_rate,
            "mean_final_a_fraction": self.mean_final_a_fraction,
            "mean_polarization_index": self.mean_polarization_index,
            "mean_edge_disagreement_rate": self.mean_edge_disagreement_rate,
            "mean_time_to_consensus": self.mean_time_to_consensus,
            "mean_opinion_variance": self.mean_opinion_variance,
            "mean_component_count": self.mean_component_count,
            "mean_user_count": self.mean_user_count,
            "mean_post_count": self.mean_post_count,
            "mean_feed_impression_count": self.mean_feed_impression_count,
            "mean_action_count": self.mean_action_count,
            "mean_like_count": self.mean_like_count,
            "mean_dm_count": self.mean_dm_count,
            "mean_follow_count": self.mean_follow_count,
            "mean_unfollow_count": self.mean_unfollow_count,
            "mean_initial_follow_edge_count": self.mean_initial_follow_edge_count,
            "mean_final_follow_edge_count": self.mean_final_follow_edge_count,
            "mean_follow_edge_delta": self.mean_follow_edge_delta,
            "mean_new_follow_edge_count": self.mean_new_follow_edge_count,
            "mean_removed_follow_edge_count": self.mean_removed_follow_edge_count,
            "mean_final_stance_mean": self.mean_final_stance_mean,
            "mean_final_stance_variance": self.mean_final_stance_variance,
            "mean_exposure_diversity": self.mean_exposure_diversity,
            "mean_states_recorded": self.mean_states_recorded,
            "mean_paid_impression_count": self.mean_paid_impression_count,
            "mean_unique_paid_reach": self.mean_unique_paid_reach,
            "mean_organic_ad_impression_count": self.mean_organic_ad_impression_count,
            "mean_unique_organic_ad_reach": self.mean_unique_organic_ad_reach,
            "mean_unique_total_ad_reach": self.mean_unique_total_ad_reach,
            "mean_relevant_paid_reach": self.mean_relevant_paid_reach,
            "mean_relevant_total_reach": self.mean_relevant_total_reach,
            "mean_mean_ad_frequency": self.mean_mean_ad_frequency,
            "mean_max_ad_frequency": self.mean_max_ad_frequency,
            "mean_frequency_cap_hit_count": self.mean_frequency_cap_hit_count,
            "mean_ad_like_count": self.mean_ad_like_count,
            "mean_advertiser_follow_count": self.mean_advertiser_follow_count,
            "mean_ad_dm_count": self.mean_ad_dm_count,
            "mean_ad_generated_post_count": self.mean_ad_generated_post_count,
            "mean_ad_negative_action_count": self.mean_ad_negative_action_count,
            "mean_paid_to_organic_spillover_rate": (
                self.mean_paid_to_organic_spillover_rate
            ),
            "mean_ad_delivery_remaining_budget": self.mean_ad_delivery_remaining_budget,
            "mean_burn_in_action_mean": self.mean_burn_in_action_mean,
            "mean_burn_in_follow_churn": self.mean_burn_in_follow_churn,
            "mean_burn_in_exposure_diversity": self.mean_burn_in_exposure_diversity,
        }


@dataclass(frozen=True)
class IncompleteRun:
    run_id: str
    status: str
    error: str
    output_dir: str

    def to_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "error": self.error,
            "output_dir": self.output_dir,
        }


@dataclass(frozen=True)
class ToplineEntry:
    name: str
    factor_name: str
    value: str
    metric_value: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "factor_name": self.factor_name,
            "value": self.value,
            "metric_value": self.metric_value,
        }


@dataclass(frozen=True)
class AdIncrementalitySummary:
    condition: str
    comparable_blocks: int
    mean_total_reach_lift_vs_no_ad: float
    mean_engagement_lift_vs_no_ad: float

    def to_dict(self) -> dict[str, object]:
        return {
            "condition": self.condition,
            "comparable_blocks": self.comparable_blocks,
            "mean_total_reach_lift_vs_no_ad": self.mean_total_reach_lift_vs_no_ad,
            "mean_engagement_lift_vs_no_ad": self.mean_engagement_lift_vs_no_ad,
        }


@dataclass(frozen=True)
class SweepAnalysisResult:
    sweep_name: str
    source_dir: Path
    analysis_dir: Path
    runs: int
    completed: int
    failed: int
    factor_names: tuple[str, ...]
    group_summaries: tuple[GroupSummary, ...]
    toplines: dict[str, ToplineEntry]
    ad_incrementality: tuple[AdIncrementalitySummary, ...]
    incomplete_runs: tuple[IncompleteRun, ...]


def analyze_sweep(output_dir: str | Path) -> SweepAnalysisResult:
    sweep_dir = Path(output_dir)
    if not sweep_dir.exists():
        raise ValueError(f"sweep output directory does not exist: {sweep_dir}")
    _validate_required_artifacts(sweep_dir)

    sweep_config = _load_sweep_config(sweep_dir / "sweep_config.json")
    summary_json = _load_json(sweep_dir / "summary.json")
    manifest_entries = _load_manifest(sweep_dir / "manifest.jsonl")
    rows, fieldnames = _load_summary_csv(sweep_dir / "summary.csv")

    factor_names = tuple(factor.name for factor in sweep_config.factors)
    _validate_summary_columns(fieldnames, factor_names)
    _validate_summary_counts(rows, summary_json)
    _validate_manifest_entries(manifest_entries, rows)

    group_summaries = _group_summaries(factor_names, rows)
    return SweepAnalysisResult(
        sweep_name=sweep_config.sweep_name,
        source_dir=sweep_dir,
        analysis_dir=sweep_dir / "analysis",
        runs=len(rows),
        completed=_count_status(rows, "completed"),
        failed=_count_status(rows, "failed"),
        factor_names=factor_names,
        group_summaries=tuple(group_summaries),
        toplines=_toplines(group_summaries),
        ad_incrementality=_ad_incrementality_summaries(factor_names, rows),
        incomplete_runs=_incomplete_runs(rows),
    )


def _validate_required_artifacts(sweep_dir: Path) -> None:
    for filename in REQUIRED_ARTIFACTS:
        if not (sweep_dir / filename).exists():
            raise ValueError(f"missing required sweep artifact: {filename}")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON artifact {path.name}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"{path.name} root must be an object")
    return data


def _load_sweep_config(path: Path) -> SweepConfig:
    data = _load_json(path)
    try:
        return parse_sweep_config(data)
    except ValueError as exc:
        raise ValueError(f"invalid {path.name}: {exc}") from exc


def _load_manifest(path: Path) -> tuple[Mapping[str, Any], ...]:
    entries: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON artifact {path.name}: {exc}") from exc
        if not isinstance(entry, Mapping):
            raise ValueError("manifest.jsonl entries must be objects")
        entries.append(entry)
    return tuple(entries)


def _load_summary_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            fieldnames = list(reader.fieldnames or [])
            rows: list[dict[str, str]] = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(
                        f"malformed CSV artifact {path.name}: row {row_number} has extra cells"
                    )
                if any(value is None for value in row.values()):
                    raise ValueError(
                        f"malformed CSV artifact {path.name}: row {row_number} has missing cells"
                    )
                rows.append({key: value for key, value in row.items() if value is not None})
            return rows, fieldnames
    except csv.Error as exc:
        raise ValueError(f"malformed CSV artifact {path.name}: {exc}") from exc


def _validate_summary_columns(fieldnames: list[str], factor_names: tuple[str, ...]) -> None:
    required_columns = (
        "run_id",
        *factor_names,
        "experiment_name",
        "output_dir",
        "status",
        "error",
        *ANALYSIS_METRIC_COLUMNS,
    )
    for column in required_columns:
        if column not in fieldnames:
            raise ValueError(f"summary.csv is missing required column: {column}")


def _validate_summary_counts(rows: list[dict[str, str]], summary_json: Mapping[str, Any]) -> None:
    expected_runs = summary_json.get("runs")
    if isinstance(expected_runs, bool) or not isinstance(expected_runs, int):
        raise ValueError("summary.json runs must be an integer")
    if len(rows) != expected_runs:
        raise ValueError(
            f"summary.csv row count {len(rows)} does not match summary.json runs {expected_runs}"
        )

    for status in ("completed", "failed"):
        expected_status_count = summary_json.get(status)
        if isinstance(expected_status_count, bool) or not isinstance(expected_status_count, int):
            raise ValueError(f"summary.json {status} must be an integer")
        actual_status_count = _count_status(rows, status)
        if actual_status_count != expected_status_count:
            raise ValueError(
                f"summary.csv {status} count {actual_status_count} "
                f"does not match summary.json {status} {expected_status_count}"
            )


def _validate_manifest_entries(
    entries: tuple[Mapping[str, Any], ...],
    rows: list[dict[str, str]],
) -> None:
    if len(entries) != len(rows):
        raise ValueError(
            f"manifest.jsonl entry count {len(entries)} does not match summary.csv row count {len(rows)}"
        )

    for row_number, (entry, row) in enumerate(zip(entries, rows), start=1):
        summary_run_id = row["run_id"]
        for field in MANIFEST_COMPARE_FIELDS:
            manifest_value = _manifest_field_value(entry, field, row_number)
            summary_value = row[field] or ""
            if manifest_value != summary_value:
                raise ValueError(
                    f"manifest.jsonl row {row_number} field {field} for run_id {summary_run_id} "
                    f"does not match summary.csv: expected {summary_value!r}, got {manifest_value!r}"
                )


def _manifest_field_value(
    entry: Mapping[str, Any],
    field: str,
    row_number: int,
) -> str:
    if field not in entry:
        raise ValueError(f"manifest.jsonl row {row_number} is missing required field: {field}")
    value = entry[field]
    if field == "error" and value is None:
        return ""
    if not isinstance(value, str):
        expected_type = "string or null" if field == "error" else "string"
        raise ValueError(
            f"manifest.jsonl row {row_number} field {field} must be a {expected_type}"
        )
    return value


def _group_summaries(
    factor_names: tuple[str, ...],
    rows: list[dict[str, str]],
) -> list[GroupSummary]:
    summaries: list[GroupSummary] = []
    for factor_name in factor_names:
        rows_by_value: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            rows_by_value.setdefault(row[factor_name], []).append(row)
        for value, group_rows in rows_by_value.items():
            summaries.append(_group_summary(factor_name, value, group_rows))
    return summaries


def _group_summary(
    factor_name: str,
    value: str,
    rows: list[dict[str, str]],
) -> GroupSummary:
    metric_means = _metric_means(rows)
    return GroupSummary(
        factor_name=factor_name,
        value=value,
        runs=len(rows),
        completed=_count_status(rows, "completed"),
        failed=_count_status(rows, "failed"),
        consensus_rate=_consensus_rate(rows),
        mean_final_a_fraction=metric_means.get("final_a_fraction"),
        mean_polarization_index=metric_means.get("polarization_index"),
        mean_edge_disagreement_rate=metric_means.get("edge_disagreement_rate"),
        mean_time_to_consensus=metric_means.get("time_to_consensus"),
        mean_opinion_variance=metric_means.get("opinion_variance"),
        mean_component_count=metric_means.get("component_count"),
        mean_user_count=metric_means.get("user_count"),
        mean_post_count=metric_means.get("post_count"),
        mean_feed_impression_count=metric_means.get("feed_impression_count"),
        mean_action_count=metric_means.get("action_count"),
        mean_like_count=metric_means.get("like_count"),
        mean_dm_count=metric_means.get("dm_count"),
        mean_follow_count=metric_means.get("follow_count"),
        mean_unfollow_count=metric_means.get("unfollow_count"),
        mean_initial_follow_edge_count=metric_means.get("initial_follow_edge_count"),
        mean_final_follow_edge_count=metric_means.get("final_follow_edge_count"),
        mean_follow_edge_delta=metric_means.get("follow_edge_delta"),
        mean_new_follow_edge_count=metric_means.get("new_follow_edge_count"),
        mean_removed_follow_edge_count=metric_means.get("removed_follow_edge_count"),
        mean_final_stance_mean=metric_means.get("final_stance_mean"),
        mean_final_stance_variance=metric_means.get("final_stance_variance"),
        mean_exposure_diversity=metric_means.get("exposure_diversity"),
        mean_states_recorded=metric_means.get("states_recorded"),
        mean_paid_impression_count=metric_means.get("paid_impression_count"),
        mean_unique_paid_reach=metric_means.get("unique_paid_reach"),
        mean_organic_ad_impression_count=metric_means.get("organic_ad_impression_count"),
        mean_unique_organic_ad_reach=metric_means.get("unique_organic_ad_reach"),
        mean_unique_total_ad_reach=metric_means.get("unique_total_ad_reach"),
        mean_relevant_paid_reach=metric_means.get("relevant_paid_reach"),
        mean_relevant_total_reach=metric_means.get("relevant_total_reach"),
        mean_mean_ad_frequency=metric_means.get("mean_ad_frequency"),
        mean_max_ad_frequency=metric_means.get("max_ad_frequency"),
        mean_frequency_cap_hit_count=metric_means.get("frequency_cap_hit_count"),
        mean_ad_like_count=metric_means.get("ad_like_count"),
        mean_advertiser_follow_count=metric_means.get("advertiser_follow_count"),
        mean_ad_dm_count=metric_means.get("ad_dm_count"),
        mean_ad_generated_post_count=metric_means.get("ad_generated_post_count"),
        mean_ad_negative_action_count=metric_means.get("ad_negative_action_count"),
        mean_paid_to_organic_spillover_rate=metric_means.get(
            "paid_to_organic_spillover_rate"
        ),
        mean_ad_delivery_remaining_budget=metric_means.get(
            "ad_delivery_remaining_budget"
        ),
        mean_burn_in_action_mean=metric_means.get("burn_in_action_mean"),
        mean_burn_in_follow_churn=metric_means.get("burn_in_follow_churn"),
        mean_burn_in_exposure_diversity=metric_means.get("burn_in_exposure_diversity"),
    )


def _metric_means(rows: list[dict[str, str]]) -> dict[str, float]:
    means: dict[str, float] = {}
    completed_rows = [row for row in rows if row["status"] == "completed"]
    for field in NUMERIC_MEAN_FIELDS:
        values = [_parse_float(row.get(field)) for row in completed_rows]
        numeric_values = [value for value in values if value is not None]
        if numeric_values:
            means[field] = sum(numeric_values) / len(numeric_values)
    return means


def _consensus_rate(rows: list[dict[str, str]]) -> float | None:
    values = [
        _parse_bool(row["consensus_reached"])
        for row in rows
        if row["status"] == "completed"
    ]
    boolean_values = [value for value in values if value is not None]
    if not boolean_values:
        return None
    return sum(1 for value in boolean_values if value) / len(boolean_values)


def _toplines(group_summaries: list[GroupSummary]) -> dict[str, ToplineEntry]:
    toplines: dict[str, ToplineEntry] = {}
    consensus = _highest_topline(
        group_summaries,
        name="highest_consensus_rate",
        metric_value=lambda group: group.consensus_rate,
    )
    if consensus is not None:
        toplines["highest_consensus_rate"] = consensus

    polarization = _highest_topline(
        group_summaries,
        name="highest_polarization",
        metric_value=lambda group: group.mean_polarization_index,
    )
    if polarization is not None:
        toplines["highest_polarization"] = polarization

    edge_disagreement = _highest_topline(
        group_summaries,
        name="highest_edge_disagreement",
        metric_value=lambda group: group.mean_edge_disagreement_rate,
    )
    if edge_disagreement is not None:
        toplines["highest_edge_disagreement"] = edge_disagreement

    action_count = _highest_topline(
        group_summaries,
        name="highest_action_count",
        metric_value=lambda group: group.mean_action_count,
    )
    if action_count is not None:
        toplines["highest_action_count"] = action_count

    like_count = _highest_topline(
        group_summaries,
        name="highest_like_count",
        metric_value=lambda group: group.mean_like_count,
    )
    if like_count is not None:
        toplines["highest_like_count"] = like_count

    dm_count = _highest_topline(
        group_summaries,
        name="highest_dm_count",
        metric_value=lambda group: group.mean_dm_count,
    )
    if dm_count is not None:
        toplines["highest_dm_count"] = dm_count

    follow_delta = _highest_topline(
        group_summaries,
        name="highest_follow_edge_delta",
        metric_value=lambda group: group.mean_follow_edge_delta,
    )
    if follow_delta is not None:
        toplines["highest_follow_edge_delta"] = follow_delta

    exposure_diversity = _highest_topline(
        group_summaries,
        name="highest_exposure_diversity",
        metric_value=lambda group: group.mean_exposure_diversity,
    )
    if exposure_diversity is not None:
        toplines["highest_exposure_diversity"] = exposure_diversity

    stance_variance = _highest_topline(
        group_summaries,
        name="highest_final_stance_variance",
        metric_value=lambda group: group.mean_final_stance_variance,
    )
    if stance_variance is not None:
        toplines["highest_final_stance_variance"] = stance_variance

    total_ad_reach = _highest_topline(
        group_summaries,
        name="highest_total_ad_reach",
        metric_value=lambda group: group.mean_unique_total_ad_reach,
    )
    if total_ad_reach is not None:
        toplines["highest_total_ad_reach"] = total_ad_reach

    relevant_total_reach = _highest_topline(
        group_summaries,
        name="highest_relevant_total_reach",
        metric_value=lambda group: group.mean_relevant_total_reach,
    )
    if relevant_total_reach is not None:
        toplines["highest_relevant_total_reach"] = relevant_total_reach

    ad_like_count = _highest_topline(
        group_summaries,
        name="highest_ad_like_count",
        metric_value=lambda group: group.mean_ad_like_count,
    )
    if ad_like_count is not None:
        toplines["highest_ad_like_count"] = ad_like_count
    return toplines


def _ad_incrementality_summaries(
    factor_names: tuple[str, ...],
    rows: list[dict[str, str]],
) -> tuple[AdIncrementalitySummary, ...]:
    if "ad_condition" not in factor_names:
        return ()
    block_factor_names = tuple(name for name in factor_names if name != "ad_condition")
    rows_by_block: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        if row["status"] != "completed":
            continue
        rows_by_block.setdefault(
            tuple(row[name] for name in block_factor_names),
            [],
        ).append(row)

    lifts_by_condition: dict[str, list[tuple[float, float]]] = {}
    for block_rows in rows_by_block.values():
        baseline_rows = [row for row in block_rows if row.get("ad_condition") == "no_ad"]
        baseline_reach = _mean_row_metric(baseline_rows, "unique_total_ad_reach")
        if baseline_reach is None:
            continue
        baseline_engagement = _mean_row_engagement(baseline_rows)
        for condition in _condition_order(
            {row.get("ad_condition", "") for row in block_rows}
        ):
            if condition == "no_ad":
                continue
            condition_rows = [row for row in block_rows if row.get("ad_condition") == condition]
            reach = _mean_row_metric(condition_rows, "unique_total_ad_reach")
            if reach is None:
                continue
            engagement = _mean_row_engagement(condition_rows)
            lifts_by_condition.setdefault(condition, []).append(
                (reach - baseline_reach, engagement - baseline_engagement)
            )

    return tuple(
        AdIncrementalitySummary(
            condition=condition,
            comparable_blocks=len(lifts),
            mean_total_reach_lift_vs_no_ad=sum(reach for reach, _ in lifts) / len(lifts),
            mean_engagement_lift_vs_no_ad=sum(engagement for _, engagement in lifts)
            / len(lifts),
        )
        for condition, lifts in (
            (condition, lifts_by_condition[condition])
            for condition in _condition_order(set(lifts_by_condition))
        )
        if lifts
    )


def _condition_order(conditions: set[str]) -> tuple[str, ...]:
    preferred = ("organic_post", "sponsored_ad")
    preferred_present = tuple(condition for condition in preferred if condition in conditions)
    remaining = tuple(sorted(conditions - set(preferred) - {"", "no_ad"}))
    return (*preferred_present, *remaining)


def _mean_row_metric(rows: list[dict[str, str]], field: str) -> float | None:
    values = [_parse_float(row.get(field)) for row in rows]
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _mean_row_engagement(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    return sum(_row_engagement(row) for row in rows) / len(rows)


def _row_engagement(row: dict[str, str]) -> float:
    return sum(
        _parse_float(row.get(field)) or 0.0
        for field in (
            "ad_like_count",
            "advertiser_follow_count",
            "ad_dm_count",
            "ad_generated_post_count",
        )
    )


def _highest_topline(
    group_summaries: list[GroupSummary],
    name: str,
    metric_value: Callable[[GroupSummary], float | None],
) -> ToplineEntry | None:
    candidates = [
        (group, value)
        for group in group_summaries
        if (value := metric_value(group)) is not None
    ]
    non_seed_candidates = [
        (group, value) for group, value in candidates if group.factor_name != "seed"
    ]
    if non_seed_candidates:
        candidates = non_seed_candidates
    if not candidates:
        return None

    group, value = max(candidates, key=lambda item: item[1])
    return ToplineEntry(
        name=name,
        factor_name=group.factor_name,
        value=group.value,
        metric_value=value,
    )


def _incomplete_runs(rows: list[dict[str, str]]) -> tuple[IncompleteRun, ...]:
    incomplete: list[IncompleteRun] = []
    for row in rows:
        if row["status"] == "completed":
            continue
        incomplete.append(
            IncompleteRun(
                run_id=row["run_id"],
                status=row["status"],
                error=row["error"] or "",
                output_dir=row["output_dir"] or "",
            )
        )
    return tuple(incomplete)


def _count_status(rows: list[dict[str, str]], status: str) -> int:
    return sum(1 for row in rows if row["status"] == status)


def _parse_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None
