from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACTS = (
    "sweep_config.json",
    "summary.csv",
    "summary.json",
    "manifest.jsonl",
)
METRIC_COLUMNS = (
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
)
NUMERIC_MEAN_FIELDS = (
    "final_a_fraction",
    "time_to_consensus",
    "polarization_index",
    "opinion_variance",
    "mean_belief",
    "edge_disagreement_rate",
    "component_count",
)


@dataclass(frozen=True)
class GroupSummary:
    factor_name: str
    value: str
    runs: int
    completed: int
    failed: int
    consensus_rate: float | None
    mean_final_a_fraction: float | None
    mean_time_to_consensus: float | None
    mean_polarization_index: float | None
    mean_opinion_variance: float | None
    mean_belief: float | None
    mean_edge_disagreement_rate: float | None
    mean_component_count: float | None
    metric_means: Mapping[str, float]


@dataclass(frozen=True)
class IncompleteRun:
    run_id: str
    status: str
    error: str | None
    factor_values: Mapping[str, str]
    output_dir: Path | None


@dataclass(frozen=True)
class ToplineEntry:
    metric_name: str
    factor_name: str
    value: str
    metric_value: float


@dataclass(frozen=True)
class SweepAnalysisResult:
    sweep_name: str
    runs: int
    completed: int
    failed: int
    output_dir: Path
    analysis_dir: Path
    group_summaries: tuple[GroupSummary, ...]
    toplines: dict[str, ToplineEntry]
    incomplete_runs: tuple[IncompleteRun, ...]


def analyze_sweep(output_dir: str | Path) -> SweepAnalysisResult:
    sweep_dir = Path(output_dir)
    _validate_required_artifacts(sweep_dir)

    sweep_config = _load_json(sweep_dir / "sweep_config.json")
    summary_json = _load_json(sweep_dir / "summary.json")
    _load_manifest(sweep_dir / "manifest.jsonl")
    rows, fieldnames = _load_summary_csv(sweep_dir / "summary.csv")

    factor_names = _factor_names(sweep_config)
    _validate_summary_columns(fieldnames, factor_names)
    _validate_summary_counts(rows, summary_json)

    group_summaries = _group_summaries(factor_names, rows)
    return SweepAnalysisResult(
        sweep_name=_sweep_name(sweep_config, summary_json, sweep_dir),
        runs=len(rows),
        completed=_count_status(rows, "completed"),
        failed=_count_status(rows, "failed"),
        output_dir=sweep_dir,
        analysis_dir=sweep_dir / "analysis",
        group_summaries=tuple(group_summaries),
        toplines=_toplines(group_summaries),
        incomplete_runs=_incomplete_runs(factor_names, rows),
    )


def _validate_required_artifacts(sweep_dir: Path) -> None:
    for filename in REQUIRED_ARTIFACTS:
        if not (sweep_dir / filename).exists():
            raise ValueError(f"missing required sweep artifact: {filename}")


def _load_json(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path.name} root must be an object")
    return data


def _load_manifest(path: Path) -> tuple[Mapping[str, Any], ...]:
    entries: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, Mapping):
            raise ValueError("manifest.jsonl entries must be objects")
        entries.append(entry)
    return tuple(entries)


def _load_summary_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _factor_names(sweep_config: Mapping[str, Any]) -> tuple[str, ...]:
    factors = sweep_config.get("factors")
    if not isinstance(factors, list):
        raise ValueError("sweep_config.json factors must be a list")

    names: list[str] = []
    for factor in factors:
        if not isinstance(factor, Mapping) or not isinstance(factor.get("name"), str):
            raise ValueError("sweep_config.json factors must include names")
        names.append(factor["name"])
    return tuple(names)


def _validate_summary_columns(fieldnames: list[str], factor_names: tuple[str, ...]) -> None:
    required_columns = (
        "run_id",
        *factor_names,
        "experiment_name",
        "output_dir",
        "status",
        "error",
        *METRIC_COLUMNS,
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
        mean_time_to_consensus=metric_means.get("time_to_consensus"),
        mean_polarization_index=metric_means.get("polarization_index"),
        mean_opinion_variance=metric_means.get("opinion_variance"),
        mean_belief=metric_means.get("mean_belief"),
        mean_edge_disagreement_rate=metric_means.get("edge_disagreement_rate"),
        mean_component_count=metric_means.get("component_count"),
        metric_means=metric_means,
    )


def _metric_means(rows: list[dict[str, str]]) -> dict[str, float]:
    means: dict[str, float] = {}
    completed_rows = [row for row in rows if row["status"] == "completed"]
    for field in NUMERIC_MEAN_FIELDS:
        values = [_parse_float(row[field]) for row in completed_rows]
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
        metric_name="consensus_rate",
        metric_value=lambda group: group.consensus_rate,
    )
    if consensus is not None:
        toplines["highest_consensus_rate"] = consensus

    polarization = _highest_topline(
        group_summaries,
        metric_name="polarization_index",
        metric_value=lambda group: group.mean_polarization_index,
    )
    if polarization is not None:
        toplines["highest_polarization"] = polarization
    return toplines


def _highest_topline(
    group_summaries: list[GroupSummary],
    metric_name: str,
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
        metric_name=metric_name,
        factor_name=group.factor_name,
        value=group.value,
        metric_value=value,
    )


def _incomplete_runs(
    factor_names: tuple[str, ...],
    rows: list[dict[str, str]],
) -> tuple[IncompleteRun, ...]:
    incomplete: list[IncompleteRun] = []
    for row in rows:
        if row["status"] == "completed":
            continue
        output_dir = Path(row["output_dir"]) if row["output_dir"] else None
        incomplete.append(
            IncompleteRun(
                run_id=row["run_id"],
                status=row["status"],
                error=row["error"] or None,
                factor_values={name: row[name] for name in factor_names},
                output_dir=output_dir,
            )
        )
    return tuple(incomplete)


def _count_status(rows: list[dict[str, str]], status: str) -> int:
    return sum(1 for row in rows if row["status"] == status)


def _parse_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
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


def _sweep_name(
    sweep_config: Mapping[str, Any],
    summary_json: Mapping[str, Any],
    sweep_dir: Path,
) -> str:
    for value in (summary_json.get("sweep_name"), sweep_config.get("sweep_name")):
        if isinstance(value, str) and value:
            return value
    return sweep_dir.name
