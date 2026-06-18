from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
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
NUMERIC_MEAN_FIELDS = (
    "final_a_fraction",
    "time_to_consensus",
    "polarization_index",
    "opinion_variance",
    "edge_disagreement_rate",
    "component_count",
)
GROUP_SUMMARY_METRICS = (
    "consensus_rate",
    "mean_final_a_fraction",
    "mean_polarization_index",
    "mean_edge_disagreement_rate",
    "mean_time_to_consensus",
    "mean_opinion_variance",
    "mean_component_count",
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
    mean_polarization_index: float | None
    mean_edge_disagreement_rate: float | None
    mean_time_to_consensus: float | None
    mean_opinion_variance: float | None
    mean_component_count: float | None

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
    incomplete_runs: tuple[IncompleteRun, ...]


def analyze_sweep(output_dir: str | Path) -> SweepAnalysisResult:
    sweep_dir = Path(output_dir)
    if not sweep_dir.exists():
        raise ValueError(f"sweep output directory does not exist: {sweep_dir}")
    _validate_required_artifacts(sweep_dir)

    sweep_config = _load_sweep_config(sweep_dir / "sweep_config.json")
    summary_json = _load_json(sweep_dir / "summary.json")
    _load_manifest(sweep_dir / "manifest.jsonl")
    rows, fieldnames = _load_summary_csv(sweep_dir / "summary.csv")

    factor_names = tuple(factor.name for factor in sweep_config.factors)
    _validate_summary_columns(fieldnames, factor_names)
    _validate_summary_counts(rows, summary_json)

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
    return toplines


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

