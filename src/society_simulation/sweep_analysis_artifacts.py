from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

from society_simulation.sweep_analysis import SweepAnalysisResult


GROUP_SUMMARY_FIELDS = (
    "factor",
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
FAILURE_FIELDS = ("run_id", "status", "error", "output_dir")

_GROUP_METRIC_FIELDS = (
    "consensus_rate",
    "mean_final_a_fraction",
    "mean_polarization_index",
    "mean_edge_disagreement_rate",
    "mean_time_to_consensus",
    "mean_opinion_variance",
    "mean_component_count",
)
_REPORT_GROUP_METRIC_FIELDS = (
    "consensus_rate",
    "mean_final_a_fraction",
    "mean_polarization_index",
    "mean_edge_disagreement_rate",
)
_TOPLINE_LABELS = {
    "highest_consensus_rate": "Highest consensus rate",
    "highest_polarization": "Highest polarization",
    "highest_edge_disagreement": "Highest edge disagreement",
}
_TOPLINE_ORDER = (
    "highest_consensus_rate",
    "highest_polarization",
    "highest_edge_disagreement",
)


@dataclass(frozen=True)
class SweepAnalysisArtifactPaths:
    output_dir: Path
    report_path: Path
    group_summary_csv_path: Path
    group_summary_json_path: Path
    failure_summary_csv_path: Path


def write_analysis_artifacts(result: SweepAnalysisResult) -> SweepAnalysisArtifactPaths:
    output_dir = result.analysis_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = SweepAnalysisArtifactPaths(
        output_dir=output_dir,
        report_path=output_dir / "report.md",
        group_summary_csv_path=output_dir / "group_summary.csv",
        group_summary_json_path=output_dir / "group_summary.json",
        failure_summary_csv_path=output_dir / "failure_summary.csv",
    )

    _write_report(paths.report_path, result)
    _write_group_summary_csv(paths.group_summary_csv_path, result)
    _write_group_summary_json(paths.group_summary_json_path, result)
    _write_failure_summary_csv(paths.failure_summary_csv_path, result)
    return paths


def _write_report(path: Path, result: SweepAnalysisResult) -> None:
    lines = [
        f"# Sweep Analysis: {result.sweep_name}",
        "",
        "## Overview",
        f"- Runs: {result.runs}",
        f"- Completed: {result.completed}",
        f"- Failed: {result.failed}",
        "",
        "## Topline",
    ]
    for topline_name in _TOPLINE_ORDER:
        lines.append(_topline_report_line(result, topline_name))

    lines.extend(["", "## Factor Summaries"])
    for factor_name in result.factor_names:
        lines.extend(
            [
                "",
                f"### {_markdown_cell(factor_name)}",
                "",
                (
                    "| value | runs | completed | failed | consensus_rate | "
                    "mean_final_a_fraction | mean_polarization_index | "
                    "mean_edge_disagreement_rate |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group in result.group_summaries:
            if group.factor_name != factor_name:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(group.value),
                        str(group.runs),
                        str(group.completed),
                        str(group.failed),
                        *(_format_numeric(getattr(group, field)) for field in _REPORT_GROUP_METRIC_FIELDS),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Failures"])
    if not result.incomplete_runs:
        lines.append("No failed or incomplete runs.")
    else:
        lines.extend(
            [
                "| run_id | status | error | output_dir |",
                "| --- | --- | --- | --- |",
            ]
        )
        for incomplete_run in result.incomplete_runs:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(incomplete_run.run_id),
                        _markdown_cell(incomplete_run.status),
                        _markdown_cell(incomplete_run.error),
                        _markdown_cell(incomplete_run.output_dir),
                    ]
                )
                + " |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_group_summary_csv(path: Path, result: SweepAnalysisResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUP_SUMMARY_FIELDS)
        writer.writeheader()
        for group in result.group_summaries:
            row = {
                "factor": group.factor_name,
                "value": group.value,
                "runs": group.runs,
                "completed": group.completed,
                "failed": group.failed,
            }
            row.update({field: _format_numeric(getattr(group, field)) for field in _GROUP_METRIC_FIELDS})
            writer.writerow(row)


def _write_group_summary_json(path: Path, result: SweepAnalysisResult) -> None:
    payload = {
        "sweep_name": result.sweep_name,
        "runs": result.runs,
        "completed": result.completed,
        "failed": result.failed,
        "groups": _json_groups(result),
        "toplines": {
            name: topline.to_dict()
            for name, topline in result.toplines.items()
        },
        "incomplete_runs": [run.to_dict() for run in result.incomplete_runs],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_failure_summary_csv(path: Path, result: SweepAnalysisResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_FIELDS)
        writer.writeheader()
        for incomplete_run in result.incomplete_runs:
            writer.writerow(incomplete_run.to_dict())


def _json_groups(result: SweepAnalysisResult) -> dict[str, dict[str, dict[str, object]]]:
    groups: dict[str, dict[str, dict[str, object]]] = {}
    for group in result.group_summaries:
        groups.setdefault(group.factor_name, {})[group.value] = group.to_dict()
    return groups


def _topline_report_line(result: SweepAnalysisResult, topline_name: str) -> str:
    label = _TOPLINE_LABELS[topline_name]
    topline = result.toplines.get(topline_name)
    if topline is None:
        return f"- {label}: no completed metric data"
    return (
        f"- {label}: {_markdown_cell(topline.factor_name)}={_markdown_cell(topline.value)} "
        f"({_format_numeric(topline.metric_value)})"
    )


def _format_numeric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")
