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
    "mean_paid_impression_count",
    "mean_unique_paid_reach",
    "mean_organic_ad_impression_count",
    "mean_unique_organic_ad_reach",
    "mean_unique_total_ad_reach",
    "mean_relevant_paid_reach",
    "mean_relevant_total_reach",
    "mean_mean_ad_frequency",
    "mean_max_ad_frequency",
    "mean_frequency_cap_hit_count",
    "mean_ad_like_count",
    "mean_advertiser_follow_count",
    "mean_ad_dm_count",
    "mean_ad_generated_post_count",
    "mean_ad_negative_action_count",
    "mean_paid_to_organic_spillover_rate",
    "mean_ad_delivery_remaining_budget",
    "mean_burn_in_action_mean",
    "mean_burn_in_follow_churn",
    "mean_burn_in_exposure_diversity",
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
    "mean_paid_impression_count",
    "mean_unique_paid_reach",
    "mean_organic_ad_impression_count",
    "mean_unique_organic_ad_reach",
    "mean_unique_total_ad_reach",
    "mean_relevant_paid_reach",
    "mean_relevant_total_reach",
    "mean_mean_ad_frequency",
    "mean_max_ad_frequency",
    "mean_frequency_cap_hit_count",
    "mean_ad_like_count",
    "mean_advertiser_follow_count",
    "mean_ad_dm_count",
    "mean_ad_generated_post_count",
    "mean_ad_negative_action_count",
    "mean_paid_to_organic_spillover_rate",
    "mean_ad_delivery_remaining_budget",
    "mean_burn_in_action_mean",
    "mean_burn_in_follow_churn",
    "mean_burn_in_exposure_diversity",
)
_REPORT_GROUP_METRIC_FIELDS = (
    "consensus_rate",
    "mean_final_a_fraction",
    "mean_polarization_index",
    "mean_edge_disagreement_rate",
)
_REPORT_SOCIAL_METRIC_FIELDS = (
    "mean_feed_impression_count",
    "mean_action_count",
    "mean_like_count",
    "mean_dm_count",
    "mean_follow_edge_delta",
    "mean_final_stance_variance",
    "mean_exposure_diversity",
)
_REPORT_AD_METRIC_FIELDS = (
    "mean_paid_impression_count",
    "mean_unique_total_ad_reach",
    "mean_relevant_total_reach",
    "mean_ad_like_count",
    "mean_advertiser_follow_count",
    "mean_ad_dm_count",
    "mean_paid_to_organic_spillover_rate",
)
_TOPLINE_LABELS = {
    "highest_consensus_rate": "Highest consensus rate",
    "highest_polarization": "Highest polarization",
    "highest_edge_disagreement": "Highest edge disagreement",
    "highest_action_count": "Highest action count",
    "highest_like_count": "Highest like count",
    "highest_dm_count": "Highest DM count",
    "highest_follow_edge_delta": "Highest follow edge delta",
    "highest_exposure_diversity": "Highest exposure diversity",
    "highest_final_stance_variance": "Highest final stance variance",
    "highest_total_ad_reach": "Highest total ad reach",
    "highest_relevant_total_reach": "Highest relevant total ad reach",
    "highest_ad_like_count": "Highest ad like count",
}
_NETWORK_TOPLINE_ORDER = (
    "highest_consensus_rate",
    "highest_polarization",
    "highest_edge_disagreement",
)
_SOCIAL_TOPLINE_ORDER = (
    "highest_action_count",
    "highest_like_count",
    "highest_dm_count",
    "highest_follow_edge_delta",
    "highest_exposure_diversity",
    "highest_final_stance_variance",
)
_AD_TOPLINE_ORDER = (
    "highest_total_ad_reach",
    "highest_relevant_total_reach",
    "highest_ad_like_count",
)
_TOPLINE_ORDER = (*_NETWORK_TOPLINE_ORDER, *_SOCIAL_TOPLINE_ORDER, *_AD_TOPLINE_ORDER)


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
    has_social_metrics = _has_social_metrics(result)
    has_ad_metrics = _has_ad_metrics(result)
    for topline_name in _TOPLINE_ORDER:
        if topline_name not in result.toplines and (
            has_social_metrics
            or has_ad_metrics
            or topline_name in (*_SOCIAL_TOPLINE_ORDER, *_AD_TOPLINE_ORDER)
        ):
            continue
        lines.append(_topline_report_line(result, topline_name))

    if _has_network_report_metrics(result) or not has_social_metrics:
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
                            *(
                                _format_numeric(getattr(group, field))
                                for field in _REPORT_GROUP_METRIC_FIELDS
                            ),
                        ]
                    )
                    + " |"
                )

    if has_social_metrics:
        lines.extend(["", "## Social Media Metrics"])
        for factor_name in result.factor_names:
            lines.extend(
                [
                    "",
                    f"### {_markdown_cell(factor_name)}",
                    "",
                    (
                        "| value | runs | completed | failed | "
                        + " | ".join(_REPORT_SOCIAL_METRIC_FIELDS)
                        + " |"
                    ),
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
                            *(
                                _format_numeric(getattr(group, field))
                                for field in _REPORT_SOCIAL_METRIC_FIELDS
                            ),
                        ]
                    )
                    + " |"
                )

    if has_ad_metrics:
        lines.extend(
            [
                "",
                "## Ad Campaign Metrics",
                "Synthetic and uncalibrated: use these tables for relative mechanism screening, not real platform reach prediction.",
            ]
        )
        for factor_name in result.factor_names:
            lines.extend(
                [
                    "",
                    f"### {_markdown_cell(factor_name)}",
                    "",
                    (
                        "| value | runs | completed | failed | "
                        + " | ".join(_REPORT_AD_METRIC_FIELDS)
                        + " |"
                    ),
                    _markdown_separator(4 + len(_REPORT_AD_METRIC_FIELDS)),
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
                            *(
                                _format_numeric(getattr(group, field))
                                for field in _REPORT_AD_METRIC_FIELDS
                            ),
                        ]
                    )
                    + " |"
                )
        incrementality_lines = _ad_incrementality_lines(result)
        if incrementality_lines:
            lines.extend(["", "## Ad Incrementality", *incrementality_lines])

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
        "ad_incrementality": [
            summary.to_dict() for summary in result.ad_incrementality
        ],
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


def _has_social_metrics(result: SweepAnalysisResult) -> bool:
    return any(
        getattr(group, field) is not None
        for group in result.group_summaries
        for field in _REPORT_SOCIAL_METRIC_FIELDS
    )


def _has_ad_metrics(result: SweepAnalysisResult) -> bool:
    return any(
        getattr(group, field) is not None
        for group in result.group_summaries
        for field in _REPORT_AD_METRIC_FIELDS
    )


def _has_network_report_metrics(result: SweepAnalysisResult) -> bool:
    return any(
        getattr(group, field) is not None
        for group in result.group_summaries
        for field in _REPORT_GROUP_METRIC_FIELDS
    )


def _ad_incrementality_lines(result: SweepAnalysisResult) -> list[str]:
    if not result.ad_incrementality:
        return []
    lines = [
        (
            "| condition | comparable_blocks | mean_total_reach_lift_vs_no_ad | "
            "mean_engagement_lift_vs_no_ad |"
        ),
        "| --- | ---: | ---: | ---: |",
    ]
    for summary in result.ad_incrementality:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(summary.condition),
                    str(summary.comparable_blocks),
                    _format_numeric(summary.mean_total_reach_lift_vs_no_ad),
                    _format_numeric(summary.mean_engagement_lift_vs_no_ad),
                ]
            )
            + " |"
        )
    return lines


def _markdown_separator(columns: int) -> str:
    return "| " + " | ".join(["---", *("---:" for _ in range(columns - 1))]) + " |"


def _format_numeric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")
