from __future__ import annotations

import argparse
import json
from typing import Sequence

from society_simulation.config import load_config
from society_simulation.runner import run_experiment
from society_simulation.sweep_analysis import analyze_sweep
from society_simulation.sweep_analysis_artifacts import write_analysis_artifacts
from society_simulation.sweep_config import load_sweep_config
from society_simulation.sweep_runner import run_sweep

EVENT_SUMMARY_FIELDS = (
    "final_private_stance_mean",
    "final_public_stance_mean",
    "final_private_public_gap",
)
SOCIAL_MEDIA_SUMMARY_FIELDS = (
    "feed_impression_count",
    "action_count",
    "final_follow_edge_count",
    "final_stance_mean",
)


def _require_action_counts(metrics: dict[str, object]) -> object:
    if "action_counts" in metrics:
        return metrics["action_counts"]
    if "final_action_counts" in metrics:
        return metrics["final_action_counts"]
    raise ValueError("metrics must include action_counts or final_action_counts")


def _print_llm_usage(metrics: dict[str, object]) -> None:
    llm_usage = metrics.get("llm_usage")
    if not isinstance(llm_usage, dict):
        return

    print(f"llm_calls={llm_usage.get('calls', 0)}")
    print(f"llm_prompt_tokens={llm_usage.get('prompt_tokens', 0)}")
    print(f"llm_completion_tokens={llm_usage.get('completion_tokens', 0)}")
    print(f"llm_estimated_cost_usd={float(llm_usage.get('total_cost_usd', 0.0)):.8f}")


def _has_event_summary_metrics(metrics: dict[str, object]) -> bool:
    present = [field for field in EVENT_SUMMARY_FIELDS if field in metrics]
    if not present:
        return False
    if len(present) != len(EVENT_SUMMARY_FIELDS):
        raise ValueError(
            "event metrics must include final_private_stance_mean, "
            "final_public_stance_mean, and final_private_public_gap"
        )
    return True


def _has_social_media_summary_metrics(metrics: dict[str, object]) -> bool:
    present = [field for field in SOCIAL_MEDIA_SUMMARY_FIELDS if field in metrics]
    if not present:
        return False
    if len(present) != len(SOCIAL_MEDIA_SUMMARY_FIELDS):
        raise ValueError(
            "social media metrics must include feed_impression_count, action_count, "
            "final_follow_edge_count, and final_stance_mean"
        )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="society-sim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run").add_argument("config")
    subparsers.add_parser("sweep").add_argument("config")
    subparsers.add_parser("analyze").add_argument("sweep_output_dir")

    return parser


def _run_single_config(parser: argparse.ArgumentParser, config_path: str) -> int:
    try:
        config = load_config(config_path)
    except OSError as exc:
        parser.error(f"Unable to read config file '{config_path}': {exc}")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(f"Invalid config file '{config_path}': {exc}")

    try:
        result = run_experiment(config)
        metrics = result.metrics
        has_event_metrics = _has_event_summary_metrics(metrics)
        has_social_metrics = _has_social_media_summary_metrics(metrics)
        action_counts = (
            None if has_event_metrics or has_social_metrics else _require_action_counts(metrics)
        )
    except (OSError, ValueError) as exc:
        parser.error(f"Experiment run failed for '{config_path}': {exc}")

    print(f"experiment={config.experiment_name}")
    if hasattr(result, "true_state"):
        print(f"true_state={result.true_state}")
    if has_event_metrics:
        print(f"final_private_stance_mean={metrics['final_private_stance_mean']}")
        print(f"final_public_stance_mean={metrics['final_public_stance_mean']}")
        print(f"final_private_public_gap={metrics['final_private_public_gap']}")
        if "final_silent_agent_rate" in metrics:
            print(f"final_silent_agent_rate={metrics['final_silent_agent_rate']}")
        if "final_public_expression_bias" in metrics:
            print(f"final_public_expression_bias={metrics['final_public_expression_bias']}")
        if "final_perceived_majority_error" in metrics:
            print(f"final_perceived_majority_error={metrics['final_perceived_majority_error']}")
        if "final_speech_action_counts" in metrics:
            print(f"final_speech_action_counts={metrics['final_speech_action_counts']}")
        print(f"message_count={metrics.get('message_count')}")
    elif has_social_metrics:
        print(f"feed_impression_count={metrics['feed_impression_count']}")
        print(f"action_count={metrics['action_count']}")
        print(f"action_counts={metrics.get('action_counts')}")
        print(f"final_follow_edge_count={metrics['final_follow_edge_count']}")
        print(f"final_stance_mean={metrics['final_stance_mean']}")
    else:
        print(f"action_counts={action_counts}")
    if "correct_cascade" in metrics:
        print(f"correct_cascade={metrics['correct_cascade']}")
    if "wrong_cascade" in metrics:
        print(f"wrong_cascade={metrics['wrong_cascade']}")
    if "consensus_reached" in metrics:
        print(f"consensus_reached={metrics['consensus_reached']}")
    if "edge_disagreement_rate" in metrics:
        print(f"edge_disagreement_rate={metrics['edge_disagreement_rate']}")
    _print_llm_usage(metrics)
    print(f"output_dir={result.output_dir}")

    return 0


def _run_sweep_config(parser: argparse.ArgumentParser, config_path: str) -> int:
    try:
        sweep = load_sweep_config(config_path)
    except OSError as exc:
        parser.error(f"Unable to read sweep config file '{config_path}': {exc}")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(f"Invalid sweep config file '{config_path}': {exc}")

    try:
        result = run_sweep(sweep)
    except (OSError, ValueError) as exc:
        parser.error(f"Sweep run failed for '{config_path}': {exc}")

    print(f"sweep={result.sweep_name}")
    print(f"runs={result.runs}")
    print(f"completed={result.completed}")
    print(f"failed={result.failed}")
    print(f"output_dir={result.output_dir}")
    print(f"summary_csv={result.summary_csv_path}")

    if result.failed > 0:
        return 1
    return 0


def _run_analyze_config(parser: argparse.ArgumentParser, sweep_output_dir: str) -> int:
    try:
        result = analyze_sweep(sweep_output_dir)
        paths = write_analysis_artifacts(result)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"Analyze failed for '{sweep_output_dir}': {exc}")

    print(f"analysis={result.sweep_name}")
    print(f"runs={result.runs}")
    print(f"completed={result.completed}")
    print(f"failed={result.failed}")
    print(f"output_dir={paths.output_dir}")
    print(f"report={paths.report_path}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        return _run_single_config(parser, args.config)
    if args.command == "sweep":
        return _run_sweep_config(parser, args.config)
    if args.command == "analyze":
        return _run_analyze_config(parser, args.sweep_output_dir)
    return 1
