from __future__ import annotations

import argparse
import json
from typing import Sequence

from society_simulation.config import load_config
from society_simulation.runner import run_experiment


def _require_action_counts(metrics: dict[str, object]) -> object:
    if "action_counts" in metrics:
        return metrics["action_counts"]
    if "final_action_counts" in metrics:
        return metrics["final_action_counts"]
    raise ValueError("metrics must include action_counts or final_action_counts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="society-sim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run").add_argument("config")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command != "run":
        return 1

    try:
        config = load_config(args.config)
    except OSError as exc:
        parser.error(f"Unable to read config file '{args.config}': {exc}")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(f"Invalid config file '{args.config}': {exc}")

    try:
        result = run_experiment(config)
        metrics = result.metrics
        action_counts = _require_action_counts(metrics)
    except (OSError, ValueError) as exc:
        parser.error(f"Experiment run failed for '{args.config}': {exc}")

    print(f"experiment={config.experiment_name}")
    if hasattr(result, "true_state"):
        print(f"true_state={result.true_state}")
    print(f"action_counts={action_counts}")
    if "correct_cascade" in metrics:
        print(f"correct_cascade={metrics['correct_cascade']}")
    if "wrong_cascade" in metrics:
        print(f"wrong_cascade={metrics['wrong_cascade']}")
    if "consensus_reached" in metrics:
        print(f"consensus_reached={metrics['consensus_reached']}")
    if "edge_disagreement_rate" in metrics:
        print(f"edge_disagreement_rate={metrics['edge_disagreement_rate']}")
    print(f"output_dir={result.output_dir}")

    return 0
