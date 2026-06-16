from __future__ import annotations

import argparse
from typing import Sequence

from society_simulation.config import load_config
from society_simulation.runner import run_experiment


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

    config = load_config(args.config)
    result = run_experiment(config)
    metrics = result.metrics

    print(f"experiment={config.experiment_name}")
    print(f"true_state={result.true_state}")
    print(f"action_counts={metrics['action_counts']}")
    print(f"correct_cascade={metrics['correct_cascade']}")
    print(f"wrong_cascade={metrics['wrong_cascade']}")
    print(f"output_dir={result.output_dir}")

    return 0
