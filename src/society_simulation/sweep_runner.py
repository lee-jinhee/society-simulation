from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from society_simulation.sweep_artifacts import SweepRunRecord, write_sweep_artifacts
from society_simulation.sweep_config import (
    SweepConfig,
    build_experiment_config,
    expand_sweep,
)
from society_simulation.runner import run_experiment


@dataclass(frozen=True)
class SweepRunResult:
    sweep_name: str
    runs: int
    completed: int
    failed: int
    output_dir: Path
    manifest_path: Path
    summary_csv_path: Path
    summary_json_path: Path
    records: tuple[SweepRunRecord, ...]


def run_sweep(sweep: SweepConfig) -> SweepRunResult:
    planned_runs = expand_sweep(sweep)
    records: list[SweepRunRecord] = []

    for planned_run in planned_runs:
        config = build_experiment_config(planned_run.config)
        try:
            result = run_experiment(config)
        except Exception as exc:
            records.append(
                SweepRunRecord(
                    run_id=planned_run.run_id,
                    labels=planned_run.labels,
                    experiment_name=str(planned_run.config["experiment_name"]),
                    output_dir=planned_run.config["output_dir"],
                    status="failed",
                    error=str(exc),
                    metrics={},
                )
            )
            continue

        records.append(
            SweepRunRecord(
                run_id=planned_run.run_id,
                labels=planned_run.labels,
                experiment_name=config.experiment_name,
                output_dir=result.output_dir,
                status="completed",
                error=None,
                metrics=result.metrics,
            )
        )

    artifact_paths = write_sweep_artifacts(sweep, planned_runs, tuple(records))
    completed = sum(1 for record in records if record.status == "completed")
    failed = sum(1 for record in records if record.status == "failed")
    return SweepRunResult(
        sweep_name=sweep.sweep_name,
        runs=len(planned_runs),
        completed=completed,
        failed=failed,
        output_dir=artifact_paths.output_dir,
        manifest_path=artifact_paths.manifest_path,
        summary_csv_path=artifact_paths.summary_csv_path,
        summary_json_path=artifact_paths.summary_json_path,
        records=tuple(records),
    )
