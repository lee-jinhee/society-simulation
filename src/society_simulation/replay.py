from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from society_simulation.config import ExperimentConfig
from society_simulation.models import Action, AgentState


class ReplayWriter:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def write(
        self,
        true_state: Action,
        states: list[AgentState],
        metrics: dict[str, Any],
    ) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._write_steps(output_dir / "steps.jsonl", true_state, states)
        self._write_summary(output_dir / "summary.txt", true_state, metrics)
        return output_dir

    def _write_steps(self, path: Path, true_state: Action, states: list[AgentState]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for state in states:
                payload = state.to_dict()
                payload["true_state"] = true_state
                payload["update_policy"] = self.config.update_policy
                payload["random_seed"] = self.config.seed
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _write_summary(self, path: Path, true_state: Action, metrics: dict[str, Any]) -> None:
        lines = [
            f"experiment_name={self.config.experiment_name}",
            f"update_policy={self.config.update_policy}",
            f"seed={self.config.seed}",
            f"true_state={true_state}",
            f"action_counts={metrics['action_counts']}",
            f"correct_cascade={metrics['correct_cascade']}",
            f"wrong_cascade={metrics['wrong_cascade']}",
            f"cascade_start_step={metrics['cascade_start_step']}",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
