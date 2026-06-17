from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from society_simulation.config import NetworkHerdingConfig
from society_simulation.graph import Graph
from society_simulation.network_models import NetworkAgentState


class NetworkReplayWriter:
    def __init__(self, config: NetworkHerdingConfig) -> None:
        self.config = config

    def write(
        self,
        graph: Graph,
        rounds: tuple[tuple[NetworkAgentState, ...], ...],
        timeseries: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(output_dir / "config.json", self.config.to_dict())
        self._write_json(output_dir / "graph.json", graph.to_dict())
        self._write_steps(output_dir / "steps.jsonl", rounds)
        self._write_timeseries(output_dir / "timeseries.jsonl", timeseries)
        self._write_json(output_dir / "metrics.json", metrics)
        self._write_summary(output_dir / "summary.txt", metrics)
        return output_dir

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_steps(
        self,
        path: Path,
        rounds: tuple[tuple[NetworkAgentState, ...], ...],
    ) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for previous_round, current_round in zip(rounds, rounds[1:]):
                previous_by_id = {state.agent_id: state for state in previous_round}
                for state in current_round:
                    previous = previous_by_id[state.agent_id]
                    payload = state.to_dict()
                    payload["previous_belief_probability"] = previous.belief_probability
                    payload["previous_action"] = previous.action
                    payload["update_policy"] = self.config.update_policy.type
                    payload["random_seed"] = self.config.seed
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _write_timeseries(self, path: Path, timeseries: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in timeseries:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _write_summary(self, path: Path, metrics: dict[str, Any]) -> None:
        lines = [
            f"experiment_name={self.config.experiment_name}",
            f"update_policy={self.config.update_policy.type}",
            f"seed={self.config.seed}",
            f"final_action_counts={metrics['final_action_counts']}",
            f"consensus_reached={metrics['consensus_reached']}",
            f"consensus_action={metrics['consensus_action']}",
            f"edge_disagreement_rate={metrics['edge_disagreement_rate']}",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
