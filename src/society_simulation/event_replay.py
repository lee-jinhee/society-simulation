from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from society_simulation.event_config import EventDrivenOpinionConfig
from society_simulation.event_models import EventAgentState, EventExposure, EventMessage


class EventReplayWriter:
    def __init__(self, config: EventDrivenOpinionConfig) -> None:
        self.config = config

    def write(
        self,
        *,
        states_by_day: tuple[tuple[EventAgentState, ...], ...],
        exposures: tuple[EventExposure, ...],
        messages: tuple[EventMessage, ...],
        metrics: dict[str, Any],
        llm_decisions: tuple[dict[str, Any], ...],
        memories: tuple[object, ...] = (),
        retrievals: tuple[dict[str, Any], ...] = (),
    ) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(output_dir / "config.json", self.config.to_dict())
        self._write_json(
            output_dir / "agents.json",
            {"agents": [agent.to_dict() for agent in self.config.agents]},
        )
        self._write_json(
            output_dir / "relationships.json",
            {
                "relationships": [
                    relationship.to_dict() for relationship in self.config.relationships
                ]
            },
        )
        self._write_json(
            output_dir / "events.json",
            {"events": [event.to_dict() for event in self.config.events]},
        )
        self._write_jsonl(
            output_dir / "exposures.jsonl",
            tuple(exposure.to_dict() for exposure in exposures),
        )
        self._write_jsonl(
            output_dir / "messages.jsonl",
            tuple(message.to_dict() for message in messages),
        )
        self._write_jsonl(
            output_dir / "agent_states.jsonl",
            tuple(state.to_dict() for day_states in states_by_day for state in day_states),
        )
        self._write_json(output_dir / "metrics.json", metrics)
        self._write_jsonl(output_dir / "llm_decisions.jsonl", llm_decisions)
        self._write_jsonl(
            output_dir / "memories.jsonl",
            tuple(_json_ready_record(memory) for memory in memories),
        )
        self._write_jsonl(output_dir / "retrievals.jsonl", retrievals)
        self._write_summary(output_dir / "summary.md", metrics)
        return output_dir

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_jsonl(self, path: Path, rows: tuple[dict[str, Any], ...]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")

    def _write_summary(self, path: Path, metrics: dict[str, Any]) -> None:
        lines = [
            f"# {self.config.scenario_name}",
            "",
            f"- experiment_name: `{self.config.experiment_name}`",
            f"- agents: `{metrics.get('agent_count')}`",
            f"- days: `{metrics.get('day_count')}`",
            f"- final_private_stance_mean: `{metrics.get('final_private_stance_mean')}`",
            f"- final_public_stance_mean: `{metrics.get('final_public_stance_mean')}`",
            f"- final_private_public_gap: `{metrics.get('final_private_public_gap')}`",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_ready_record(value: object) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
    else:
        data = value
    if not isinstance(data, dict):
        raise ValueError("replay records must serialize to objects")
    return data
