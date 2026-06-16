from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

Action = Literal["A", "B"]


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    seed: int
    num_agents: int
    true_state: Action | None
    signal_accuracy: float
    prior_probability: float
    scheduler: str
    observation_policy: str
    update_policy: str
    output_dir: str

    def validate(self) -> None:
        if self.experiment_name != "sequential_information_cascade":
            raise ValueError("unsupported experiment_name")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if self.true_state not in ("A", "B", None):
            raise ValueError("true_state must be A, B, or null")
        if not 0.5 <= self.signal_accuracy <= 1.0:
            raise ValueError("signal_accuracy must be between 0.5 and 1.0")
        if not 0.0 < self.prior_probability < 1.0:
            raise ValueError("prior_probability must be greater than 0 and less than 1")
        if self.scheduler != "sequential":
            raise ValueError("unsupported scheduler")
        if self.observation_policy != "previous_actions":
            raise ValueError("unsupported observation_policy")
        if self.update_policy not in ("bayesian_cascade", "simple_heuristic"):
            raise ValueError("unsupported update_policy")
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    config = ExperimentConfig(**data)
    config.validate()
    return config
