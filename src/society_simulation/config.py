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


def _format_bound(value: float, *, force_decimal: bool = False) -> str:
    if value.is_integer():
        if force_decimal:
            return f"{value:.1f}"
        return str(int(value))
    return str(value)


def _validate_probability(
    value: object,
    field: str,
    min_value: float,
    max_value: float,
    *,
    min_label: str | None = None,
    max_label: str | None = None,
) -> float:
    parsed = _require_float(value, field)

    if not min_value <= parsed <= max_value:
        raise ValueError(
            f"{field} must be between "
            f"{min_label or _format_bound(min_value)} and "
            f"{max_label or _format_bound(max_value)}"
        )
    return parsed


def _require_mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _require_field(data: dict[str, object], field: str, path: str) -> object:
    if field not in data:
        raise ValueError(f"{path} is required")
    return data[field]


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _require_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _require_non_empty_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class InitialOpinionConfig:
    type: str
    probability_a: float | None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> InitialOpinionConfig:
        data = _require_mapping(data, "initial_opinion")
        return cls(
            type=_require_field(data, "type", "initial_opinion.type"),  # type: ignore[assignment]
            probability_a=_require_float(
                _require_field(
                    data,
                    "probability_a",
                    "initial_opinion.probability_a",
                ),
                "probability_a",
            ),
        )

    def validate(self) -> None:
        if self.type != "bernoulli":
            raise ValueError("unsupported initial_opinion type")

        if self.probability_a is None:
            raise ValueError("probability_a must be between 0 and 1")
        if not 0.0 <= self.probability_a <= 1.0:
            raise ValueError("probability_a must be between 0 and 1")

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, "probability_a": self.probability_a}


@dataclass(frozen=True)
class TopologyConfig:
    type: str
    degree: int | None = None
    edge_probability: float | None = None
    rewiring_probability: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TopologyConfig:
        data = _require_mapping(data, "topology")
        edge_probability: float | None
        if "edge_probability" in data:
            edge_probability = _require_float(data["edge_probability"], "topology.edge_probability")
        else:
            edge_probability = None
        rewiring_probability: float | None
        if "rewiring_probability" in data:
            rewiring_probability = _require_float(
                data["rewiring_probability"], "topology.rewiring_probability"
            )
        else:
            rewiring_probability = None
        return cls(
            type=_require_field(data, "type", "topology.type"),  # type: ignore[assignment]
            degree=data.get("degree"),
            edge_probability=edge_probability,
            rewiring_probability=rewiring_probability,
        )

    def validate(self, num_agents: int) -> None:
        if self.type == "complete":
            return
        if self.type == "cycle":
            if num_agents < 3:
                raise ValueError("cycle topology requires at least 3 agents")
            return
        if self.type == "erdos_renyi":
            if self.edge_probability is None:
                raise ValueError("erdos_renyi edge_probability must be between 0 and 1")
            _validate_probability(
                self.edge_probability,
                "erdos_renyi edge_probability",
                0.0,
                1.0,
            )
            return
        if self.type == "small_world":
            if self.degree is None:
                raise ValueError("small_world degree must be a positive even integer")
            if isinstance(self.degree, bool) or not isinstance(self.degree, int):
                raise ValueError("small_world degree must be a positive even integer")
            degree = self.degree
            if degree <= 0 or degree % 2 != 0:
                raise ValueError("small_world degree must be a positive even integer")
            if degree >= num_agents:
                raise ValueError("small_world degree must be less than num_agents")
            if self.rewiring_probability is None:
                raise ValueError("rewiring_probability must be between 0 and 1")
            _validate_probability(
                self.rewiring_probability,
                "rewiring_probability",
                0.0,
                1.0,
            )
            return

        raise ValueError("unsupported topology type")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"type": self.type}
        if self.degree is not None:
            data["degree"] = self.degree
        if self.edge_probability is not None:
            data["edge_probability"] = self.edge_probability
        if self.rewiring_probability is not None:
            data["rewiring_probability"] = self.rewiring_probability
        return data


@dataclass(frozen=True)
class NetworkSchedulerConfig:
    type: str
    rounds: int

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> NetworkSchedulerConfig:
        data = _require_mapping(data, "scheduler")
        return cls(
            type=_require_field(data, "type", "scheduler.type"),  # type: ignore[assignment]
            rounds=_require_int(
                _require_field(data, "rounds", "scheduler.rounds"),
                "scheduler.rounds",
            ),
        )

    def validate(self) -> None:
        if self.type != "synchronous_rounds":
            raise ValueError("unsupported scheduler type")
        if self.rounds <= 0:
            raise ValueError("rounds must be positive")

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, "rounds": self.rounds}


@dataclass(frozen=True)
class NetworkObservationPolicyConfig:
    type: str

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> NetworkObservationPolicyConfig:
        data = _require_mapping(data, "observation_policy")
        return cls(type=_require_field(data, "type", "observation_policy.type"))  # type: ignore[assignment]

    def validate(self) -> None:
        if self.type != "neighbor_actions":
            raise ValueError("unsupported observation_policy type")

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type}


@dataclass(frozen=True)
class NetworkUpdatePolicyConfig:
    type: str
    adoption_threshold: float | None = None
    self_weight: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> NetworkUpdatePolicyConfig:
        data = _require_mapping(data, "update_policy")
        if "adoption_threshold" in data:
            adoption_threshold: float | None = _require_float(
                data["adoption_threshold"], "adoption_threshold"
            )
        else:
            adoption_threshold = None
        if "self_weight" in data:
            self_weight: float | None = _require_float(data["self_weight"], "self_weight")
        else:
            self_weight = None
        return cls(
            type=_require_field(data, "type", "update_policy.type"),  # type: ignore[assignment]
            adoption_threshold=adoption_threshold,
            self_weight=self_weight,
        )

    def validate(self) -> None:
        if self.type == "majority_rule":
            return
        if self.type == "threshold":
            if self.adoption_threshold is None:
                raise ValueError("adoption_threshold must be between 0.5 and 1.0")
            _validate_probability(
                self.adoption_threshold,
                "adoption_threshold",
                0.5,
                1.0,
                min_label="0.5",
                max_label="1.0",
            )
            return
        if self.type == "degroot":
            if self.self_weight is None:
                raise ValueError("self_weight must be between 0 and 1")
            _validate_probability(self.self_weight, "self_weight", 0.0, 1.0)
            return

        raise ValueError("unsupported network update_policy type")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"type": self.type}
        if self.adoption_threshold is not None:
            data["adoption_threshold"] = self.adoption_threshold
        if self.self_weight is not None:
            data["self_weight"] = self.self_weight
        return data


@dataclass(frozen=True)
class NetworkHerdingConfig:
    experiment_name: str
    seed: int
    num_agents: int
    initial_opinion: InitialOpinionConfig
    topology: TopologyConfig
    scheduler: NetworkSchedulerConfig
    observation_policy: NetworkObservationPolicyConfig
    update_policy: NetworkUpdatePolicyConfig
    output_dir: str

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> NetworkHerdingConfig:
        data = _require_mapping(data, "network config")
        return cls(
            experiment_name=_require_field(data, "experiment_name", "experiment_name"),  # type: ignore[assignment]
            seed=_require_int(_require_field(data, "seed", "seed"), "seed"),
            num_agents=_require_int(_require_field(data, "num_agents", "num_agents"), "num_agents"),
            initial_opinion=InitialOpinionConfig.from_dict(
                _require_field(data, "initial_opinion", "initial_opinion")
            ),
            topology=TopologyConfig.from_dict(_require_field(data, "topology", "topology")),
            scheduler=NetworkSchedulerConfig.from_dict(_require_field(data, "scheduler", "scheduler")),
            observation_policy=NetworkObservationPolicyConfig.from_dict(
                _require_field(data, "observation_policy", "observation_policy")
            ),
            update_policy=NetworkUpdatePolicyConfig.from_dict(
                _require_field(data, "update_policy", "update_policy")
            ),
            output_dir=_require_non_empty_str(
                _require_field(data, "output_dir", "output_dir"),
                "output_dir",
            ),
        )

    def validate(self) -> None:
        if self.experiment_name != "network_herding":
            raise ValueError("unsupported experiment_name")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be a non-empty string")

        self.initial_opinion.validate()
        self.topology.validate(self.num_agents)
        self.scheduler.validate()
        self.observation_policy.validate()
        self.update_policy.validate()

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "num_agents": self.num_agents,
            "initial_opinion": self.initial_opinion.to_dict(),
            "topology": self.topology.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "observation_policy": self.observation_policy.to_dict(),
            "update_policy": self.update_policy.to_dict(),
            "output_dir": self.output_dir,
        }


Config = ExperimentConfig | NetworkHerdingConfig


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config root must be an object")
    if data.get("experiment_name") == "network_herding":
        config = NetworkHerdingConfig.from_dict(data)
    else:
        config = ExperimentConfig(**data)
    config.validate()
    return config
