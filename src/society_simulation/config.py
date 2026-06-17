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
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be between "
            f"{min_label or _format_bound(min_value)} and "
            f"{max_label or _format_bound(max_value)}"
        ) from exc

    if not min_value <= parsed <= max_value:
        raise ValueError(
            f"{field} must be between "
            f"{min_label or _format_bound(min_value)} and "
            f"{max_label or _format_bound(max_value)}"
        )
    return parsed


@dataclass(frozen=True)
class InitialOpinionConfig:
    type: str
    probability_a: float | None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> InitialOpinionConfig:
        return cls(type=data["type"], probability_a=data["probability_a"])

    def validate(self) -> None:
        if self.type != "bernoulli":
            raise ValueError("unsupported initial_opinion type")

        if self.probability_a is None:
            raise ValueError("probability_a must be between 0 and 1")
        _validate_probability(self.probability_a, "probability_a", 0.0, 1.0)


@dataclass(frozen=True)
class TopologyConfig:
    type: str
    degree: int | None = None
    edge_probability: float | None = None
    rewiring_probability: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TopologyConfig:
        return cls(
            type=data["type"],
            degree=data.get("degree"),
            edge_probability=data.get("edge_probability"),
            rewiring_probability=data.get("rewiring_probability"),
        )

    def validate(self, num_agents: int) -> None:
        if self.type == "complete":
            return
        if self.type == "cycle":
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
            if self.degree <= 0 or self.degree % 2 != 0:
                raise ValueError("small_world degree must be a positive even integer")
            if self.degree >= num_agents:
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


@dataclass(frozen=True)
class NetworkSchedulerConfig:
    type: str
    rounds: int

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> NetworkSchedulerConfig:
        return cls(type=data["type"], rounds=data["rounds"])

    def validate(self) -> None:
        if self.type != "synchronous_rounds":
            raise ValueError("unsupported scheduler type")
        if self.rounds <= 0:
            raise ValueError("rounds must be positive")


@dataclass(frozen=True)
class NetworkObservationPolicyConfig:
    type: str

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> NetworkObservationPolicyConfig:
        return cls(type=data["type"])

    def validate(self) -> None:
        if self.type != "neighbor_actions":
            raise ValueError("unsupported observation_policy type")


@dataclass(frozen=True)
class NetworkUpdatePolicyConfig:
    type: str
    adoption_threshold: float | None = None
    self_weight: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> NetworkUpdatePolicyConfig:
        return cls(
            type=data["type"],
            adoption_threshold=data.get("adoption_threshold"),
            self_weight=data.get("self_weight"),
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
        return cls(
            experiment_name=data["experiment_name"],
            seed=int(data["seed"]),
            num_agents=int(data["num_agents"]),
            initial_opinion=InitialOpinionConfig.from_dict(data["initial_opinion"]),
            topology=TopologyConfig.from_dict(data["topology"]),
            scheduler=NetworkSchedulerConfig.from_dict(data["scheduler"]),
            observation_policy=NetworkObservationPolicyConfig.from_dict(
                data["observation_policy"]
            ),
            update_policy=NetworkUpdatePolicyConfig.from_dict(data["update_policy"]),
            output_dir=data["output_dir"],
        )

    def validate(self) -> None:
        if self.experiment_name != "network_herding":
            raise ValueError("unsupported experiment_name")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")

        self.initial_opinion.validate()
        self.topology.validate(self.num_agents)
        self.scheduler.validate()
        self.observation_policy.validate()
        self.update_policy.validate()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


Config = ExperimentConfig | NetworkHerdingConfig


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if data.get("experiment_name") == "network_herding":
        config = NetworkHerdingConfig.from_dict(data)
    else:
        config = ExperimentConfig(**data)
    config.validate()
    return config
