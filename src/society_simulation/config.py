from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Literal

from society_simulation.event_config import EventDrivenOpinionConfig

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


def _parse_optional_int(data: dict[str, object], field: str, path: str) -> int | None:
    if field not in data:
        return None
    return _require_int(data[field], path)


def _parse_optional_float(data: dict[str, object], field: str, path: str) -> float | None:
    if field not in data:
        return None
    return _require_float(data[field], path)


def _parse_optional_str(data: dict[str, object], field: str, path: str) -> str | None:
    if field not in data:
        return None
    return _require_non_empty_str(data[field], path)


def _validate_non_negative_finite_number(value: object, field: str) -> float:
    parsed = _require_float(value, field)
    if not isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be a non-negative finite number")
    return parsed


def _validate_positive_finite_number(value: object, field: str) -> float:
    parsed = _require_float(value, field)
    if not isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return parsed


def _validate_positive_int(value: object, field: str) -> int:
    parsed = _require_int(value, field)
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _reject_present_fields(
    data: dict[str, object],
    *,
    path_prefix: str,
    selected_type: str,
    kind: str,
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        if field in data:
            raise ValueError(f"{path_prefix}.{field} is not allowed for {selected_type} {kind}")


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
        topology_type = _require_field(data, "type", "topology.type")
        if not isinstance(topology_type, str):
            raise ValueError("topology.type must be a non-empty string")

        if topology_type == "complete":
            _reject_present_fields(
                data,
                path_prefix="topology",
                selected_type=topology_type,
                kind="topology",
                fields=("degree", "edge_probability", "rewiring_probability"),
            )
        elif topology_type == "cycle":
            _reject_present_fields(
                data,
                path_prefix="topology",
                selected_type=topology_type,
                kind="topology",
                fields=("degree", "edge_probability", "rewiring_probability"),
            )
        elif topology_type == "erdos_renyi":
            _reject_present_fields(
                data,
                path_prefix="topology",
                selected_type=topology_type,
                kind="topology",
                fields=("degree", "rewiring_probability"),
            )
        elif topology_type == "small_world":
            _reject_present_fields(
                data,
                path_prefix="topology",
                selected_type=topology_type,
                kind="topology",
                fields=("edge_probability",),
            )
        return cls(
            type=topology_type,
            degree=_parse_optional_int(data, "degree", "topology.degree"),
            edge_probability=_parse_optional_float(
                data, "edge_probability", "topology.edge_probability"
            ),
            rewiring_probability=_parse_optional_float(
                data, "rewiring_probability", "topology.rewiring_probability"
            ),
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
    provider: str | None = None
    model: str | None = None
    response_style: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float | None = None
    max_completion_tokens: int | None = None
    token_limit_parameter: str | None = None
    timeout_seconds: float | None = None
    input_cost_per_1m_tokens: float | None = None
    output_cost_per_1m_tokens: float | None = None
    max_estimated_cost_usd: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> NetworkUpdatePolicyConfig:
        data = _require_mapping(data, "update_policy")
        policy_type = _require_field(data, "type", "update_policy.type")
        if not isinstance(policy_type, str):
            raise ValueError("update_policy.type must be a non-empty string")

        llm_fields = (
            "provider",
            "model",
            "response_style",
            "base_url",
            "api_key_env",
            "temperature",
            "max_completion_tokens",
            "token_limit_parameter",
            "timeout_seconds",
            "input_cost_per_1m_tokens",
            "output_cost_per_1m_tokens",
            "max_estimated_cost_usd",
        )
        real_llm_fields = (
            "base_url",
            "api_key_env",
            "temperature",
            "max_completion_tokens",
            "token_limit_parameter",
            "timeout_seconds",
            "max_estimated_cost_usd",
        )
        if policy_type == "majority_rule":
            _reject_present_fields(
                data,
                path_prefix="update_policy",
                selected_type=policy_type,
                kind="update_policy",
                fields=("adoption_threshold", "self_weight", *llm_fields),
            )
        elif policy_type == "threshold":
            _reject_present_fields(
                data,
                path_prefix="update_policy",
                selected_type=policy_type,
                kind="update_policy",
                fields=("self_weight", *llm_fields),
            )
        elif policy_type == "degroot":
            _reject_present_fields(
                data,
                path_prefix="update_policy",
                selected_type=policy_type,
                kind="update_policy",
                fields=("adoption_threshold", *llm_fields),
            )
        elif policy_type == "mock_llm":
            _reject_present_fields(
                data,
                path_prefix="update_policy",
                selected_type=policy_type,
                kind="update_policy",
                fields=("adoption_threshold", "self_weight"),
            )
            _reject_present_fields(
                data,
                path_prefix="update_policy",
                selected_type=policy_type,
                kind="update_policy",
                fields=real_llm_fields,
            )
        elif policy_type == "llm":
            _reject_present_fields(
                data,
                path_prefix="update_policy",
                selected_type=policy_type,
                kind="update_policy",
                fields=("adoption_threshold", "self_weight", "response_style"),
            )
        return cls(
            type=policy_type,
            adoption_threshold=_parse_optional_float(
                data, "adoption_threshold", "update_policy.adoption_threshold"
            ),
            self_weight=_parse_optional_float(data, "self_weight", "update_policy.self_weight"),
            provider=_parse_optional_str(data, "provider", "update_policy.provider"),
            model=_parse_optional_str(data, "model", "update_policy.model"),
            response_style=_parse_optional_str(
                data,
                "response_style",
                "update_policy.response_style",
            ),
            base_url=_parse_optional_str(data, "base_url", "update_policy.base_url"),
            api_key_env=_parse_optional_str(
                data,
                "api_key_env",
                "update_policy.api_key_env",
            ),
            temperature=_parse_optional_float(
                data,
                "temperature",
                "update_policy.temperature",
            ),
            max_completion_tokens=_parse_optional_int(
                data,
                "max_completion_tokens",
                "update_policy.max_completion_tokens",
            ),
            token_limit_parameter=_parse_optional_str(
                data,
                "token_limit_parameter",
                "update_policy.token_limit_parameter",
            ),
            timeout_seconds=_parse_optional_float(
                data,
                "timeout_seconds",
                "update_policy.timeout_seconds",
            ),
            input_cost_per_1m_tokens=_parse_optional_float(
                data,
                "input_cost_per_1m_tokens",
                "update_policy.input_cost_per_1m_tokens",
            ),
            output_cost_per_1m_tokens=_parse_optional_float(
                data,
                "output_cost_per_1m_tokens",
                "update_policy.output_cost_per_1m_tokens",
            ),
            max_estimated_cost_usd=_parse_optional_float(
                data,
                "max_estimated_cost_usd",
                "update_policy.max_estimated_cost_usd",
            ),
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
        if self.type == "mock_llm":
            if self.provider is not None and self.provider != "mock":
                raise ValueError("unsupported llm provider")
            if self.response_style is not None and self.response_style not in (
                "neighbor_majority",
                "current",
                "contrarian",
            ):
                raise ValueError("unsupported mock llm response_style")
            if self.input_cost_per_1m_tokens is not None:
                _validate_non_negative_finite_number(
                    self.input_cost_per_1m_tokens,
                    "input_cost_per_1m_tokens",
                )
            if self.output_cost_per_1m_tokens is not None:
                _validate_non_negative_finite_number(
                    self.output_cost_per_1m_tokens,
                    "output_cost_per_1m_tokens",
                )
            return
        if self.type == "llm":
            if self.provider is not None and self.provider != "openai_compatible":
                raise ValueError("unsupported llm provider")
            if self.model is None:
                raise ValueError("model must be a non-empty string")
            if self.temperature is not None:
                _validate_probability(self.temperature, "temperature", 0.0, 2.0)
            if self.max_completion_tokens is not None:
                _validate_positive_int(
                    self.max_completion_tokens,
                    "max_completion_tokens",
                )
            if self.token_limit_parameter is not None and self.token_limit_parameter not in (
                "max_completion_tokens",
                "max_tokens",
            ):
                raise ValueError("token_limit_parameter must be max_completion_tokens or max_tokens")
            if self.timeout_seconds is not None:
                _validate_positive_finite_number(self.timeout_seconds, "timeout_seconds")
            if self.input_cost_per_1m_tokens is not None:
                _validate_non_negative_finite_number(
                    self.input_cost_per_1m_tokens,
                    "input_cost_per_1m_tokens",
                )
            if self.output_cost_per_1m_tokens is not None:
                _validate_non_negative_finite_number(
                    self.output_cost_per_1m_tokens,
                    "output_cost_per_1m_tokens",
                )
            if self.max_estimated_cost_usd is not None:
                _validate_non_negative_finite_number(
                    self.max_estimated_cost_usd,
                    "max_estimated_cost_usd",
                )
            return

        raise ValueError("unsupported network update_policy type")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"type": self.type}
        if self.adoption_threshold is not None:
            data["adoption_threshold"] = self.adoption_threshold
        if self.self_weight is not None:
            data["self_weight"] = self.self_weight
        if self.provider is not None:
            data["provider"] = self.provider
        if self.model is not None:
            data["model"] = self.model
        if self.response_style is not None:
            data["response_style"] = self.response_style
        if self.base_url is not None:
            data["base_url"] = self.base_url
        if self.api_key_env is not None:
            data["api_key_env"] = self.api_key_env
        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.max_completion_tokens is not None:
            data["max_completion_tokens"] = self.max_completion_tokens
        if self.token_limit_parameter is not None:
            data["token_limit_parameter"] = self.token_limit_parameter
        if self.timeout_seconds is not None:
            data["timeout_seconds"] = self.timeout_seconds
        if self.input_cost_per_1m_tokens is not None:
            data["input_cost_per_1m_tokens"] = self.input_cost_per_1m_tokens
        if self.output_cost_per_1m_tokens is not None:
            data["output_cost_per_1m_tokens"] = self.output_cost_per_1m_tokens
        if self.max_estimated_cost_usd is not None:
            data["max_estimated_cost_usd"] = self.max_estimated_cost_usd
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


Config = ExperimentConfig | NetworkHerdingConfig | EventDrivenOpinionConfig


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config root must be an object")
    if data.get("experiment_name") == "network_herding":
        config = NetworkHerdingConfig.from_dict(data)
    elif data.get("experiment_name") == "event_driven_opinion_dynamics":
        config = EventDrivenOpinionConfig.from_dict(data)
    else:
        config = ExperimentConfig(**data)
    config.validate()
    return config
