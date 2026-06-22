from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
import re
from typing import TypeAlias

from society_simulation.config import (
    Config,
    EventDrivenOpinionConfig,
    ExperimentConfig,
    NetworkHerdingConfig,
)
from society_simulation.social_media_config import InstagramSocialDynamicsConfig

Scalar: TypeAlias = str | int | float | bool | None
_FACTOR_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_NETWORK_HERDING_KEYS = {
    "experiment_name",
    "seed",
    "num_agents",
    "initial_opinion",
    "topology",
    "scheduler",
    "observation_policy",
    "update_policy",
    "output_dir",
}
_SEQUENTIAL_CASCADE_KEYS = {
    "experiment_name",
    "seed",
    "num_agents",
    "true_state",
    "signal_accuracy",
    "prior_probability",
    "scheduler",
    "observation_policy",
    "update_policy",
    "output_dir",
}
_EVENT_DRIVEN_OPINION_KEYS = {
    "experiment_name",
    "seed",
    "scenario_name",
    "days",
    "agents",
    "relationships",
    "events",
    "channels",
    "update_policy",
    "output_dir",
}
_INSTAGRAM_SOCIAL_DYNAMICS_KEYS = {
    "experiment_name",
    "seed",
    "scenario_name",
    "ticks",
    "num_users",
    "historical_posts_per_user",
    "feed_size",
    "activation_probability",
    "topics",
    "seed_generator",
    "feed_policy",
    "update_policy",
    "memory_retrieval",
    "seed_posts",
    "ad_campaigns",
    "output_dir",
}


@dataclass(frozen=True)
class SweepFactorValue:
    label: str
    value: Scalar | None = None
    overrides: dict[str, object] | None = None

    def to_override_dict(self) -> dict[str, object]:
        if self.overrides is None:
            raise ValueError("override value must include overrides")
        return {"label": self.label, "overrides": deepcopy(self.overrides)}


@dataclass(frozen=True)
class SweepFactor:
    name: str
    path: str | None
    values: tuple[SweepFactorValue, ...]

    def to_dict(self) -> dict[str, object]:
        if self.path is None:
            return {
                "name": self.name,
                "values": [value.to_override_dict() for value in self.values],
            }
        return {
            "name": self.name,
            "path": self.path,
            "values": [deepcopy(value.value) for value in self.values],
        }


@dataclass(frozen=True)
class SweepConfig:
    sweep_name: str
    base_config: dict[str, object]
    factors: tuple[SweepFactor, ...]
    output_dir: str

    def to_dict(self) -> dict[str, object]:
        return {
            "sweep_name": self.sweep_name,
            "base_config": deepcopy(self.base_config),
            "factors": [factor.to_dict() for factor in self.factors],
            "output_dir": self.output_dir,
        }


@dataclass(frozen=True)
class MaterializedRun:
    run_id: str
    labels: dict[str, str]
    config: dict[str, object]


def load_sweep_config(path: str | Path) -> SweepConfig:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return parse_sweep_config(data)


def parse_sweep_config(data: object) -> SweepConfig:
    if not isinstance(data, dict):
        raise ValueError("sweep config root must be an object")

    _reject_unknown_keys(
        data,
        allowed={"sweep_name", "base_config", "factors", "output_dir"},
        path="sweep config",
    )
    sweep_name = _require_non_empty_str(data.get("sweep_name"), "sweep_name")
    base_config = _require_mapping(data.get("base_config"), "base_config")
    factors_data = data.get("factors")
    if not isinstance(factors_data, list) or not factors_data:
        raise ValueError("factors must be a non-empty list")
    output_dir = _require_non_empty_str(data.get("output_dir"), "output_dir")

    factors = tuple(_parse_factor(factor) for factor in factors_data)
    names = [factor.name for factor in factors]
    if len(names) != len(set(names)):
        raise ValueError("factor names must be unique")

    sweep = SweepConfig(
        sweep_name=sweep_name,
        base_config=deepcopy(base_config),
        factors=factors,
        output_dir=output_dir,
    )
    expand_sweep(sweep)
    return sweep


def expand_sweep(sweep: SweepConfig) -> tuple[MaterializedRun, ...]:
    runs: list[MaterializedRun] = []
    for values in product(*(factor.values for factor in sweep.factors)):
        config = deepcopy(sweep.base_config)
        labels: dict[str, str] = {}
        id_parts: list[str] = []

        for factor, factor_value in zip(sweep.factors, values):
            labels[factor.name] = factor_value.label
            id_parts.append(f"{factor.name}-{factor_value.label}")
            if factor.path is None:
                assert factor_value.overrides is not None
                for key, value in factor_value.overrides.items():
                    config[key] = deepcopy(value)
            else:
                apply_path_override(config, factor.path, factor_value.value)

        run_id = "__".join(id_parts)
        config["output_dir"] = str(Path(sweep.output_dir) / "runs" / run_id)
        build_experiment_config(config)
        runs.append(MaterializedRun(run_id=run_id, labels=labels, config=config))
    return tuple(runs)


def apply_path_override(config: dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    if not parts or any(part == "" for part in parts):
        raise ValueError("path must be a non-empty dotted string")

    target: object = config
    traversed: list[str] = []
    for part in parts[:-1]:
        traversed.append(part)
        if isinstance(target, dict):
            next_value = target.get(part)
        elif isinstance(target, list):
            next_value = _list_item(target, part, ".".join(traversed))
        else:
            raise ValueError(f"traverses non-object field {'.'.join(traversed[:-1])}")
        if not isinstance(next_value, (dict, list)):
            if part != parts[-2]:
                raise ValueError(f"traverses non-object field {'.'.join(traversed)}")
        target = next_value
    final_part = parts[-1]
    if isinstance(target, dict):
        if final_part not in target:
            raise ValueError(f"factor path {path} does not exist")
        target[final_part] = value
        return
    if isinstance(target, list):
        index = _list_index(target, final_part, path)
        target[index] = value
        return
    traversed_path = ".".join(parts[:-1])
    raise ValueError(f"traverses non-object field {traversed_path}")


def _list_item(target: list[object], part: str, path: str) -> object:
    return target[_list_index(target, part, path)]


def _list_index(target: list[object], part: str, path: str) -> int:
    if not part.isdigit():
        raise ValueError(f"factor path {path} must use a non-negative list index")
    index = int(part)
    if index >= len(target):
        raise ValueError(f"factor path {path} list index is out of range")
    return index


def build_experiment_config(data: dict[str, object]) -> Config:
    if data.get("experiment_name") == "network_herding":
        _reject_unknown_keys(data, allowed=_NETWORK_HERDING_KEYS, path="experiment config")
        config = NetworkHerdingConfig.from_dict(data)
    elif data.get("experiment_name") == "event_driven_opinion_dynamics":
        _reject_unknown_keys(data, allowed=_EVENT_DRIVEN_OPINION_KEYS, path="experiment config")
        config = EventDrivenOpinionConfig.from_dict(data)
    elif data.get("experiment_name") == "instagram_social_dynamics":
        _reject_unknown_keys(
            data,
            allowed=_INSTAGRAM_SOCIAL_DYNAMICS_KEYS,
            path="experiment config",
        )
        config = InstagramSocialDynamicsConfig.from_dict(data)
    else:
        _reject_unknown_keys(data, allowed=_SEQUENTIAL_CASCADE_KEYS, path="experiment config")
        config = ExperimentConfig(**data)
    config.validate()
    return config


def safe_label(value: object) -> str:
    if value is True:
        text = "true"
    elif value is False:
        text = "false"
    elif value is None:
        text = "null"
    else:
        text = str(value)

    label = re.sub(r"[^A-Za-z0-9-]+", "_", text).strip("_")
    if not label:
        raise ValueError("label must contain at least one filesystem-safe character")
    return label


def _parse_factor(data: object) -> SweepFactor:
    if not isinstance(data, dict):
        raise ValueError("factor must be an object")

    name = _require_non_empty_str(data.get("name"), "factor name")
    if not _FACTOR_NAME_PATTERN.fullmatch(name):
        raise ValueError("factor name must match ^[A-Za-z0-9_-]+$")
    _reject_unknown_keys(data, allowed={"name", "path", "values"}, path=f"factor {name}")

    path: str | None
    if "path" not in data:
        path = None
    elif data["path"] is None:
        raise ValueError(f"factor {name} path must be omitted for override factors")
    elif not isinstance(data["path"], str):
        raise ValueError(f"factor {name} path must be a non-empty string")
    else:
        path = data["path"]

    if path == "":
        raise ValueError(f"factor {name} path must be a non-empty string")

    values_data = data.get("values")
    if not isinstance(values_data, list) or not values_data:
        raise ValueError(f"factor {name} values must be a non-empty list")

    if path is None:
        values = tuple(_parse_override_value(name, value) for value in values_data)
    else:
        values = tuple(_parse_path_value(name, value) for value in values_data)

    labels = [value.label for value in values]
    if len(labels) != len(set(labels)):
        raise ValueError(f"factor {name} labels must be unique")

    return SweepFactor(name=name, path=path, values=values)


def _parse_path_value(factor_name: str, value: object) -> SweepFactorValue:
    if not _is_scalar(value):
        raise ValueError(f"factor {factor_name} path values must be scalar")
    return SweepFactorValue(label=safe_label(value), value=value)


def _parse_override_value(factor_name: str, value: object) -> SweepFactorValue:
    if not isinstance(value, dict):
        raise ValueError(f"factor {factor_name} values must be override objects")
    _reject_unknown_keys(
        value,
        allowed={"label", "overrides"},
        path=f"factor {factor_name} value",
    )
    raw_label = value.get("label")
    if not isinstance(raw_label, str) or not raw_label:
        raise ValueError(f"factor {factor_name} label must be a non-empty string")
    overrides = value.get("overrides")
    if not isinstance(overrides, dict):
        raise ValueError(f"factor {factor_name} overrides must be an object")
    return SweepFactorValue(label=safe_label(raw_label), overrides=deepcopy(overrides))


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _require_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _require_non_empty_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _reject_unknown_keys(
    data: dict[str, object],
    *,
    allowed: set[str],
    path: str,
) -> None:
    for key in data:
        if key not in allowed:
            raise ValueError(f"{path} contains unknown key {key}")
