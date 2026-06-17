import json
from pathlib import Path

import pytest

from society_simulation.config import NetworkHerdingConfig, load_config


def valid_network_config(tmp_path: Path) -> dict[str, object]:
    return {
        "experiment_name": "network_herding",
        "seed": 42,
        "num_agents": 12,
        "initial_opinion": {"type": "bernoulli", "probability_a": 0.45},
        "topology": {
            "type": "small_world",
            "degree": 4,
            "rewiring_probability": 0.2,
        },
        "scheduler": {"type": "synchronous_rounds", "rounds": 8},
        "observation_policy": {"type": "neighbor_actions"},
        "update_policy": {"type": "threshold", "adoption_threshold": 0.6},
        "output_dir": str(tmp_path / "network-run"),
    }


def test_load_network_herding_config(tmp_path: Path) -> None:
    config_path = tmp_path / "network.json"
    config_path.write_text(json.dumps(valid_network_config(tmp_path)), encoding="utf-8")

    config = load_config(config_path)

    assert isinstance(config, NetworkHerdingConfig)
    assert config.experiment_name == "network_herding"
    assert config.topology.type == "small_world"
    assert config.scheduler.rounds == 8
    assert config.update_policy.type == "threshold"
    assert config.output_dir == str(tmp_path / "network-run")


def test_network_config_to_dict_round_trips_without_none_optionals(tmp_path: Path) -> None:
    config = NetworkHerdingConfig.from_dict(valid_network_config(tmp_path))
    payload = config.to_dict()

    assert "edge_probability" not in payload["topology"]
    assert "self_weight" not in payload["update_policy"]

    NetworkHerdingConfig.from_dict(payload).validate()


def test_network_config_to_dict_preserves_present_optional_fields(tmp_path: Path) -> None:
    data = valid_network_config(tmp_path)
    data["topology"] = {"type": "erdos_renyi", "edge_probability": 0.25}
    data["update_policy"] = {"type": "degroot", "self_weight": 0.3}

    config = NetworkHerdingConfig.from_dict(data)
    payload = config.to_dict()

    assert payload["topology"]["edge_probability"] == 0.25
    assert payload["update_policy"]["self_weight"] == 0.3

    NetworkHerdingConfig.from_dict(payload).validate()


def test_load_network_herding_config_rejects_root_non_object(tmp_path: Path) -> None:
    config_path = tmp_path / "network.json"
    config_path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="config root must be an object"):
        load_config(config_path)


def test_network_config_from_dict_rejects_missing_topology(tmp_path: Path) -> None:
    data = valid_network_config(tmp_path)
    del data["topology"]

    with pytest.raises(ValueError, match="topology is required"):
        NetworkHerdingConfig.from_dict(data)


def test_network_config_from_dict_rejects_missing_initial_opinion_probability_a(tmp_path: Path) -> None:
    data = valid_network_config(tmp_path)
    del data["initial_opinion"]["probability_a"]

    with pytest.raises(ValueError, match="initial_opinion.probability_a is required"):
        NetworkHerdingConfig.from_dict(data)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("num_agents",), 0, "num_agents must be positive"),
        (("initial_opinion", "probability_a"), -0.1, "probability_a must be between 0 and 1"),
        (("initial_opinion", "type"), "fixed", "unsupported initial_opinion type"),
        (("topology", "type"), "scale_free", "unsupported topology type"),
        (("topology", "degree"), 3, "small_world degree must be a positive even integer"),
        (("topology", "degree"), 2.0, "small_world degree must be a positive even integer"),
        (("topology", "degree"), 12, "small_world degree must be less than num_agents"),
        (("topology", "rewiring_probability"), 1.2, "rewiring_probability must be between 0 and 1"),
        (("scheduler", "rounds"), 0, "rounds must be positive"),
        (("scheduler", "type"), "asynchronous", "unsupported scheduler type"),
        (("observation_policy", "type"), "global_actions", "unsupported observation_policy type"),
        (("update_policy", "type"), "llm", "unsupported network update_policy type"),
        (("update_policy", "adoption_threshold"), 0.49, "adoption_threshold must be between 0.5 and 1.0"),
    ],
)
def test_network_config_rejects_invalid_values(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    data = valid_network_config(tmp_path)
    target = data
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]

    config = NetworkHerdingConfig.from_dict(data)

    with pytest.raises(ValueError, match=message):
        config.validate()


@pytest.mark.parametrize(
    ("path", "value", "message", "stage"),
    [
        (
            ("initial_opinion", "probability_a"),
            "0.6",
            "probability_a must be a number",
            "from_dict",
        ),
        (
            ("initial_opinion", "probability_a"),
            True,
            "probability_a must be a number",
            "from_dict",
        ),
        (("num_agents",), "12", "num_agents must be an integer", "from_dict"),
        (("scheduler", "rounds"), 2.5, "scheduler.rounds must be an integer", "from_dict"),
        (("scheduler", "rounds"), True, "scheduler.rounds must be an integer", "from_dict"),
        (("output_dir",), 123, "output_dir must be a non-empty string", "from_dict"),
        (("topology", "type"), "cycle", "cycle topology requires at least 3 agents", "validate"),
    ],
)
def test_network_config_rejects_invalid_types_and_shapes(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
    stage: str,
) -> None:
    data = valid_network_config(tmp_path)
    target = data
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]

    if stage == "validate":
        data["num_agents"] = 2
        config = NetworkHerdingConfig.from_dict(data)
        with pytest.raises(ValueError, match=message):
            config.validate()
    else:
        with pytest.raises(ValueError, match=message):
            NetworkHerdingConfig.from_dict(data)


def test_network_config_normalizes_numeric_types(tmp_path: Path) -> None:
    data = valid_network_config(tmp_path)
    data["seed"] = 42
    data["num_agents"] = 8
    data["scheduler"]["rounds"] = 3
    data["initial_opinion"]["probability_a"] = 1

    config = NetworkHerdingConfig.from_dict(data)

    assert isinstance(config.seed, int)
    assert isinstance(config.num_agents, int)
    assert isinstance(config.scheduler.rounds, int)
    assert isinstance(config.initial_opinion.probability_a, float)


def test_network_config_supports_complete_topology(tmp_path: Path) -> None:
    data = valid_network_config(tmp_path)
    data["topology"] = {"type": "complete"}

    config = NetworkHerdingConfig.from_dict(data)

    config.validate()
    assert config.topology.type == "complete"


def test_network_config_supports_erdos_renyi_topology(tmp_path: Path) -> None:
    data = valid_network_config(tmp_path)
    data["topology"] = {"type": "erdos_renyi", "edge_probability": 0.25}

    config = NetworkHerdingConfig.from_dict(data)

    config.validate()
    assert config.topology.edge_probability == 0.25


def test_network_config_supports_degroot_policy(tmp_path: Path) -> None:
    data = valid_network_config(tmp_path)
    data["update_policy"] = {"type": "degroot", "self_weight": 0.3}

    config = NetworkHerdingConfig.from_dict(data)

    config.validate()
    assert config.update_policy.self_weight == 0.3
