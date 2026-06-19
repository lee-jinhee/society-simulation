import json
from pathlib import Path

import pytest

from society_simulation.sweep_config import (
    SweepConfig,
    apply_path_override,
    build_experiment_config,
    expand_sweep,
    load_sweep_config,
    safe_label,
)
from tests.test_event_config import valid_event_config


def valid_base_config(tmp_path: Path) -> dict[str, object]:
    return {
        "experiment_name": "network_herding",
        "seed": 1,
        "num_agents": 6,
        "initial_opinion": {"type": "bernoulli", "probability_a": 0.5},
        "topology": {"type": "cycle"},
        "scheduler": {"type": "synchronous_rounds", "rounds": 2},
        "observation_policy": {"type": "neighbor_actions"},
        "update_policy": {"type": "threshold", "adoption_threshold": 0.6},
        "output_dir": str(tmp_path / "ignored"),
    }


def valid_sweep_dict(tmp_path: Path) -> dict[str, object]:
    return {
        "sweep_name": "network_topology_sweep",
        "base_config": valid_base_config(tmp_path),
        "factors": [
            {"name": "seed", "path": "seed", "values": [1, 2]},
            {
                "name": "initial_a",
                "path": "initial_opinion.probability_a",
                "values": [0.45, 0.55],
            },
            {
                "name": "topology",
                "values": [
                    {"label": "cycle", "overrides": {"topology": {"type": "cycle"}}},
                    {
                        "label": "complete",
                        "overrides": {"topology": {"type": "complete"}},
                    },
                ],
            },
        ],
        "output_dir": str(tmp_path / "sweep"),
    }


def test_load_sweep_config_parses_valid_config(tmp_path: Path) -> None:
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(valid_sweep_dict(tmp_path)), encoding="utf-8")

    sweep = load_sweep_config(path)

    assert isinstance(sweep, SweepConfig)
    assert sweep.sweep_name == "network_topology_sweep"
    assert sweep.output_dir == str(tmp_path / "sweep")
    assert [factor.name for factor in sweep.factors] == [
        "seed",
        "initial_a",
        "topology",
    ]


def test_expand_sweep_creates_deterministic_run_ids_and_output_dirs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(valid_sweep_dict(tmp_path)), encoding="utf-8")
    sweep = load_sweep_config(path)

    runs = expand_sweep(sweep)

    assert isinstance(runs, tuple)
    assert len(runs) == 8
    assert runs[0].run_id == "seed-1__initial_a-0_45__topology-cycle"
    assert runs[0].labels == {
        "seed": "1",
        "initial_a": "0_45",
        "topology": "cycle",
    }
    assert runs[0].config["seed"] == 1
    assert runs[0].config["initial_opinion"] == {
        "type": "bernoulli",
        "probability_a": 0.45,
    }
    assert runs[0].config["topology"] == {"type": "cycle"}
    assert runs[0].config["output_dir"] == str(
        tmp_path
        / "sweep"
        / "runs"
        / "seed-1__initial_a-0_45__topology-cycle"
    )
    assert runs[-1].run_id == "seed-2__initial_a-0_55__topology-complete"


def test_override_bundle_replaces_entire_subtree(tmp_path: Path) -> None:
    data = valid_sweep_dict(tmp_path)
    base_config = data["base_config"]
    assert isinstance(base_config, dict)
    base_config["topology"] = {
        "type": "small_world",
        "degree": 4,
        "rewiring_probability": 0.1,
    }
    data["factors"] = [
        {
            "name": "topology",
            "values": [
                {"label": "complete", "overrides": {"topology": {"type": "complete"}}}
            ],
        }
    ]
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    run = expand_sweep(load_sweep_config(path))[0]

    assert run.config["topology"] == {"type": "complete"}


def test_apply_path_override_requires_existing_object_path(tmp_path: Path) -> None:
    config = valid_base_config(tmp_path)

    apply_path_override(config, "initial_opinion.probability_a", 0.7)

    assert config["initial_opinion"] == {
        "type": "bernoulli",
        "probability_a": 0.7,
    }


def test_safe_label_converts_values_to_filesystem_safe_labels() -> None:
    assert safe_label(0.45) == "0_45"
    assert safe_label("small/world") == "small_world"
    assert safe_label(True) == "true"
    assert safe_label(None) == "null"


def test_sweep_config_to_dict_preserves_user_facing_schema(tmp_path: Path) -> None:
    data = valid_sweep_dict(tmp_path)
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    sweep = load_sweep_config(path)

    assert sweep.to_dict() == data


def test_load_sweep_config_rejects_missing_final_path_segment(tmp_path: Path) -> None:
    data = valid_sweep_dict(tmp_path)
    data["factors"] = [{"name": "bad", "path": "topology.typo", "values": ["complete"]}]
    path = tmp_path / "bad_path.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=r"factor path topology\.typo does not exist"):
        load_sweep_config(path)


def test_load_sweep_config_rejects_unknown_base_experiment_key(tmp_path: Path) -> None:
    data = valid_sweep_dict(tmp_path)
    base_config = data["base_config"]
    assert isinstance(base_config, dict)
    base_config["toplogy"] = {"type": "complete"}
    path = tmp_path / "unknown_base_key.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="experiment config contains unknown key toplogy"):
        load_sweep_config(path)


def test_load_sweep_config_rejects_factor_path_to_unknown_experiment_key(
    tmp_path: Path,
) -> None:
    data = valid_sweep_dict(tmp_path)
    base_config = data["base_config"]
    assert isinstance(base_config, dict)
    base_config["toplogy"] = {"type": "cycle"}
    data["factors"] = [{"name": "typo_topology", "path": "toplogy.type", "values": ["complete"]}]
    path = tmp_path / "unknown_path_key.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="experiment config contains unknown key toplogy"):
        load_sweep_config(path)


def test_load_sweep_config_rejects_override_bundle_unknown_experiment_key(
    tmp_path: Path,
) -> None:
    data = valid_sweep_dict(tmp_path)
    data["factors"] = [
        {
            "name": "typo_topology",
            "values": [
                {"label": "complete", "overrides": {"toplogy": {"type": "complete"}}}
            ],
        }
    ]
    path = tmp_path / "unknown_override_key.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="experiment config contains unknown key toplogy"):
        load_sweep_config(path)


@pytest.mark.parametrize("name", ["../escape", "bad/name", "!!!"])
def test_load_sweep_config_rejects_unsafe_factor_names(
    tmp_path: Path,
    name: str,
) -> None:
    data = valid_sweep_dict(tmp_path)
    data["factors"] = [{"name": name, "path": "seed", "values": [1]}]
    path = tmp_path / "bad_factor_name.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="factor name must match"):
        load_sweep_config(path)


def test_load_sweep_config_rejects_explicit_null_path_for_override_factor(
    tmp_path: Path,
) -> None:
    data = valid_sweep_dict(tmp_path)
    data["factors"] = [
        {
            "name": "topology",
            "path": None,
            "values": [{"label": "cycle", "overrides": {"topology": {"type": "cycle"}}}],
        }
    ]
    path = tmp_path / "null_path.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="factor topology path must be omitted"):
        load_sweep_config(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda data: data.update({"extra": True}),
            "sweep config contains unknown key extra",
        ),
        (
            lambda data: data.update(
                {"factors": [{"name": "seed", "path": "seed", "values": [1], "extra": True}]}
            ),
            "factor seed contains unknown key extra",
        ),
        (
            lambda data: data.update(
                {
                    "factors": [
                        {
                            "name": "topology",
                            "values": [
                                {
                                    "label": "cycle",
                                    "overrides": {"topology": {"type": "cycle"}},
                                    "extra": True,
                                }
                            ],
                        }
                    ]
                }
            ),
            "factor topology value contains unknown key extra",
        ),
    ],
)
def test_load_sweep_config_rejects_unknown_schema_keys(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    data = valid_sweep_dict(tmp_path)
    mutation(data)
    path = tmp_path / "unknown_key.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_sweep_config(path)


def test_safe_label_rejects_values_that_do_not_produce_labels() -> None:
    with pytest.raises(ValueError, match="label must contain"):
        safe_label("///")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update({"sweep_name": ""}), "sweep_name must be a non-empty string"),
        (lambda data: data.update({"base_config": []}), "base_config must be an object"),
        (lambda data: data.update({"factors": []}), "factors must be a non-empty list"),
        (
            lambda data: data.update(
                {
                    "factors": [
                        {"name": "seed", "path": "seed", "values": [1]},
                        {"name": "seed", "path": "seed", "values": [2]},
                    ]
                }
            ),
            "factor names must be unique",
        ),
        (
            lambda data: data.update(
                {"factors": [{"name": "seed", "path": "seed", "values": [1, 1]}]}
            ),
            "factor seed labels must be unique",
        ),
        (
            lambda data: data.update(
                {"factors": [{"name": "bad", "path": "seed.bad", "values": [1]}]}
            ),
            "traverses non-object field seed",
        ),
        (
            lambda data: data.update(
                {
                    "factors": [
                        {
                            "name": "bad_value",
                            "path": "seed",
                            "values": [{"label": "one", "overrides": {"seed": 1}}],
                        }
                    ]
                }
            ),
            "factor bad_value path values must be scalar",
        ),
        (
            lambda data: data.update(
                {"factors": [{"name": "topology", "values": [{"label": "cycle"}]}]}
            ),
            "factor topology overrides must be an object",
        ),
    ],
)
def test_load_sweep_config_rejects_invalid_shapes(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    data = valid_sweep_dict(tmp_path)
    mutation(data)
    path = tmp_path / "bad_sweep.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_sweep_config(path)


def test_load_sweep_config_rejects_generated_invalid_experiment_config(
    tmp_path: Path,
) -> None:
    data = valid_sweep_dict(tmp_path)
    data["factors"] = [
        {
            "name": "bad_threshold",
            "path": "update_policy.adoption_threshold",
            "values": [0.1],
        }
    ]
    path = tmp_path / "bad_generated.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="adoption_threshold must be between 0.5 and 1.0"):
        load_sweep_config(path)


def test_build_experiment_config_dispatches_event_driven_config(tmp_path: Path) -> None:
    config = build_experiment_config(valid_event_config(tmp_path))

    assert config.experiment_name == "event_driven_opinion_dynamics"
    assert config.scenario_name == "congestion_pricing"
