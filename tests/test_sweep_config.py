import json
from pathlib import Path

import pytest

from society_simulation.sweep_config import (
    SweepConfig,
    apply_path_override,
    expand_sweep,
    load_sweep_config,
    safe_label,
)


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
