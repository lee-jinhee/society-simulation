from pathlib import Path

import pytest

from society_simulation.config import ExperimentConfig, load_config


def test_config_accepts_valid_values(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_name="sequential_information_cascade",
        seed=42,
        num_agents=5,
        true_state="A",
        signal_accuracy=0.7,
        prior_probability=0.5,
        scheduler="sequential",
        observation_policy="previous_actions",
        update_policy="bayesian_cascade",
        output_dir=str(tmp_path / "run"),
    )

    config.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_agents", 0, "num_agents must be positive"),
        ("signal_accuracy", 0.49, "signal_accuracy must be between 0.5 and 1.0"),
        ("signal_accuracy", 1.01, "signal_accuracy must be between 0.5 and 1.0"),
        ("prior_probability", 0.0, "prior_probability must be greater than 0 and less than 1"),
        ("prior_probability", 1.0, "prior_probability must be greater than 0 and less than 1"),
        ("experiment_name", "unknown", "unsupported experiment_name"),
        ("scheduler", "random", "unsupported scheduler"),
        ("observation_policy", "global_truth", "unsupported observation_policy"),
        ("update_policy", "llm", "unsupported update_policy"),
        ("true_state", "C", "true_state must be A, B, or null"),
    ],
)
def test_config_rejects_invalid_values(tmp_path: Path, field: str, value: object, message: str) -> None:
    kwargs = {
        "experiment_name": "sequential_information_cascade",
        "seed": 42,
        "num_agents": 5,
        "true_state": "A",
        "signal_accuracy": 0.7,
        "prior_probability": 0.5,
        "scheduler": "sequential",
        "observation_policy": "previous_actions",
        "update_policy": "bayesian_cascade",
        "output_dir": str(tmp_path / "run"),
    }
    kwargs[field] = value
    config = ExperimentConfig(**kwargs)

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_load_config_from_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "experiment_name": "sequential_information_cascade",
          "seed": 7,
          "num_agents": 4,
          "true_state": null,
          "signal_accuracy": 0.65,
          "prior_probability": 0.5,
          "scheduler": "sequential",
          "observation_policy": "previous_actions",
          "update_policy": "simple_heuristic",
          "output_dir": "runs/example"
        }
        """,
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.seed == 7
    assert config.true_state is None
    assert config.update_policy == "simple_heuristic"


def test_load_config_keeps_sequential_cascade_type(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "experiment_name": "sequential_information_cascade",
          "seed": 7,
          "num_agents": 4,
          "true_state": null,
          "signal_accuracy": 0.65,
          "prior_probability": 0.5,
          "scheduler": "sequential",
          "observation_policy": "previous_actions",
          "update_policy": "simple_heuristic",
          "output_dir": "runs/example"
        }
        """,
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == "sequential_information_cascade"
