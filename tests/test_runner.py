import json
from pathlib import Path

from society_simulation.config import ExperimentConfig
from society_simulation.runner import run_experiment


def make_config(tmp_path: Path, seed: int = 123) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="sequential_information_cascade",
        seed=seed,
        num_agents=6,
        true_state="A",
        signal_accuracy=0.7,
        prior_probability=0.5,
        scheduler="sequential",
        observation_policy="previous_actions",
        update_policy="bayesian_cascade",
        output_dir=str(tmp_path / f"run-{seed}"),
    )


def test_run_experiment_writes_one_step_per_agent(tmp_path: Path) -> None:
    result = run_experiment(make_config(tmp_path))

    assert result.true_state == "A"
    assert result.output_dir == tmp_path / "run-123"
    assert len(result.states) == 6
    assert (result.output_dir / "steps.jsonl").exists()
    assert len((result.output_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()) == 6


def test_run_experiment_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    first_config = make_config(tmp_path, seed=77)
    second_config = ExperimentConfig(
        **{
            **first_config.to_dict(),
            "output_dir": str(tmp_path / "run-77-copy"),
        }
    )

    first = run_experiment(first_config)
    second = run_experiment(second_config)

    first_steps = (first.output_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    second_steps = (second.output_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    normalized_first = [
        {key: value for key, value in json.loads(line).items()}
        for line in first_steps
    ]
    normalized_second = [
        {key: value for key, value in json.loads(line).items()}
        for line in second_steps
    ]
    assert normalized_first == normalized_second
    assert first.metrics == second.metrics
