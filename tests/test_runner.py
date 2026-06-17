import json
from pathlib import Path

from society_simulation.config import ExperimentConfig
from society_simulation.models import Action
from society_simulation.runner import RunResult, run_experiment


def make_config(
    tmp_path: Path,
    seed: int = 123,
    true_state: Action | None = "A",
    update_policy: str = "bayesian_cascade",
    output_name: str | None = None,
) -> ExperimentConfig:
    run_name = output_name or f"run-{seed}-{true_state or 'sampled'}-{update_policy}"
    return ExperimentConfig(
        experiment_name="sequential_information_cascade",
        seed=seed,
        num_agents=6,
        true_state=true_state,
        signal_accuracy=0.7,
        prior_probability=0.5,
        scheduler="sequential",
        observation_policy="previous_actions",
        update_policy=update_policy,
        output_dir=str(tmp_path / run_name),
    )


def assert_replay_matches_result(config: ExperimentConfig, result: RunResult) -> None:
    steps_path = result.output_dir / "steps.jsonl"
    step_lines = steps_path.read_text(encoding="utf-8").splitlines()

    assert len(step_lines) == len(result.states)
    for step, state in zip((json.loads(line) for line in step_lines), result.states):
        assert step["agent_id"] == state.agent_id
        assert step["step_index"] == state.step_index
        assert step["private_signal"] == state.private_signal
        assert step["belief_probability"] == state.belief_probability
        assert step["confidence"] == state.confidence
        assert step["action"] == state.action
        assert step["observed_actions"] == list(state.observed_actions)
        assert step["true_state"] == result.true_state
        assert step["update_policy"] == config.update_policy
        assert step["random_seed"] == config.seed

    metrics = json.loads((result.output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == result.metrics


def test_run_experiment_writes_one_step_per_agent(tmp_path: Path) -> None:
    config = make_config(tmp_path, output_name="run-123")
    result = run_experiment(config)

    assert result.true_state == "A"
    assert result.output_dir == tmp_path / "run-123"
    assert isinstance(result.states, tuple)
    assert len(result.states) == 6
    assert (result.output_dir / "steps.jsonl").exists()
    assert_replay_matches_result(config, result)


def test_run_experiment_samples_true_state_when_config_is_none(tmp_path: Path) -> None:
    config = make_config(tmp_path, seed=101, true_state=None)

    result = run_experiment(config)

    assert result.true_state == "B"
    assert_replay_matches_result(config, result)


def test_run_experiment_supports_fixed_b_true_state(tmp_path: Path) -> None:
    config = make_config(tmp_path, seed=202, true_state="B")

    result = run_experiment(config)

    assert result.true_state == "B"
    assert_replay_matches_result(config, result)


def test_run_experiment_supports_simple_heuristic_policy(tmp_path: Path) -> None:
    config = make_config(tmp_path, seed=303, update_policy="simple_heuristic")

    result = run_experiment(config)

    assert result.true_state == "A"
    assert tuple(state.action for state in result.states) == ("A", "B", "A", "B", "B", "A")
    assert tuple(state.belief_probability for state in result.states) == (
        1.0,
        0.5,
        2 / 3,
        0.5,
        0.4,
        0.5,
    )
    assert_replay_matches_result(config, result)


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


def test_run_experiment_still_supports_sequential_config(tmp_path: Path) -> None:
    config = make_config(tmp_path, output_name="sequential-after-dispatch")

    result = run_experiment(config)

    assert result.true_state == "A"
    assert len(result.states) == 6
    assert (result.output_dir / "steps.jsonl").exists()
