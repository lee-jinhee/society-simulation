import json
from pathlib import Path

from society_simulation.config import NetworkHerdingConfig
from society_simulation.runner import run_experiment


def make_config(
    tmp_path: Path,
    seed: int = 123,
    update_policy: dict[str, object] | None = None,
    output_name: str = "network-run",
) -> NetworkHerdingConfig:
    return NetworkHerdingConfig.from_dict(
        {
            "experiment_name": "network_herding",
            "seed": seed,
            "num_agents": 6,
            "initial_opinion": {"type": "bernoulli", "probability_a": 0.5},
            "topology": {"type": "cycle"},
            "scheduler": {"type": "synchronous_rounds", "rounds": 3},
            "observation_policy": {"type": "neighbor_actions"},
            "update_policy": update_policy or {"type": "majority_rule"},
            "output_dir": str(tmp_path / output_name),
        }
    )


def test_run_experiment_dispatches_network_herding(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    result = run_experiment(config)

    assert result.output_dir == tmp_path / "network-run"
    assert len(result.rounds) == 4
    assert len(result.rounds[0]) == 6
    assert result.metrics["final_action_counts"]["A"] + result.metrics["final_action_counts"]["B"] == 6
    assert (result.output_dir / "graph.json").exists()
    assert (result.output_dir / "timeseries.jsonl").exists()


def test_network_runner_writes_one_step_per_agent_per_update_round(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    result = run_experiment(config)

    steps = (result.output_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(steps) == config.num_agents * config.scheduler.rounds
    first_step = json.loads(steps[0])
    assert first_step["round_index"] == 1
    assert first_step["update_policy"] == "majority_rule"


def test_network_runner_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    first = run_experiment(make_config(tmp_path, seed=77, output_name="first"))
    second = run_experiment(make_config(tmp_path, seed=77, output_name="second"))

    assert first.metrics == second.metrics
    assert (first.output_dir / "steps.jsonl").read_text(encoding="utf-8") == (
        second.output_dir / "steps.jsonl"
    ).read_text(encoding="utf-8")
    assert (first.output_dir / "timeseries.jsonl").read_text(encoding="utf-8") == (
        second.output_dir / "timeseries.jsonl"
    ).read_text(encoding="utf-8")


def test_network_runner_supports_degroot_policy(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        update_policy={"type": "degroot", "self_weight": 0.4},
        output_name="degroot",
    )

    result = run_experiment(config)

    assert result.metrics["mean_belief"] >= 0.0
    assert result.metrics["mean_belief"] <= 1.0
