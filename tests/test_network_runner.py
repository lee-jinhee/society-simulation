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
    assert not (result.output_dir / "llm_decisions.jsonl").exists()


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


def test_network_runner_records_mock_llm_usage_metrics(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        update_policy={
            "type": "mock_llm",
            "response_style": "current",
            "input_cost_per_1m_tokens": 1.0,
            "output_cost_per_1m_tokens": 2.0,
        },
        output_name="mock-llm",
    )

    result = run_experiment(config)

    usage = result.metrics["llm_usage"]
    assert usage["provider"] == "mock"
    assert usage["model"] == "mock-current"
    assert usage["calls"] == config.num_agents * config.scheduler.rounds
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_cost_usd"] > 0

    metrics_json = json.loads((result.output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_json["llm_usage"] == usage


def test_network_runner_writes_mock_llm_decision_audit_artifact(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        update_policy={
            "type": "mock_llm",
            "response_style": "current",
            "input_cost_per_1m_tokens": 1.0,
            "output_cost_per_1m_tokens": 2.0,
        },
        output_name="mock-llm-audit",
    )

    result = run_experiment(config)

    path = result.output_dir / "llm_decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == config.num_agents * config.scheduler.rounds
    first_row = rows[0]
    assert first_row["agent_id"] == 0
    assert first_row["round_index"] == 1
    assert first_row["provider"] == "mock"
    assert first_row["model"] == "mock-current"
    assert first_row["policy_type"] == "mock_llm"
    assert first_row["prompt"]
    assert first_row["raw_response"]
    assert first_row["parsed_action"] in ("A", "B")
    assert first_row["prompt_tokens"] > 0
    assert first_row["completion_tokens"] > 0
    assert first_row["total_cost_usd"] > 0
