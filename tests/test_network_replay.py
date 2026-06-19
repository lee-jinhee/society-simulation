import json
from pathlib import Path

import pytest

from society_simulation.config import NetworkHerdingConfig
from society_simulation.graph import Graph
from society_simulation.network_models import NetworkAgentState
from society_simulation.network_replay import NetworkReplayWriter

REQUIRED_TIMESERIES_ROW = {
    "round_index": 0,
    "a_fraction": 0.5,
    "belief_mean": 0.5,
    "belief_variance": 0.25,
    "edge_disagreement_rate": 1.0,
    "action_changes": 0,
}

REQUIRED_METRICS = {
    "final_action_counts": {"A": 1, "B": 1},
    "final_a_fraction": 0.5,
    "consensus_reached": False,
    "consensus_action": None,
    "time_to_consensus": None,
    "polarization_index": 1.0,
    "opinion_variance": 0.25,
    "mean_belief": 0.5,
    "edge_disagreement_rate": 1.0,
    "component_count": 1,
}


def llm_decision_record() -> dict[str, object]:
    return {
        "agent_id": 0,
        "round_index": 1,
        "provider": "mock",
        "model": "mock-current",
        "policy_type": "mock_llm",
        "prompt": "agent_id=0",
        "raw_response": {"content": '{"action":"A","belief_probability":0.75}'},
        "parsed_action": "A",
        "parsed_belief_probability": 0.75,
        "confidence": 0.5,
        "prompt_tokens": 9,
        "completion_tokens": 7,
        "input_cost_usd": 0.000009,
        "output_cost_usd": 0.000014,
        "total_cost_usd": 0.000023,
        "latency_ms": 1.25,
    }


def make_config(tmp_path: Path) -> NetworkHerdingConfig:
    return NetworkHerdingConfig.from_dict(
        {
            "experiment_name": "network_herding",
            "seed": 42,
            "num_agents": 2,
            "initial_opinion": {"type": "bernoulli", "probability_a": 0.5},
            "topology": {"type": "complete"},
            "scheduler": {"type": "synchronous_rounds", "rounds": 1},
            "observation_policy": {"type": "neighbor_actions"},
            "update_policy": {"type": "majority_rule"},
            "output_dir": str(tmp_path / "network-run"),
        }
    )


def state(agent_id: int, action: str, belief: float, round_index: int) -> NetworkAgentState:
    return NetworkAgentState(
        agent_id=agent_id,
        belief_probability=belief,
        confidence=abs(belief - 0.5) * 2,
        action=action,  # type: ignore[arg-type]
        round_index=round_index,
        observed_neighbor_ids=(1 - agent_id,),
        observed_neighbor_actions=("B" if agent_id == 0 else "A",),  # type: ignore[arg-type]
    )


def test_network_replay_writer_writes_all_artifacts(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    graph = Graph({0: (1,), 1: (0,)}, topology={"type": "complete"})
    rounds = (
        (state(0, "A", 1.0, 0), state(1, "B", 0.0, 0)),
        (state(0, "B", 0.0, 1), state(1, "A", 1.0, 1)),
    )
    timeseries = [
        REQUIRED_TIMESERIES_ROW,
        {**REQUIRED_TIMESERIES_ROW, "round_index": 1, "action_changes": 2},
    ]
    metrics = REQUIRED_METRICS

    output_dir = NetworkReplayWriter(config).write(
        graph=graph,
        rounds=rounds,
        timeseries=timeseries,
        metrics=metrics,
    )

    assert output_dir == tmp_path / "network-run"
    assert (output_dir / "config.json").exists()
    assert (output_dir / "graph.json").exists()
    assert (output_dir / "steps.jsonl").exists()
    assert (output_dir / "timeseries.jsonl").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "summary.txt").exists()
    assert not (output_dir / "llm_decisions.jsonl").exists()

    graph_payload = json.loads((output_dir / "graph.json").read_text(encoding="utf-8"))
    assert graph_payload["adjacency"] == {"0": [1], "1": [0]}

    step_lines = (output_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(step_lines) == 2
    first_step = json.loads(step_lines[0])
    assert first_step["round_index"] == 1
    assert first_step["agent_id"] == 0
    assert first_step["previous_action"] == "A"
    assert first_step["action"] == "B"
    assert first_step["observed_neighbor_ids"] == [1]
    assert first_step["update_policy"] == "majority_rule"

    timeseries_lines = (output_dir / "timeseries.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(timeseries_lines) == 2


def test_network_replay_writer_writes_llm_decision_artifact(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    graph = Graph({0: (1,), 1: (0,)}, topology={"type": "complete"})
    rounds = (
        (state(0, "A", 1.0, 0), state(1, "B", 0.0, 0)),
        (state(0, "B", 0.0, 1), state(1, "A", 1.0, 1)),
    )
    timeseries = [
        REQUIRED_TIMESERIES_ROW,
        {**REQUIRED_TIMESERIES_ROW, "round_index": 1, "action_changes": 2},
    ]

    output_dir = NetworkReplayWriter(config).write(
        graph=graph,
        rounds=rounds,
        timeseries=timeseries,
        metrics=REQUIRED_METRICS,
        llm_decisions=(llm_decision_record(),),
    )

    lines = (output_dir / "llm_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["agent_id"] == 0
    assert row["provider"] == "mock"
    assert row["parsed_action"] == "A"
    assert row["raw_response"] == {"content": '{"action":"A","belief_probability":0.75}'}


def test_network_replay_writer_rejects_missing_llm_decision_key(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    graph = Graph({0: (1,), 1: (0,)}, topology={"type": "complete"})
    rounds = (
        (state(0, "A", 1.0, 0), state(1, "B", 0.0, 0)),
        (state(0, "B", 0.0, 1), state(1, "A", 1.0, 1)),
    )
    timeseries = [
        REQUIRED_TIMESERIES_ROW,
        {**REQUIRED_TIMESERIES_ROW, "round_index": 1, "action_changes": 2},
    ]
    incomplete_record = {
        key: value for key, value in llm_decision_record().items() if key != "prompt"
    }

    with pytest.raises(
        ValueError,
        match="llm_decisions row 0 is missing required key: prompt",
    ):
        NetworkReplayWriter(config).write(
            graph=graph,
            rounds=rounds,
            timeseries=timeseries,
            metrics=REQUIRED_METRICS,
            llm_decisions=(incomplete_record,),
        )

    assert not (tmp_path / "network-run").exists()


@pytest.mark.parametrize(
    ("rounds", "expected_message"),
    [
        ((), "rounds must not be empty"),
        (((),), "rounds must not contain empty rounds"),
        (
            (
                (state(0, "A", 1.0, 0),),
                (state(0, "B", 0.0, 1), state(0, "A", 1.0, 1)),
            ),
            "round 0 must contain one state per graph node",
        ),
        (
            (
                (state(0, "A", 1.0, 0), state(1, "B", 0.0, 0), state(2, "A", 1.0, 0)),
                (state(0, "B", 0.0, 1), state(1, "A", 1.0, 1)),
            ),
            "round 0 must contain one state per graph node",
        ),
        (
            (
                (
                    state(0, "A", 1.0, 0),
                    state(0, "B", 1.0, 0),
                ),
            ),
            "round 0 must contain one state per graph node",
        ),
    ],
)
def test_network_replay_writer_rejects_invalid_rounds(
    tmp_path: Path,
    rounds: tuple[tuple[NetworkAgentState, ...], ...],
    expected_message: str,
) -> None:
    config = make_config(tmp_path)
    graph = Graph({0: (1,), 1: (0,)}, topology={"type": "complete"})

    with pytest.raises(ValueError, match=expected_message):
        NetworkReplayWriter(config).write(
            graph=graph,
            rounds=rounds,
            timeseries=[],
            metrics=REQUIRED_METRICS,
        )

    assert not (tmp_path / "network-run").exists()


def test_network_replay_writer_rejects_mismatched_timeseries_length(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    graph = Graph({0: (1,), 1: (0,)}, topology={"type": "complete"})
    rounds = (
        (state(0, "A", 1.0, 0), state(1, "B", 0.0, 0)),
        (state(0, "B", 0.0, 1), state(1, "A", 1.0, 1)),
    )

    with pytest.raises(ValueError, match="timeseries must contain one row per round"):
        NetworkReplayWriter(config).write(
            graph=graph,
            rounds=rounds,
            timeseries=[{"round_index": 0}],
            metrics=REQUIRED_METRICS,
        )

    assert not (tmp_path / "network-run").exists()


def test_network_replay_writer_rejects_mismatched_timeseries_round_index(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    graph = Graph({0: (1,), 1: (0,)}, topology={"type": "complete"})
    rounds = (
        (state(0, "A", 1.0, 0), state(1, "B", 0.0, 0)),
        (state(0, "B", 0.0, 1), state(1, "A", 1.0, 1)),
    )

    with pytest.raises(ValueError, match="timeseries row 1 must have round_index 1"):
        NetworkReplayWriter(config).write(
            graph=graph,
            rounds=rounds,
            timeseries=[
                REQUIRED_TIMESERIES_ROW,
                {**REQUIRED_TIMESERIES_ROW, "round_index": 2},
            ],
            metrics=REQUIRED_METRICS,
        )

    assert not (tmp_path / "network-run").exists()


def test_network_replay_writer_rejects_mixed_round_indices(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    graph = Graph({0: (1,), 1: (0,)}, topology={"type": "complete"})
    rounds = (
        (
            state(0, "A", 1.0, 0),
            state(1, "B", 0.0, 1),
        ),
        (
            state(0, "B", 0.0, 1),
            state(1, "A", 1.0, 1),
        ),
    )

    with pytest.raises(ValueError, match="round 0 states must share the same round_index"):
        NetworkReplayWriter(config).write(
            graph=graph,
            rounds=rounds,
            timeseries=[
                REQUIRED_TIMESERIES_ROW,
                {**REQUIRED_TIMESERIES_ROW, "round_index": 1},
            ],
            metrics=REQUIRED_METRICS,
        )

    assert not (tmp_path / "network-run").exists()


def test_network_replay_writer_rejects_missing_metric_key(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    graph = Graph({0: (1,), 1: (0,)}, topology={"type": "complete"})
    rounds = (
        (state(0, "A", 1.0, 0), state(1, "B", 0.0, 0)),
        (state(0, "B", 0.0, 1), state(1, "A", 1.0, 1)),
    )

    with pytest.raises(ValueError, match="metrics is missing required key: component_count"):
        NetworkReplayWriter(config).write(
            graph=graph,
            rounds=rounds,
            timeseries=[
                REQUIRED_TIMESERIES_ROW,
                {**REQUIRED_TIMESERIES_ROW, "round_index": 1},
            ],
            metrics={key: value for key, value in REQUIRED_METRICS.items() if key != "component_count"},
        )

    assert not (tmp_path / "network-run").exists()


def test_network_replay_writer_rejects_missing_timeseries_row_key(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    graph = Graph({0: (1,), 1: (0,)}, topology={"type": "complete"})
    rounds = (
        (state(0, "A", 1.0, 0), state(1, "B", 0.0, 0)),
        (state(0, "B", 0.0, 1), state(1, "A", 1.0, 1)),
    )

    with pytest.raises(
        ValueError,
        match="timeseries row 1 is missing required key: action_changes",
    ):
        NetworkReplayWriter(config).write(
            graph=graph,
            rounds=rounds,
            timeseries=[
                REQUIRED_TIMESERIES_ROW,
                {
                    key: value
                    for key, value in {**REQUIRED_TIMESERIES_ROW, "round_index": 1}.items()
                    if key != "action_changes"
                },
            ],
            metrics=REQUIRED_METRICS,
        )

    assert not (tmp_path / "network-run").exists()
