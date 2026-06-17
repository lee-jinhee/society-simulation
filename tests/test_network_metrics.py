import pytest

from society_simulation.graph import Graph
from society_simulation.network_metrics import (
    compute_final_network_metrics,
    compute_round_metrics,
    edge_disagreement_rate,
)
from society_simulation.network_models import NetworkAgentState


def state(agent_id: int, action: str, belief: float, round_index: int) -> NetworkAgentState:
    return NetworkAgentState(
        agent_id=agent_id,
        belief_probability=belief,
        confidence=abs(belief - 0.5) * 2,
        action=action,  # type: ignore[arg-type]
        round_index=round_index,
        observed_neighbor_ids=(),
        observed_neighbor_actions=(),
    )


def test_round_metrics_measure_fraction_variance_and_edge_disagreement() -> None:
    graph = Graph({0: (1,), 1: (0, 2), 2: (1,)})
    previous = (
        state(0, "A", 0.8, 0),
        state(1, "B", 0.2, 0),
        state(2, "B", 0.3, 0),
    )
    current = (
        state(0, "A", 0.9, 1),
        state(1, "A", 0.6, 1),
        state(2, "B", 0.1, 1),
    )

    metrics = compute_round_metrics(graph, current, previous)

    assert metrics["round_index"] == 1
    assert metrics["a_fraction"] == pytest.approx(2 / 3)
    assert metrics["belief_mean"] == pytest.approx((0.9 + 0.6 + 0.1) / 3)
    assert metrics["edge_disagreement_rate"] == pytest.approx(0.5)
    assert metrics["action_changes"] == 1


def test_round_metrics_rejects_empty_states() -> None:
    graph = Graph({0: ()})

    with pytest.raises(ValueError, match="states must not be empty"):
        compute_round_metrics(graph, ())


def test_round_metrics_rejects_missing_graph_node_snapshot() -> None:
    graph = Graph({0: (1,), 1: (0,), 2: ()})
    states = (
        state(0, "A", 0.8, 1),
        state(1, "B", 0.2, 1),
    )

    with pytest.raises(ValueError, match="states must contain one state per graph node"):
        compute_round_metrics(graph, states)


def test_round_metrics_rejects_extra_graph_node_snapshot() -> None:
    graph = Graph({0: (1,), 1: (0,), 2: ()})
    states = (
        state(0, "A", 0.8, 1),
        state(1, "B", 0.2, 1),
        state(2, "A", 0.6, 1),
        state(3, "B", 0.4, 1),
    )

    with pytest.raises(ValueError, match="states must contain one state per graph node"):
        compute_round_metrics(graph, states)


def test_round_metrics_rejects_duplicate_state_ids() -> None:
    graph = Graph({0: (1,), 1: (0,), 2: ()})
    states = (
        state(0, "A", 0.8, 1),
        state(0, "B", 0.2, 1),
        state(2, "A", 0.6, 1),
    )

    with pytest.raises(ValueError, match="states must contain one state per graph node"):
        compute_round_metrics(graph, states)


def test_round_metrics_rejects_mixed_round_indices_in_snapshot() -> None:
    graph = Graph({0: (1,), 1: (0,), 2: ()})
    states = (
        state(0, "A", 0.8, 1),
        state(1, "B", 0.2, 2),
        state(2, "A", 0.6, 1),
    )

    with pytest.raises(ValueError, match="states must share the same round_index"):
        compute_round_metrics(graph, states)


def test_round_metrics_rejects_previous_state_id_mismatch() -> None:
    graph = Graph({0: (1,), 1: (0,), 2: ()})
    current = (
        state(0, "A", 0.8, 1),
        state(1, "B", 0.2, 1),
        state(2, "A", 0.6, 1),
    )
    previous = (
        state(0, "B", 0.7, 0),
        state(0, "A", 0.3, 0),
        state(2, "B", 0.2, 0),
    )

    with pytest.raises(ValueError, match="previous_states must contain one state per graph node"):
        compute_round_metrics(graph, current, previous)


def test_edge_disagreement_rate_rejects_malformed_snapshot_on_edgeless_graph() -> None:
    graph = Graph({0: (), 1: ()})
    states = (
        state(0, "A", 0.8, 0),
    )

    with pytest.raises(ValueError, match="states must contain one state per graph node"):
        edge_disagreement_rate(graph, states)


def test_final_metrics_detect_consensus_time() -> None:
    graph = Graph({0: (1,), 1: (0, 2), 2: (1,)})
    rounds = (
        (
            state(0, "A", 0.9, 0),
            state(1, "B", 0.2, 0),
            state(2, "B", 0.1, 0),
        ),
        (
            state(0, "A", 0.9, 1),
            state(1, "A", 0.7, 1),
            state(2, "B", 0.2, 1),
        ),
        (
            state(0, "A", 0.9, 2),
            state(1, "A", 0.8, 2),
            state(2, "A", 0.6, 2),
        ),
    )

    metrics = compute_final_network_metrics(graph, rounds)

    assert metrics["final_action_counts"] == {"A": 3, "B": 0}
    assert metrics["final_a_fraction"] == 1.0
    assert metrics["consensus_reached"] is True
    assert metrics["consensus_action"] == "A"
    assert metrics["time_to_consensus"] == 2
    assert metrics["edge_disagreement_rate"] == 0.0
    assert metrics["component_count"] == 1


def test_final_metrics_rejects_empty_rounds() -> None:
    graph = Graph({0: ()})

    with pytest.raises(ValueError, match="rounds must not be empty"):
        compute_final_network_metrics(graph, ())


def test_final_metrics_rejects_empty_round_in_history() -> None:
    graph = Graph({0: ()})

    with pytest.raises(ValueError, match="rounds must not contain empty rounds"):
        compute_final_network_metrics(graph, ((),))


def test_final_metrics_rejects_state_id_mismatch_in_any_round() -> None:
    graph = Graph({0: (1,), 1: (0,)})
    rounds = (
        (
            state(0, "A", 0.9, 0),
            state(1, "B", 0.1, 0),
        ),
        (
            state(0, "A", 0.9, 1),
        ),
    )

    with pytest.raises(ValueError, match="rounds must contain one state per graph node"):
        compute_final_network_metrics(graph, rounds)


def test_final_metrics_rejects_mixed_round_indices_within_a_round() -> None:
    graph = Graph({0: (1,), 1: (0,)})
    rounds = (
        (
            state(0, "A", 0.9, 0),
            state(1, "B", 0.1, 1),
        ),
    )

    with pytest.raises(
        ValueError,
        match="rounds must contain states with matching round_index",
    ):
        compute_final_network_metrics(graph, rounds)


def test_final_metrics_reports_zero_disagreement_for_graph_without_edges() -> None:
    graph = Graph({0: (), 1: ()})
    rounds = (
        (
            state(0, "A", 0.9, 0),
            state(1, "B", 0.1, 0),
        ),
    )

    metrics = compute_final_network_metrics(graph, rounds)

    assert metrics["edge_disagreement_rate"] == 0.0


def test_final_metrics_identify_no_consensus_and_components() -> None:
    graph = Graph({0: (1,), 1: (0,), 2: ()})
    rounds = (
        (
            state(0, "A", 0.9, 0),
            state(1, "B", 0.1, 0),
            state(2, "A", 0.8, 0),
        ),
    )

    metrics = compute_final_network_metrics(graph, rounds)

    assert metrics["consensus_reached"] is False
    assert metrics["consensus_action"] is None
    assert metrics["time_to_consensus"] is None
    assert metrics["component_count"] == 2
    assert metrics["polarization_index"] > 0.0


@pytest.mark.parametrize(
    "beliefs",
    [
        (0.9, 0.8, 0.7),
        (0.49, 0.51, 0.48, 0.52),
    ],
)
def test_final_metrics_polarization_zero_for_single_side_or_center_cluster(
    beliefs: tuple[float, ...],
) -> None:
    graph = Graph({index: () for index in range(len(beliefs))})
    rounds = (
        tuple(
            state(index, "A" if belief >= 0.5 else "B", belief, 0)
            for index, belief in enumerate(beliefs)
        ),
    )

    metrics = compute_final_network_metrics(graph, rounds)

    assert metrics["polarization_index"] == 0.0
