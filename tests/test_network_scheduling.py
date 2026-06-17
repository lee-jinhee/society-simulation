import random

import pytest

from society_simulation.graph import Graph
from society_simulation.models import Action
from society_simulation.network_models import NetworkAgentState
from society_simulation.network_scheduling import (
    build_neighbor_observations,
    initialize_network_states,
)


def make_state(agent_id: int, action: Action, belief: float) -> NetworkAgentState:
    return NetworkAgentState(
        agent_id=agent_id,
        belief_probability=belief,
        confidence=abs(belief - 0.5) * 2,
        action=action,
        round_index=0,
        observed_neighbor_ids=(),
        observed_neighbor_actions=(),
    )


def test_initialize_network_states_is_deterministic() -> None:
    first = initialize_network_states(5, probability_a=0.4, rng=random.Random(11))
    second = initialize_network_states(5, probability_a=0.4, rng=random.Random(11))

    assert first == second
    assert all(state.round_index == 0 for state in first)


def test_initialize_network_states_rejects_invalid_num_agents() -> None:
    with pytest.raises(ValueError, match="num_agents must be positive"):
        initialize_network_states(0, probability_a=0.4, rng=random.Random(11))


@pytest.mark.parametrize("probability_a", [-0.1, 1.1])
def test_initialize_network_states_rejects_invalid_probability_a(
    probability_a: float,
) -> None:
    with pytest.raises(
        ValueError, match="probability_a must be between 0 and 1"
    ):
        initialize_network_states(5, probability_a=probability_a, rng=random.Random(11))


def test_initialize_network_states_respects_probability_extremes() -> None:
    all_b = initialize_network_states(3, probability_a=0.0, rng=random.Random(11))
    assert all(
        state.action == "B"
        and state.belief_probability == 0.0
        and state.confidence == 1.0
        for state in all_b
    )

    all_a = initialize_network_states(4, probability_a=1.0, rng=random.Random(11))
    assert all(
        state.action == "A"
        and state.belief_probability == 1.0
        and state.confidence == 1.0
        for state in all_a
    )


def test_initialize_network_states_agent_ids_are_contiguous() -> None:
    states = initialize_network_states(4, probability_a=0.5, rng=random.Random(11))
    assert tuple(state.agent_id for state in states) == (0, 1, 2, 3)


def test_neighbor_observations_expose_only_neighbors_in_sorted_order() -> None:
    graph = Graph({0: (2, 1), 1: (0,), 2: (0,)})
    states = (
        make_state(0, "A", 0.8),
        make_state(1, "B", 0.2),
        make_state(2, "A", 0.7),
    )

    observations = build_neighbor_observations(graph, states, round_index=1)

    assert observations[0].observed_neighbor_ids == (1, 2)
    assert observations[0].observed_neighbor_actions == ("B", "A")
    assert observations[0].observed_neighbor_beliefs == (0.2, 0.7)
    assert observations[1].observed_neighbor_ids == (0,)
    assert observations[2].observed_neighbor_ids == (0,)


def test_neighbor_observations_reject_duplicate_state_ids() -> None:
    graph = Graph({0: (1,), 1: (0,)})
    states = (
        make_state(0, "A", 0.8),
        make_state(0, "B", 0.2),
    )

    with pytest.raises(ValueError, match="states must contain one state per graph node"):
        build_neighbor_observations(graph, states, round_index=1)


def test_neighbor_observations_reject_missing_graph_node_snapshot() -> None:
    graph = Graph({0: (), 1: ()})
    states = (make_state(0, "A", 0.8),)

    with pytest.raises(ValueError, match="states must contain one state per graph node"):
        build_neighbor_observations(graph, states, round_index=1)


def test_neighbor_observations_reject_extra_state_snapshot() -> None:
    graph = Graph({0: ()})
    states = (
        make_state(0, "A", 0.8),
        make_state(1, "B", 0.2),
    )

    with pytest.raises(ValueError, match="states must contain one state per graph node"):
        build_neighbor_observations(graph, states, round_index=1)


def test_neighbor_observations_sorted_order_follows_agent_id() -> None:
    graph = Graph({0: (1,), 1: (0, 2), 2: (1,)})
    states = (
        make_state(2, "A", 0.7),
        make_state(0, "A", 0.8),
        make_state(1, "B", 0.2),
    )

    observations = build_neighbor_observations(graph, states, round_index=1)

    assert tuple(observation.agent_id for observation in observations) == (0, 1, 2)


def test_neighbor_observations_use_previous_round_state_snapshot() -> None:
    graph = Graph({0: (1,), 1: (0,)})
    states = (
        make_state(0, "A", 0.8),
        make_state(1, "B", 0.2),
    )

    observations = build_neighbor_observations(graph, states, round_index=3)

    assert observations[0].current_action == "A"
    assert observations[0].observed_neighbor_actions == ("B",)
    assert observations[1].current_action == "B"
    assert observations[1].observed_neighbor_actions == ("A",)
