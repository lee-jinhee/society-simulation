import random

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
