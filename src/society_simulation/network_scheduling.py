from __future__ import annotations

import random
from collections.abc import Iterable

from society_simulation.graph import Graph
from society_simulation.models import Action
from society_simulation.network_models import NetworkAgentState, NetworkObservation
from society_simulation.policies import confidence_from_belief


def initialize_network_states(
    num_agents: int,
    probability_a: float,
    rng: random.Random,
) -> tuple[NetworkAgentState, ...]:
    if num_agents <= 0:
        raise ValueError("num_agents must be positive")
    if probability_a < 0.0 or probability_a > 1.0:
        raise ValueError("probability_a must be between 0 and 1")

    states: list[NetworkAgentState] = []
    for agent_id in range(num_agents):
        action: Action = "A" if rng.random() < probability_a else "B"
        belief_probability = 1.0 if action == "A" else 0.0
        states.append(
            NetworkAgentState(
                agent_id=agent_id,
                belief_probability=belief_probability,
                confidence=confidence_from_belief(belief_probability),
                action=action,
                round_index=0,
                observed_neighbor_ids=(),
                observed_neighbor_actions=(),
            )
        )
    return tuple(states)


def build_neighbor_observations(
    graph: Graph,
    states: Iterable[NetworkAgentState],
    round_index: int,
) -> tuple[NetworkObservation, ...]:
    states_snapshot = tuple(states)
    state_ids = tuple(state.agent_id for state in states_snapshot)
    graph_node_ids = set(graph.adjacency)
    if len(state_ids) != len(set(state_ids)) or set(state_ids) != graph_node_ids:
        raise ValueError("states must contain one state per graph node")
    if len(state_ids) != len(graph_node_ids):
        raise ValueError("states must contain one state per graph node")

    states_by_id = {state.agent_id: state for state in states_snapshot}

    observations: list[NetworkObservation] = []
    for agent_id in sorted(states_by_id):
        state = states_by_id[agent_id]
        observed_neighbor_ids = tuple(graph.neighbors(state.agent_id))
        observations.append(
            NetworkObservation(
                agent_id=state.agent_id,
                round_index=round_index,
                current_belief_probability=state.belief_probability,
                current_action=state.action,
                observed_neighbor_ids=observed_neighbor_ids,
                observed_neighbor_actions=tuple(
                    states_by_id[neighbor_id].action for neighbor_id in observed_neighbor_ids
                ),
                observed_neighbor_beliefs=tuple(
                    states_by_id[neighbor_id].belief_probability
                    for neighbor_id in observed_neighbor_ids
                ),
            )
        )

    return tuple(observations)
