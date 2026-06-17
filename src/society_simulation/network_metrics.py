from __future__ import annotations

from collections import deque
from statistics import mean, pvariance
from typing import Any

from society_simulation.graph import Graph
from society_simulation.models import Action
from society_simulation.network_models import NetworkAgentState

_STATE_SNAPSHOT_ERROR = "states must contain one state per graph node"
_PREVIOUS_STATE_SNAPSHOT_ERROR = "previous_states must contain one state per graph node"
_ROUND_HISTORY_ERROR = "rounds must contain one state per graph node"
_EMPTY_ROUND_ERROR = "rounds must not contain empty rounds"


def compute_round_metrics(
    graph: Graph,
    states: tuple[NetworkAgentState, ...],
    previous_states: tuple[NetworkAgentState, ...] | None = None,
) -> dict[str, Any]:
    if not states:
        raise ValueError("states must not be empty")

    _validate_state_snapshot(graph, states, _STATE_SNAPSHOT_ERROR)
    beliefs = [state.belief_probability for state in states]
    actions = [state.action for state in states]
    action_changes = 0

    if previous_states is not None:
        previous_actions = _validate_state_snapshot(
            graph,
            previous_states,
            _PREVIOUS_STATE_SNAPSHOT_ERROR,
        )
        action_changes = sum(
            1 for state in states if previous_actions[state.agent_id].action != state.action
        )

    return {
        "round_index": states[0].round_index,
        "a_fraction": actions.count("A") / len(actions),
        "belief_mean": mean(beliefs),
        "belief_variance": pvariance(beliefs) if len(beliefs) > 1 else 0.0,
        "edge_disagreement_rate": edge_disagreement_rate(graph, states),
        "action_changes": action_changes,
    }


def compute_final_network_metrics(
    graph: Graph,
    rounds: tuple[tuple[NetworkAgentState, ...], ...],
) -> dict[str, Any]:
    if not rounds:
        raise ValueError("rounds must not be empty")

    validated_rounds = _validate_round_history(graph, rounds)
    final_states = validated_rounds[-1]
    final_actions = [state.action for state in final_states]
    final_beliefs = [state.belief_probability for state in final_states]
    consensus_action = _consensus_action(final_states)

    return {
        "final_action_counts": {
            "A": final_actions.count("A"),
            "B": final_actions.count("B"),
        },
        "final_a_fraction": final_actions.count("A") / len(final_actions),
        "consensus_reached": consensus_action is not None,
        "consensus_action": consensus_action,
        "time_to_consensus": _time_to_consensus(validated_rounds),
        "polarization_index": polarization_index(final_beliefs),
        "opinion_variance": pvariance(final_beliefs) if len(final_beliefs) > 1 else 0.0,
        "mean_belief": mean(final_beliefs),
        "edge_disagreement_rate": edge_disagreement_rate(graph, final_states),
        "component_count": component_count(graph),
    }


def edge_disagreement_rate(graph: Graph, states: tuple[NetworkAgentState, ...]) -> float:
    _validate_state_snapshot(graph, states, _STATE_SNAPSHOT_ERROR)
    edges = graph.edges()
    if not edges:
        return 0.0

    action_by_id = {state.agent_id: state for state in states}
    disagreements = sum(
        1
        for source, target in edges
        if action_by_id[source].action != action_by_id[target].action
    )
    return disagreements / len(edges)


def polarization_index(beliefs: list[float]) -> float:
    lower = [belief for belief in beliefs if belief < 0.5]
    upper = [belief for belief in beliefs if belief > 0.5]
    if not lower or not upper:
        return 0.0

    away_from_center = [belief for belief in beliefs if abs(belief - 0.5) > 0.1]
    center_weight = len(away_from_center) / len(beliefs)
    return abs(mean(upper) - mean(lower)) * center_weight


def component_count(graph: Graph) -> int:
    remaining = set(graph.adjacency)
    count = 0

    while remaining:
        count += 1
        start = remaining.pop()
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph.neighbors(node):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)

    return count


def _consensus_action(states: tuple[NetworkAgentState, ...]) -> Action | None:
    actions = {state.action for state in states}
    if len(actions) == 1:
        return next(iter(actions))
    return None


def _validate_state_snapshot(
    graph: Graph,
    states: tuple[NetworkAgentState, ...],
    error_message: str,
) -> dict[int, NetworkAgentState]:
    state_ids = tuple(state.agent_id for state in states)
    graph_node_ids = set(graph.adjacency)
    if len(state_ids) != len(set(state_ids)) or set(state_ids) != graph_node_ids:
        raise ValueError(error_message)
    return {state.agent_id: state for state in states}


def _validate_round_history(
    graph: Graph,
    rounds: tuple[tuple[NetworkAgentState, ...], ...],
) -> tuple[tuple[NetworkAgentState, ...], ...]:
    validated_rounds: list[tuple[NetworkAgentState, ...]] = []
    for states in rounds:
        if not states:
            raise ValueError(_EMPTY_ROUND_ERROR)
        _validate_state_snapshot(graph, states, _ROUND_HISTORY_ERROR)
        validated_rounds.append(states)
    return tuple(validated_rounds)


def _time_to_consensus(
    rounds: tuple[tuple[NetworkAgentState, ...], ...],
) -> int | None:
    for index, states in enumerate(rounds):
        action = _consensus_action(states)
        if action is None:
            continue
        if all(_consensus_action(later_states) == action for later_states in rounds[index:]):
            return states[0].round_index
    return None
