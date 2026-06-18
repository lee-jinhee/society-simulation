from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

from society_simulation.config import NetworkHerdingConfig
from society_simulation.graph import Graph, build_graph
from society_simulation.network_metrics import compute_final_network_metrics, compute_round_metrics
from society_simulation.network_models import NetworkAgentState
from society_simulation.network_policies import build_network_update_policy
from society_simulation.network_replay import NetworkReplayWriter
from society_simulation.network_scheduling import (
    build_neighbor_observations,
    initialize_network_states,
)


@dataclass(frozen=True)
class NetworkRunResult:
    graph: Graph
    rounds: tuple[tuple[NetworkAgentState, ...], ...]
    timeseries: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    output_dir: Path


def run_network_herding(config: NetworkHerdingConfig) -> NetworkRunResult:
    config.validate()
    rng = random.Random(config.seed)
    graph = build_graph(config.topology, num_agents=config.num_agents, rng=rng)
    policy = build_network_update_policy(config.update_policy)
    initial_states = initialize_network_states(
        config.num_agents,
        probability_a=config.initial_opinion.probability_a,
        rng=rng,
    )

    rounds: list[tuple[NetworkAgentState, ...]] = [initial_states]
    timeseries: list[dict[str, Any]] = [compute_round_metrics(graph, initial_states)]
    for round_index in range(1, config.scheduler.rounds + 1):
        previous_states = rounds[-1]
        observations = build_neighbor_observations(graph, previous_states, round_index)
        next_states: list[NetworkAgentState] = []
        for observation in observations:
            decision = policy.decide(observation)
            next_states.append(
                NetworkAgentState(
                    agent_id=observation.agent_id,
                    belief_probability=decision.belief_probability,
                    confidence=decision.confidence,
                    action=decision.action,
                    round_index=round_index,
                    observed_neighbor_ids=observation.observed_neighbor_ids,
                    observed_neighbor_actions=observation.observed_neighbor_actions,
                )
            )
        current_states = tuple(next_states)
        rounds.append(current_states)
        timeseries.append(compute_round_metrics(graph, current_states, previous_states))

    frozen_rounds = tuple(rounds)
    metrics = compute_final_network_metrics(graph, frozen_rounds)
    usage_summary = getattr(policy, "usage_summary", None)
    if callable(usage_summary):
        metrics["llm_usage"] = usage_summary()
    output_dir = NetworkReplayWriter(config).write(
        graph=graph,
        rounds=frozen_rounds,
        timeseries=timeseries,
        metrics=metrics,
    )
    return NetworkRunResult(
        graph=graph,
        rounds=frozen_rounds,
        timeseries=tuple(timeseries),
        metrics=metrics,
        output_dir=output_dir,
    )
