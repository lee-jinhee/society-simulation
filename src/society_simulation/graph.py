from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from society_simulation.config import TopologyConfig


@dataclass(frozen=True)
class Graph:
    adjacency: dict[int, tuple[int, ...]]
    topology: dict[str, object] | None = None

    def __post_init__(self) -> None:
        normalized = {}
        for node, neighbors in self.adjacency.items():
            normalized[node] = tuple(sorted(set(neighbors)))

        object.__setattr__(self, "adjacency", dict(sorted(normalized.items())))
        nodes = set(self.adjacency)
        if nodes != set(range(len(nodes))):
            raise ValueError("graph node ids must be contiguous from 0")

        for node, neighbors in self.adjacency.items():
            if node in neighbors:
                raise ValueError("graph cannot contain self edges")

        for node, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                if neighbor not in nodes:
                    raise ValueError("graph references missing nodes")
                if node not in self.adjacency[neighbor]:
                    raise ValueError("graph edges must be undirected")

    def neighbors(self, node: int) -> tuple[int, ...]:
        return self.adjacency[node]

    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            sorted(
                (node, neighbor)
                for node, neighbors in self.adjacency.items()
                for neighbor in neighbors
                if node < neighbor
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology,
            "adjacency": {
                str(node): list(neighbors)
                for node, neighbors in sorted(self.adjacency.items())
            },
        }


def _complete_adjacency(num_agents: int) -> dict[int, tuple[int, ...]]:
    return {
        node: tuple(neighbor for neighbor in range(num_agents) if neighbor != node)
        for node in range(num_agents)
    }


def _cycle_adjacency(num_agents: int) -> dict[int, tuple[int, ...]]:
    if num_agents < 3:
        raise ValueError("cycle topology requires at least 3 agents")
    return {
        node: tuple(sorted(((node - 1) % num_agents, (node + 1) % num_agents)))
        for node in range(num_agents)
    }


def _erdos_renyi_adjacency(num_agents: int, edge_probability: float, rng: random.Random) -> dict[int, tuple[int, ...]]:
    adjacency: dict[int, set[int]] = {node: set() for node in range(num_agents)}
    for node_a in range(num_agents):
        for node_b in range(node_a + 1, num_agents):
            if rng.random() < edge_probability:
                adjacency[node_a].add(node_b)
                adjacency[node_b].add(node_a)
    return {node: tuple(sorted(neighbors)) for node, neighbors in adjacency.items()}


def _small_world_adjacency(
    num_agents: int, degree: int, rewiring_probability: float, rng: random.Random
) -> dict[int, tuple[int, ...]]:
    if degree is None:
        raise ValueError("small_world degree must be a positive even integer")
    if rewiring_probability is None:
        raise ValueError("rewiring_probability must be between 0 and 1")
    if not isinstance(degree, int) or degree % 2 != 0 or degree <= 0:
        raise ValueError("small_world degree must be a positive even integer")
    if degree >= num_agents:
        raise ValueError("small_world degree must be less than num_agents")

    half_degree = degree // 2
    adjacency: dict[int, set[int]] = {node: set() for node in range(num_agents)}
    edges: set[tuple[int, int]] = set()

    for node in range(num_agents):
        for step in range(1, half_degree + 1):
            neighbor = (node + step) % num_agents
            node_a, node_b = sorted((node, neighbor))
            if node_b not in adjacency[node_a]:
                adjacency[node_a].add(node_b)
                adjacency[node_b].add(node_a)
                edges.add((node_a, node_b))

    for node_a, node_b in sorted(edges):
        if rng.random() >= rewiring_probability:
            continue
        candidates = tuple(
            neighbor
            for neighbor in range(num_agents)
            if neighbor != node_a and neighbor not in adjacency[node_a]
        )
        if not candidates:
            continue
        new_neighbor = rng.choice(candidates)
        adjacency[node_a].remove(node_b)
        adjacency[node_b].remove(node_a)
        adjacency[node_a].add(new_neighbor)
        adjacency[new_neighbor].add(node_a)

    return {node: tuple(sorted(neighbors)) for node, neighbors in adjacency.items()}


def build_graph(
    config: TopologyConfig, num_agents: int, rng: random.Random
) -> Graph:
    if config.type == "complete":
        adjacency: dict[int, tuple[int, ...]] = _complete_adjacency(num_agents)
    elif config.type == "cycle":
        adjacency = _cycle_adjacency(num_agents)
    elif config.type == "erdos_renyi":
        edge_probability = config.edge_probability
        if edge_probability is None:
            raise ValueError("erdos_renyi edge_probability must be between 0 and 1")
        adjacency = _erdos_renyi_adjacency(num_agents, edge_probability, rng)
    elif config.type == "small_world":
        adjacency = _small_world_adjacency(
            num_agents, config.degree, config.rewiring_probability, rng
        )
    else:
        raise ValueError("unsupported topology type")

    topology = asdict(config)
    return Graph(adjacency=adjacency, topology=topology)
