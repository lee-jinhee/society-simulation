import random

import pytest

from society_simulation.config import TopologyConfig
from society_simulation.graph import Graph, build_graph


def test_complete_graph_connects_every_pair() -> None:
    graph = build_graph(TopologyConfig(type="complete"), num_agents=4, rng=random.Random(1))

    assert graph.neighbors(0) == (1, 2, 3)
    assert graph.neighbors(1) == (0, 2, 3)
    assert graph.edges() == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def test_cycle_graph_connects_local_neighbors() -> None:
    graph = build_graph(TopologyConfig(type="cycle"), num_agents=5, rng=random.Random(1))

    assert graph.neighbors(0) == (1, 4)
    assert graph.neighbors(2) == (1, 3)


def test_erdos_renyi_graph_is_deterministic() -> None:
    config = TopologyConfig(type="erdos_renyi", edge_probability=0.4)

    first = build_graph(config, num_agents=8, rng=random.Random(99))
    second = build_graph(config, num_agents=8, rng=random.Random(99))

    assert first.to_dict() == second.to_dict()


def test_small_world_graph_is_deterministic_and_undirected() -> None:
    config = TopologyConfig(type="small_world", degree=4, rewiring_probability=0.2)

    graph = build_graph(config, num_agents=10, rng=random.Random(7))

    for node, neighbors in graph.adjacency.items():
        assert node not in neighbors
        for neighbor in neighbors:
            assert node in graph.neighbors(neighbor)
    assert graph.to_dict() == build_graph(config, num_agents=10, rng=random.Random(7)).to_dict()


def test_graph_rejects_self_edges() -> None:
    with pytest.raises(ValueError, match="graph cannot contain self edges"):
        Graph({0: (0,)})


def test_graph_rejects_asymmetric_edges() -> None:
    with pytest.raises(ValueError, match="graph edges must be undirected"):
        Graph({0: (1,), 1: ()})


def test_graph_rejects_non_contiguous_node_ids() -> None:
    with pytest.raises(ValueError, match="graph node ids must be contiguous from 0"):
        Graph({2: (3,), 3: (2,)})


def test_graph_rejects_negative_node_ids() -> None:
    with pytest.raises(ValueError, match="graph node ids must be contiguous from 0"):
        Graph({-1: (0,), 0: (-1,)})
