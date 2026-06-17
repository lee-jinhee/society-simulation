# Network Herding v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic graph-local opinion dynamics scenario that can measure network herding, consensus, edge disagreement, and polarization without LLM calls.

**Architecture:** Keep the v0 sequential cascade intact and add a scenario dispatcher that routes `sequential_information_cascade` to the existing runner and `network_herding` to a new graph-local runner. Network herding uses explicit topology generation, synchronous rounds, neighbor-only observations, transparent update policies, time-series metrics, and replay artifacts.

**Tech Stack:** Python 3.11+, standard library only for runtime, `pytest` for tests. Do not add NetworkX in v1.

---

## File Structure

- Modify `src/society_simulation/config.py`: keep `ExperimentConfig` for v0; add `NetworkHerdingConfig` and nested config dataclasses; update `load_config`.
- Create `src/society_simulation/graph.py`: graph dataclass, topology validation, deterministic topology generation.
- Create `src/society_simulation/network_models.py`: network-specific state, observation, and decision dataclasses.
- Create `src/society_simulation/network_scheduling.py`: synchronous neighbor observation builder.
- Create `src/society_simulation/network_policies.py`: majority, threshold, and DeGroot policies plus factory.
- Create `src/society_simulation/network_metrics.py`: time-series and final network metrics.
- Create `src/society_simulation/network_replay.py`: network replay artifact writer.
- Create `src/society_simulation/network_runner.py`: `network_herding` run loop.
- Modify `src/society_simulation/runner.py`: add scenario dispatch while preserving existing v0 behavior.
- Modify `src/society_simulation/cli.py`: print action counts for both old and new metric schemas; keep clean error handling.
- Create `examples/network_herding.json`: default runnable v1 experiment.
- Modify `README.md`: document the network example and replay artifacts.
- Create tests:
  - `tests/test_network_config.py`
  - `tests/test_graph.py`
  - `tests/test_network_scheduling.py`
  - `tests/test_network_policies.py`
  - `tests/test_network_metrics.py`
  - `tests/test_network_replay.py`
  - `tests/test_network_runner.py`
  - update `tests/test_cli.py`

---

### Task 1: Network Config Loading and Validation

**Files:**
- Modify: `src/society_simulation/config.py`
- Create: `tests/test_network_config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing network config tests**

Create `tests/test_network_config.py`:

```python
import json
from pathlib import Path

import pytest

from society_simulation.config import NetworkHerdingConfig, load_config


def valid_network_config(tmp_path: Path) -> dict[str, object]:
    return {
        "experiment_name": "network_herding",
        "seed": 42,
        "num_agents": 12,
        "initial_opinion": {"type": "bernoulli", "probability_a": 0.45},
        "topology": {
            "type": "small_world",
            "degree": 4,
            "rewiring_probability": 0.2,
        },
        "scheduler": {"type": "synchronous_rounds", "rounds": 8},
        "observation_policy": {"type": "neighbor_actions"},
        "update_policy": {"type": "threshold", "adoption_threshold": 0.6},
        "output_dir": str(tmp_path / "network-run"),
    }


def test_load_network_herding_config(tmp_path: Path) -> None:
    config_path = tmp_path / "network.json"
    config_path.write_text(json.dumps(valid_network_config(tmp_path)), encoding="utf-8")

    config = load_config(config_path)

    assert isinstance(config, NetworkHerdingConfig)
    assert config.experiment_name == "network_herding"
    assert config.topology.type == "small_world"
    assert config.scheduler.rounds == 8
    assert config.update_policy.type == "threshold"
    assert config.output_dir == str(tmp_path / "network-run")


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("num_agents",), 0, "num_agents must be positive"),
        (("initial_opinion", "probability_a"), -0.1, "probability_a must be between 0 and 1"),
        (("initial_opinion", "type"), "fixed", "unsupported initial_opinion type"),
        (("topology", "type"), "scale_free", "unsupported topology type"),
        (("topology", "degree"), 3, "small_world degree must be a positive even integer"),
        (("topology", "degree"), 12, "small_world degree must be less than num_agents"),
        (("topology", "rewiring_probability"), 1.2, "rewiring_probability must be between 0 and 1"),
        (("scheduler", "rounds"), 0, "rounds must be positive"),
        (("scheduler", "type"), "asynchronous", "unsupported scheduler type"),
        (("observation_policy", "type"), "global_actions", "unsupported observation_policy type"),
        (("update_policy", "type"), "llm", "unsupported network update_policy type"),
        (("update_policy", "adoption_threshold"), 0.49, "adoption_threshold must be between 0.5 and 1.0"),
    ],
)
def test_network_config_rejects_invalid_values(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    data = valid_network_config(tmp_path)
    target = data
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]

    config = NetworkHerdingConfig.from_dict(data)

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_network_config_supports_complete_topology(tmp_path: Path) -> None:
    data = valid_network_config(tmp_path)
    data["topology"] = {"type": "complete"}

    config = NetworkHerdingConfig.from_dict(data)

    config.validate()
    assert config.topology.type == "complete"


def test_network_config_supports_erdos_renyi_topology(tmp_path: Path) -> None:
    data = valid_network_config(tmp_path)
    data["topology"] = {"type": "erdos_renyi", "edge_probability": 0.25}

    config = NetworkHerdingConfig.from_dict(data)

    config.validate()
    assert config.topology.edge_probability == 0.25


def test_network_config_supports_degroot_policy(tmp_path: Path) -> None:
    data = valid_network_config(tmp_path)
    data["update_policy"] = {"type": "degroot", "self_weight": 0.3}

    config = NetworkHerdingConfig.from_dict(data)

    config.validate()
    assert config.update_policy.self_weight == 0.3
```

Add this regression to `tests/test_config.py`:

```python
def test_load_config_keeps_sequential_cascade_type(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "experiment_name": "sequential_information_cascade",
          "seed": 7,
          "num_agents": 4,
          "true_state": null,
          "signal_accuracy": 0.65,
          "prior_probability": 0.5,
          "scheduler": "sequential",
          "observation_policy": "previous_actions",
          "update_policy": "simple_heuristic",
          "output_dir": "runs/example"
        }
        """,
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == "sequential_information_cascade"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_network_config.py tests/test_config.py -v
```

Expected: FAIL because `NetworkHerdingConfig` is not defined.

- [ ] **Step 3: Implement config dataclasses and validation**

Replace `src/society_simulation/config.py` with:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

Action = Literal["A", "B"]


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    seed: int
    num_agents: int
    true_state: Action | None
    signal_accuracy: float
    prior_probability: float
    scheduler: str
    observation_policy: str
    update_policy: str
    output_dir: str

    def validate(self) -> None:
        if self.experiment_name != "sequential_information_cascade":
            raise ValueError("unsupported experiment_name")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if self.true_state not in ("A", "B", None):
            raise ValueError("true_state must be A, B, or null")
        if not 0.5 <= self.signal_accuracy <= 1.0:
            raise ValueError("signal_accuracy must be between 0.5 and 1.0")
        if not 0.0 < self.prior_probability < 1.0:
            raise ValueError("prior_probability must be greater than 0 and less than 1")
        if self.scheduler != "sequential":
            raise ValueError("unsupported scheduler")
        if self.observation_policy != "previous_actions":
            raise ValueError("unsupported observation_policy")
        if self.update_policy not in ("bayesian_cascade", "simple_heuristic"):
            raise ValueError("unsupported update_policy")
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class InitialOpinionConfig:
    type: str
    probability_a: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InitialOpinionConfig:
        return cls(
            type=str(data.get("type", "")),
            probability_a=float(data.get("probability_a", 0.5)),
        )


@dataclass(frozen=True)
class TopologyConfig:
    type: str
    degree: int | None = None
    rewiring_probability: float | None = None
    edge_probability: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopologyConfig:
        return cls(
            type=str(data.get("type", "")),
            degree=data.get("degree"),
            rewiring_probability=data.get("rewiring_probability"),
            edge_probability=data.get("edge_probability"),
        )


@dataclass(frozen=True)
class NetworkSchedulerConfig:
    type: str
    rounds: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NetworkSchedulerConfig:
        return cls(type=str(data.get("type", "")), rounds=int(data.get("rounds", 0)))


@dataclass(frozen=True)
class NetworkObservationPolicyConfig:
    type: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NetworkObservationPolicyConfig:
        return cls(type=str(data.get("type", "")))


@dataclass(frozen=True)
class NetworkUpdatePolicyConfig:
    type: str
    adoption_threshold: float | None = None
    self_weight: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NetworkUpdatePolicyConfig:
        return cls(
            type=str(data.get("type", "")),
            adoption_threshold=data.get("adoption_threshold"),
            self_weight=data.get("self_weight"),
        )


@dataclass(frozen=True)
class NetworkHerdingConfig:
    experiment_name: str
    seed: int
    num_agents: int
    initial_opinion: InitialOpinionConfig
    topology: TopologyConfig
    scheduler: NetworkSchedulerConfig
    observation_policy: NetworkObservationPolicyConfig
    update_policy: NetworkUpdatePolicyConfig
    output_dir: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NetworkHerdingConfig:
        return cls(
            experiment_name=str(data.get("experiment_name", "")),
            seed=int(data.get("seed", 0)),
            num_agents=int(data.get("num_agents", 0)),
            initial_opinion=InitialOpinionConfig.from_dict(data.get("initial_opinion", {})),
            topology=TopologyConfig.from_dict(data.get("topology", {})),
            scheduler=NetworkSchedulerConfig.from_dict(data.get("scheduler", {})),
            observation_policy=NetworkObservationPolicyConfig.from_dict(
                data.get("observation_policy", {})
            ),
            update_policy=NetworkUpdatePolicyConfig.from_dict(data.get("update_policy", {})),
            output_dir=str(data.get("output_dir", "")),
        )

    def validate(self) -> None:
        if self.experiment_name != "network_herding":
            raise ValueError("unsupported experiment_name")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")
        self._validate_initial_opinion()
        self._validate_topology()
        self._validate_scheduler()
        self._validate_observation_policy()
        self._validate_update_policy()

    def _validate_initial_opinion(self) -> None:
        if self.initial_opinion.type != "bernoulli":
            raise ValueError("unsupported initial_opinion type")
        if not 0.0 <= self.initial_opinion.probability_a <= 1.0:
            raise ValueError("probability_a must be between 0 and 1")

    def _validate_topology(self) -> None:
        topology = self.topology
        if topology.type == "complete":
            return
        if topology.type == "cycle":
            if self.num_agents < 3:
                raise ValueError("cycle topology requires at least 3 agents")
            return
        if topology.type == "erdos_renyi":
            if topology.edge_probability is None:
                raise ValueError("edge_probability is required for erdos_renyi")
            if not 0.0 <= float(topology.edge_probability) <= 1.0:
                raise ValueError("edge_probability must be between 0 and 1")
            return
        if topology.type == "small_world":
            if topology.degree is None:
                raise ValueError("small_world degree is required")
            if topology.degree <= 0 or topology.degree % 2 != 0:
                raise ValueError("small_world degree must be a positive even integer")
            if topology.degree >= self.num_agents:
                raise ValueError("small_world degree must be less than num_agents")
            if topology.rewiring_probability is None:
                raise ValueError("rewiring_probability is required for small_world")
            if not 0.0 <= float(topology.rewiring_probability) <= 1.0:
                raise ValueError("rewiring_probability must be between 0 and 1")
            return
        raise ValueError("unsupported topology type")

    def _validate_scheduler(self) -> None:
        if self.scheduler.type != "synchronous_rounds":
            raise ValueError("unsupported scheduler type")
        if self.scheduler.rounds <= 0:
            raise ValueError("rounds must be positive")

    def _validate_observation_policy(self) -> None:
        if self.observation_policy.type != "neighbor_actions":
            raise ValueError("unsupported observation_policy type")

    def _validate_update_policy(self) -> None:
        policy = self.update_policy
        if policy.type == "majority_rule":
            return
        if policy.type == "threshold":
            if policy.adoption_threshold is None:
                raise ValueError("adoption_threshold is required for threshold policy")
            if not 0.5 <= float(policy.adoption_threshold) <= 1.0:
                raise ValueError("adoption_threshold must be between 0.5 and 1.0")
            return
        if policy.type == "degroot":
            if policy.self_weight is None:
                raise ValueError("self_weight is required for degroot policy")
            if not 0.0 <= float(policy.self_weight) <= 1.0:
                raise ValueError("self_weight must be between 0 and 1")
            return
        raise ValueError("unsupported network update_policy type")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


Config = ExperimentConfig | NetworkHerdingConfig


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_name = data.get("experiment_name")
    if experiment_name == "network_herding":
        config = NetworkHerdingConfig.from_dict(data)
    else:
        config = ExperimentConfig(**data)
    config.validate()
    return config
```

- [ ] **Step 4: Run config tests**

Run:

```bash
pytest tests/test_network_config.py tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/config.py tests/test_network_config.py tests/test_config.py
git commit -m "feat: add network herding config"
```

---

### Task 2: Graph Topology Primitives

**Files:**
- Create: `src/society_simulation/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write failing graph tests**

Create `tests/test_graph.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_graph.py -v
```

Expected: FAIL because `society_simulation.graph` does not exist.

- [ ] **Step 3: Implement graph primitives**

Create `src/society_simulation/graph.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import random

from society_simulation.config import TopologyConfig


@dataclass(frozen=True)
class Graph:
    adjacency: dict[int, tuple[int, ...]]
    topology: dict[str, object] | None = None

    def __post_init__(self) -> None:
        normalized = {
            int(node): tuple(sorted(int(neighbor) for neighbor in neighbors))
            for node, neighbors in self.adjacency.items()
        }
        object.__setattr__(self, "adjacency", normalized)
        self._validate()

    def _validate(self) -> None:
        for node, neighbors in self.adjacency.items():
            if node in neighbors:
                raise ValueError("graph cannot contain self edges")
            for neighbor in neighbors:
                if neighbor not in self.adjacency:
                    raise ValueError("graph neighbor is missing from adjacency")
                if node not in self.adjacency[neighbor]:
                    raise ValueError("graph edges must be undirected")

    def neighbors(self, node: int) -> tuple[int, ...]:
        return self.adjacency[node]

    def edges(self) -> tuple[tuple[int, int], ...]:
        edges: set[tuple[int, int]] = set()
        for node, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                edges.add((min(node, neighbor), max(node, neighbor)))
        return tuple(sorted(edges))

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology or {},
            "adjacency": {str(node): list(neighbors) for node, neighbors in sorted(self.adjacency.items())},
        }


def build_graph(config: TopologyConfig, num_agents: int, rng: random.Random) -> Graph:
    if config.type == "complete":
        adjacency = {
            node: tuple(other for other in range(num_agents) if other != node)
            for node in range(num_agents)
        }
    elif config.type == "cycle":
        adjacency = {
            node: tuple(sorted({(node - 1) % num_agents, (node + 1) % num_agents}))
            for node in range(num_agents)
        }
    elif config.type == "erdos_renyi":
        adjacency = _build_erdos_renyi(num_agents, float(config.edge_probability), rng)
    elif config.type == "small_world":
        adjacency = _build_small_world(
            num_agents=num_agents,
            degree=int(config.degree),
            rewiring_probability=float(config.rewiring_probability),
            rng=rng,
        )
    else:
        raise ValueError("unsupported topology type")
    return Graph(adjacency, topology=_topology_to_dict(config))


def _build_erdos_renyi(
    num_agents: int,
    edge_probability: float,
    rng: random.Random,
) -> dict[int, tuple[int, ...]]:
    neighbors = {node: set() for node in range(num_agents)}
    for source in range(num_agents):
        for target in range(source + 1, num_agents):
            if rng.random() < edge_probability:
                neighbors[source].add(target)
                neighbors[target].add(source)
    return _freeze_neighbors(neighbors)


def _build_small_world(
    num_agents: int,
    degree: int,
    rewiring_probability: float,
    rng: random.Random,
) -> dict[int, tuple[int, ...]]:
    neighbors = {node: set() for node in range(num_agents)}
    half_degree = degree // 2
    base_edges: list[tuple[int, int]] = []
    for source in range(num_agents):
        for offset in range(1, half_degree + 1):
            target = (source + offset) % num_agents
            edge = (min(source, target), max(source, target))
            if edge not in base_edges:
                base_edges.append(edge)

    for source, target in sorted(base_edges):
        if rng.random() < rewiring_probability:
            candidates = [
                candidate
                for candidate in range(num_agents)
                if candidate != source and candidate not in neighbors[source]
            ]
            replacement = rng.choice(candidates)
            neighbors[source].add(replacement)
            neighbors[replacement].add(source)
        else:
            neighbors[source].add(target)
            neighbors[target].add(source)
    return _freeze_neighbors(neighbors)


def _freeze_neighbors(neighbors: dict[int, set[int]]) -> dict[int, tuple[int, ...]]:
    return {node: tuple(sorted(values)) for node, values in sorted(neighbors.items())}


def _topology_to_dict(config: TopologyConfig) -> dict[str, object]:
    payload: dict[str, object] = {"type": config.type}
    if config.degree is not None:
        payload["degree"] = config.degree
    if config.rewiring_probability is not None:
        payload["rewiring_probability"] = config.rewiring_probability
    if config.edge_probability is not None:
        payload["edge_probability"] = config.edge_probability
    return payload
```

- [ ] **Step 4: Run graph tests**

Run:

```bash
pytest tests/test_graph.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/graph.py tests/test_graph.py
git commit -m "feat: add deterministic graph topologies"
```

---

### Task 3: Network State, Synchronous Observations, and Initial Opinions

**Files:**
- Create: `src/society_simulation/network_models.py`
- Create: `src/society_simulation/network_scheduling.py`
- Create: `tests/test_network_scheduling.py`

- [ ] **Step 1: Write failing scheduling tests**

Create `tests/test_network_scheduling.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_network_scheduling.py -v
```

Expected: FAIL because network model modules do not exist.

- [ ] **Step 3: Implement network models**

Create `src/society_simulation/network_models.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass

from society_simulation.models import Action


@dataclass(frozen=True)
class NetworkAgentState:
    agent_id: int
    belief_probability: float
    confidence: float
    action: Action
    round_index: int
    observed_neighbor_ids: tuple[int, ...]
    observed_neighbor_actions: tuple[Action, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NetworkObservation:
    agent_id: int
    round_index: int
    current_belief_probability: float
    current_action: Action
    observed_neighbor_ids: tuple[int, ...]
    observed_neighbor_actions: tuple[Action, ...]
    observed_neighbor_beliefs: tuple[float, ...]


@dataclass(frozen=True)
class NetworkDecision:
    belief_probability: float
    confidence: float
    action: Action
```

- [ ] **Step 4: Implement synchronous observation building**

Create `src/society_simulation/network_scheduling.py`:

```python
from __future__ import annotations

import random

from society_simulation.graph import Graph
from society_simulation.models import Action
from society_simulation.network_models import NetworkAgentState, NetworkObservation
from society_simulation.policies import confidence_from_belief


def initialize_network_states(
    num_agents: int,
    probability_a: float,
    rng: random.Random,
) -> tuple[NetworkAgentState, ...]:
    states: list[NetworkAgentState] = []
    for agent_id in range(num_agents):
        action: Action = "A" if rng.random() < probability_a else "B"
        belief = 1.0 if action == "A" else 0.0
        states.append(
            NetworkAgentState(
                agent_id=agent_id,
                belief_probability=belief,
                confidence=confidence_from_belief(belief),
                action=action,
                round_index=0,
                observed_neighbor_ids=(),
                observed_neighbor_actions=(),
            )
        )
    return tuple(states)


def build_neighbor_observations(
    graph: Graph,
    states: tuple[NetworkAgentState, ...],
    round_index: int,
) -> tuple[NetworkObservation, ...]:
    state_by_id = {state.agent_id: state for state in states}
    observations: list[NetworkObservation] = []
    for state in sorted(states, key=lambda item: item.agent_id):
        neighbor_ids = graph.neighbors(state.agent_id)
        neighbor_states = tuple(state_by_id[neighbor_id] for neighbor_id in neighbor_ids)
        observations.append(
            NetworkObservation(
                agent_id=state.agent_id,
                round_index=round_index,
                current_belief_probability=state.belief_probability,
                current_action=state.action,
                observed_neighbor_ids=neighbor_ids,
                observed_neighbor_actions=tuple(neighbor.action for neighbor in neighbor_states),
                observed_neighbor_beliefs=tuple(
                    neighbor.belief_probability for neighbor in neighbor_states
                ),
            )
        )
    return tuple(observations)
```

- [ ] **Step 5: Run scheduling tests**

Run:

```bash
pytest tests/test_network_scheduling.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/society_simulation/network_models.py src/society_simulation/network_scheduling.py tests/test_network_scheduling.py
git commit -m "feat: add network observations"
```

---

### Task 4: Network Update Policies

**Files:**
- Create: `src/society_simulation/network_policies.py`
- Create: `tests/test_network_policies.py`

- [ ] **Step 1: Write failing policy tests**

Create `tests/test_network_policies.py`:

```python
import pytest

from society_simulation.config import NetworkUpdatePolicyConfig
from society_simulation.network_models import NetworkObservation
from society_simulation.network_policies import (
    DeGrootPolicy,
    MajorityRulePolicy,
    ThresholdPolicy,
    build_network_update_policy,
)


def observation(
    actions: tuple[str, ...],
    beliefs: tuple[float, ...],
    current_action: str = "B",
    current_belief: float = 0.25,
) -> NetworkObservation:
    return NetworkObservation(
        agent_id=0,
        round_index=1,
        current_belief_probability=current_belief,
        current_action=current_action,  # type: ignore[arg-type]
        observed_neighbor_ids=tuple(range(len(actions))),
        observed_neighbor_actions=actions,  # type: ignore[arg-type]
        observed_neighbor_beliefs=beliefs,
    )


def test_majority_rule_chooses_neighbor_majority() -> None:
    policy = MajorityRulePolicy()

    decision = policy.decide(observation(("A", "A", "B"), (0.9, 0.8, 0.2)))

    assert decision.action == "A"
    assert decision.belief_probability == pytest.approx(2 / 3)


def test_majority_rule_tie_keeps_current_action() -> None:
    policy = MajorityRulePolicy()

    decision = policy.decide(observation(("A", "B"), (0.9, 0.1), current_action="B"))

    assert decision.action == "B"
    assert decision.belief_probability == pytest.approx(0.5)


def test_threshold_policy_switches_when_threshold_reached() -> None:
    policy = ThresholdPolicy(adoption_threshold=0.75)

    decision = policy.decide(observation(("A", "A", "A", "B"), (0.9, 0.8, 0.7, 0.1)))

    assert decision.action == "A"
    assert decision.belief_probability == pytest.approx(0.75)


def test_threshold_policy_keeps_current_action_when_no_side_reaches_threshold() -> None:
    policy = ThresholdPolicy(adoption_threshold=0.75)

    decision = policy.decide(observation(("A", "B", "B", "A"), (0.9, 0.1, 0.2, 0.8)))

    assert decision.action == "B"
    assert decision.belief_probability == pytest.approx(0.5)


def test_degroot_policy_averages_current_and_neighbor_beliefs() -> None:
    policy = DeGrootPolicy(self_weight=0.25)

    decision = policy.decide(observation(("A", "B"), (0.8, 0.4), current_belief=0.2))

    assert decision.belief_probability == pytest.approx(0.25 * 0.2 + 0.75 * 0.6)
    assert decision.action == "A"


def test_policy_factory_builds_supported_policies() -> None:
    assert isinstance(
        build_network_update_policy(NetworkUpdatePolicyConfig(type="majority_rule")),
        MajorityRulePolicy,
    )
    assert isinstance(
        build_network_update_policy(
            NetworkUpdatePolicyConfig(type="threshold", adoption_threshold=0.6)
        ),
        ThresholdPolicy,
    )
    assert isinstance(
        build_network_update_policy(NetworkUpdatePolicyConfig(type="degroot", self_weight=0.4)),
        DeGrootPolicy,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_network_policies.py -v
```

Expected: FAIL because `network_policies.py` does not exist.

- [ ] **Step 3: Implement network policies**

Create `src/society_simulation/network_policies.py`:

```python
from __future__ import annotations

from statistics import mean

from society_simulation.config import NetworkUpdatePolicyConfig
from society_simulation.models import Action
from society_simulation.network_models import NetworkDecision, NetworkObservation
from society_simulation.policies import confidence_from_belief


class MajorityRulePolicy:
    name = "majority_rule"

    def decide(self, observation: NetworkObservation) -> NetworkDecision:
        if not observation.observed_neighbor_actions:
            belief = observation.current_belief_probability
            return NetworkDecision(
                belief_probability=belief,
                confidence=confidence_from_belief(belief),
                action=observation.current_action,
            )
        a_fraction = observation.observed_neighbor_actions.count("A") / len(
            observation.observed_neighbor_actions
        )
        if a_fraction > 0.5:
            action: Action = "A"
        elif a_fraction < 0.5:
            action = "B"
        else:
            action = observation.current_action
        return NetworkDecision(
            belief_probability=a_fraction,
            confidence=confidence_from_belief(a_fraction),
            action=action,
        )


class ThresholdPolicy:
    name = "threshold"

    def __init__(self, adoption_threshold: float) -> None:
        if not 0.5 <= adoption_threshold <= 1.0:
            raise ValueError("adoption_threshold must be between 0.5 and 1.0")
        self.adoption_threshold = adoption_threshold

    def decide(self, observation: NetworkObservation) -> NetworkDecision:
        if not observation.observed_neighbor_actions:
            belief = observation.current_belief_probability
            return NetworkDecision(
                belief_probability=belief,
                confidence=confidence_from_belief(belief),
                action=observation.current_action,
            )
        a_fraction = observation.observed_neighbor_actions.count("A") / len(
            observation.observed_neighbor_actions
        )
        b_fraction = 1.0 - a_fraction
        if a_fraction >= self.adoption_threshold:
            action: Action = "A"
        elif b_fraction >= self.adoption_threshold:
            action = "B"
        else:
            action = observation.current_action
        return NetworkDecision(
            belief_probability=a_fraction,
            confidence=confidence_from_belief(a_fraction),
            action=action,
        )


class DeGrootPolicy:
    name = "degroot"

    def __init__(self, self_weight: float) -> None:
        if not 0.0 <= self_weight <= 1.0:
            raise ValueError("self_weight must be between 0 and 1")
        self.self_weight = self_weight

    def decide(self, observation: NetworkObservation) -> NetworkDecision:
        if observation.observed_neighbor_beliefs:
            neighbor_mean = mean(observation.observed_neighbor_beliefs)
        else:
            neighbor_mean = observation.current_belief_probability
        belief = (
            self.self_weight * observation.current_belief_probability
            + (1.0 - self.self_weight) * neighbor_mean
        )
        action: Action = "A" if belief >= 0.5 else "B"
        return NetworkDecision(
            belief_probability=belief,
            confidence=confidence_from_belief(belief),
            action=action,
        )


NetworkUpdatePolicy = MajorityRulePolicy | ThresholdPolicy | DeGrootPolicy


def build_network_update_policy(config: NetworkUpdatePolicyConfig) -> NetworkUpdatePolicy:
    if config.type == "majority_rule":
        return MajorityRulePolicy()
    if config.type == "threshold":
        return ThresholdPolicy(adoption_threshold=float(config.adoption_threshold))
    if config.type == "degroot":
        return DeGrootPolicy(self_weight=float(config.self_weight))
    raise ValueError("unsupported network update_policy type")
```

- [ ] **Step 4: Run policy tests**

Run:

```bash
pytest tests/test_network_policies.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/network_policies.py tests/test_network_policies.py
git commit -m "feat: add network update policies"
```

---

### Task 5: Network Metrics

**Files:**
- Create: `src/society_simulation/network_metrics.py`
- Create: `tests/test_network_metrics.py`

- [ ] **Step 1: Write failing metrics tests**

Create `tests/test_network_metrics.py`:

```python
import pytest

from society_simulation.graph import Graph
from society_simulation.network_metrics import compute_final_network_metrics, compute_round_metrics
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_network_metrics.py -v
```

Expected: FAIL because `network_metrics.py` does not exist.

- [ ] **Step 3: Implement network metrics**

Create `src/society_simulation/network_metrics.py`:

```python
from __future__ import annotations

from collections import deque
from statistics import mean, pvariance
from typing import Any

from society_simulation.graph import Graph
from society_simulation.models import Action
from society_simulation.network_models import NetworkAgentState


def compute_round_metrics(
    graph: Graph,
    states: tuple[NetworkAgentState, ...],
    previous_states: tuple[NetworkAgentState, ...] | None = None,
) -> dict[str, Any]:
    if not states:
        raise ValueError("states must not be empty")
    beliefs = [state.belief_probability for state in states]
    actions = [state.action for state in states]
    action_changes = 0
    if previous_states is not None:
        previous_actions = {state.agent_id: state.action for state in previous_states}
        action_changes = sum(
            1 for state in states if previous_actions[state.agent_id] != state.action
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
    final_states = rounds[-1]
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
        "time_to_consensus": _time_to_consensus(rounds),
        "polarization_index": polarization_index(final_beliefs),
        "opinion_variance": pvariance(final_beliefs) if len(final_beliefs) > 1 else 0.0,
        "mean_belief": mean(final_beliefs),
        "edge_disagreement_rate": edge_disagreement_rate(graph, final_states),
        "component_count": component_count(graph),
    }


def edge_disagreement_rate(graph: Graph, states: tuple[NetworkAgentState, ...]) -> float:
    edges = graph.edges()
    if not edges:
        return 0.0
    action_by_id = {state.agent_id: state.action for state in states}
    disagreements = sum(1 for source, target in edges if action_by_id[source] != action_by_id[target])
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


def _time_to_consensus(rounds: tuple[tuple[NetworkAgentState, ...], ...]) -> int | None:
    for index, states in enumerate(rounds):
        action = _consensus_action(states)
        if action is None:
            continue
        if all(_consensus_action(later_states) == action for later_states in rounds[index:]):
            return states[0].round_index
    return None
```

- [ ] **Step 4: Run metrics tests**

Run:

```bash
pytest tests/test_network_metrics.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/network_metrics.py tests/test_network_metrics.py
git commit -m "feat: add network herding metrics"
```

---

### Task 6: Network Replay Writer

**Files:**
- Create: `src/society_simulation/network_replay.py`
- Create: `tests/test_network_replay.py`

- [ ] **Step 1: Write failing replay tests**

Create `tests/test_network_replay.py`:

```python
import json
from pathlib import Path

from society_simulation.config import NetworkHerdingConfig
from society_simulation.graph import Graph
from society_simulation.network_models import NetworkAgentState
from society_simulation.network_replay import NetworkReplayWriter


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
    timeseries = [{"round_index": 0}, {"round_index": 1}]
    metrics = {
        "final_action_counts": {"A": 1, "B": 1},
        "consensus_reached": False,
        "consensus_action": None,
        "edge_disagreement_rate": 1.0,
    }

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_network_replay.py -v
```

Expected: FAIL because `network_replay.py` does not exist.

- [ ] **Step 3: Implement replay writer**

Create `src/society_simulation/network_replay.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from society_simulation.config import NetworkHerdingConfig
from society_simulation.graph import Graph
from society_simulation.network_models import NetworkAgentState


class NetworkReplayWriter:
    def __init__(self, config: NetworkHerdingConfig) -> None:
        self.config = config

    def write(
        self,
        graph: Graph,
        rounds: tuple[tuple[NetworkAgentState, ...], ...],
        timeseries: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(output_dir / "config.json", self.config.to_dict())
        self._write_json(output_dir / "graph.json", graph.to_dict())
        self._write_steps(output_dir / "steps.jsonl", rounds)
        self._write_timeseries(output_dir / "timeseries.jsonl", timeseries)
        self._write_json(output_dir / "metrics.json", metrics)
        self._write_summary(output_dir / "summary.txt", metrics)
        return output_dir

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_steps(
        self,
        path: Path,
        rounds: tuple[tuple[NetworkAgentState, ...], ...],
    ) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for previous_round, current_round in zip(rounds, rounds[1:]):
                previous_by_id = {state.agent_id: state for state in previous_round}
                for state in current_round:
                    previous = previous_by_id[state.agent_id]
                    payload = state.to_dict()
                    payload["previous_belief_probability"] = previous.belief_probability
                    payload["previous_action"] = previous.action
                    payload["update_policy"] = self.config.update_policy.type
                    payload["random_seed"] = self.config.seed
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _write_timeseries(self, path: Path, timeseries: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in timeseries:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _write_summary(self, path: Path, metrics: dict[str, Any]) -> None:
        lines = [
            f"experiment_name={self.config.experiment_name}",
            f"update_policy={self.config.update_policy.type}",
            f"seed={self.config.seed}",
            f"final_action_counts={metrics['final_action_counts']}",
            f"consensus_reached={metrics['consensus_reached']}",
            f"consensus_action={metrics['consensus_action']}",
            f"edge_disagreement_rate={metrics['edge_disagreement_rate']}",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run replay tests**

Run:

```bash
pytest tests/test_network_replay.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/network_replay.py tests/test_network_replay.py
git commit -m "feat: add network replay writer"
```

---

### Task 7: Network Runner and Scenario Dispatch

**Files:**
- Create: `src/society_simulation/network_runner.py`
- Modify: `src/society_simulation/runner.py`
- Create: `tests/test_network_runner.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write failing network runner tests**

Create `tests/test_network_runner.py`:

```python
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
```

Add this regression to `tests/test_runner.py`:

```python
def test_run_experiment_still_supports_sequential_config(tmp_path: Path) -> None:
    config = make_config(tmp_path, output_name="sequential-after-dispatch")

    result = run_experiment(config)

    assert result.true_state == "A"
    assert len(result.states) == 6
    assert (result.output_dir / "steps.jsonl").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_network_runner.py tests/test_runner.py -v
```

Expected: FAIL because `run_experiment` does not dispatch network configs.

- [ ] **Step 3: Implement network runner**

Create `src/society_simulation/network_runner.py`:

```python
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
```

- [ ] **Step 4: Add scenario dispatch while preserving v0**

Modify `src/society_simulation/runner.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

from society_simulation.config import Config, ExperimentConfig, NetworkHerdingConfig
from society_simulation.metrics import compute_metrics
from society_simulation.models import Action, AgentProfile, AgentState
from society_simulation.network_runner import NetworkRunResult, run_network_herding
from society_simulation.policies import build_update_policy
from society_simulation.replay import ReplayWriter
from society_simulation.scheduling import PreviousActionsObservation, SequentialScheduler
from society_simulation.signals import BinarySignalModel


@dataclass(frozen=True)
class RunResult:
    true_state: Action
    states: tuple[AgentState, ...]
    metrics: dict[str, Any]
    output_dir: Path


def run_experiment(config: Config) -> RunResult | NetworkRunResult:
    if isinstance(config, NetworkHerdingConfig):
        return run_network_herding(config)
    return run_sequential_information_cascade(config)


def run_sequential_information_cascade(config: ExperimentConfig) -> RunResult:
    config.validate()
    rng = random.Random(config.seed)
    signal_model = BinarySignalModel(signal_accuracy=config.signal_accuracy, rng=rng)
    true_state = config.true_state or signal_model.sample_true_state()
    private_signals = signal_model.generate_private_signals(true_state, config.num_agents)
    profiles = [
        AgentProfile(agent_id=agent_id, prior_probability=config.prior_probability)
        for agent_id in range(config.num_agents)
    ]
    scheduler = SequentialScheduler(config.num_agents)
    observation_policy = PreviousActionsObservation()
    update_policy = build_update_policy(
        config.update_policy,
        signal_accuracy=config.signal_accuracy,
        prior_probability=config.prior_probability,
    )

    states: list[AgentState] = []
    for step_index, agent_id in enumerate(scheduler):
        profile = profiles[agent_id]
        observation = observation_policy.build(
            agent_id=agent_id,
            private_signal=private_signals[agent_id],
            prior_states=states,
        )
        decision = update_policy.decide(observation)
        states.append(
            AgentState(
                agent_id=profile.agent_id,
                private_signal=observation.private_signal,
                belief_probability=decision.belief_probability,
                confidence=decision.confidence,
                action=decision.action,
                step_index=step_index,
                observed_actions=observation.observed_actions,
            )
        )

    metrics = compute_metrics(states, true_state=true_state)
    output_dir = ReplayWriter(config).write(
        true_state=true_state,
        states=states,
        metrics=metrics,
    )
    return RunResult(
        true_state=true_state,
        states=tuple(states),
        metrics=metrics,
        output_dir=output_dir,
    )
```

- [ ] **Step 5: Run runner tests**

Run:

```bash
pytest tests/test_network_runner.py tests/test_runner.py -v
```

Expected: PASS.

- [ ] **Step 6: Run broader component tests**

Run:

```bash
pytest tests/test_network_config.py tests/test_graph.py tests/test_network_scheduling.py tests/test_network_policies.py tests/test_network_metrics.py tests/test_network_replay.py tests/test_network_runner.py tests/test_runner.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/society_simulation/network_runner.py src/society_simulation/runner.py tests/test_network_runner.py tests/test_runner.py
git commit -m "feat: add network herding runner"
```

---

### Task 8: CLI, Example Config, and README

**Files:**
- Modify: `src/society_simulation/cli.py`
- Create: `examples/network_herding.json`
- Modify: `README.md`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI/example tests**

Add to `tests/test_cli.py`:

```python
def test_cli_runs_network_herding_config_and_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "network.json"
    output_dir = tmp_path / "network-run"
    config_path.write_text(
        json.dumps(
            {
                "experiment_name": "network_herding",
                "seed": 5,
                "num_agents": 6,
                "initial_opinion": {"type": "bernoulli", "probability_a": 0.5},
                "topology": {"type": "cycle"},
                "scheduler": {"type": "synchronous_rounds", "rounds": 2},
                "observation_policy": {"type": "neighbor_actions"},
                "update_policy": {"type": "majority_rule"},
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["run", str(config_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "experiment=network_herding" in output
    assert "action_counts=" in output
    assert "output_dir=" in output
    assert (output_dir / "graph.json").exists()
    assert (output_dir / "timeseries.jsonl").exists()


def test_example_network_config_exists_and_is_valid() -> None:
    from society_simulation.config import load_config, NetworkHerdingConfig

    config = load_config("examples/network_herding.json")

    assert isinstance(config, NetworkHerdingConfig)
    assert config.experiment_name == "network_herding"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL because `examples/network_herding.json` does not exist and CLI expects `metrics["action_counts"]`.

- [ ] **Step 3: Update CLI summary for both metric schemas**

Modify the successful run section in `src/society_simulation/cli.py`:

```python
    config = load_config(args.config)
    try:
        result = run_experiment(config)
    except (OSError, ValueError) as exc:
        parser.error(f"Experiment run failed for '{args.config}': {exc}")
    metrics = result.metrics
    action_counts = metrics.get("action_counts", metrics.get("final_action_counts"))

    print(f"experiment={config.experiment_name}")
    if hasattr(result, "true_state"):
        print(f"true_state={result.true_state}")
    print(f"action_counts={action_counts}")
    if "correct_cascade" in metrics:
        print(f"correct_cascade={metrics['correct_cascade']}")
    if "wrong_cascade" in metrics:
        print(f"wrong_cascade={metrics['wrong_cascade']}")
    if "consensus_reached" in metrics:
        print(f"consensus_reached={metrics['consensus_reached']}")
    if "edge_disagreement_rate" in metrics:
        print(f"edge_disagreement_rate={metrics['edge_disagreement_rate']}")
    print(f"output_dir={result.output_dir}")

    return 0
```

Keep the existing clean config-load error handling. The final structure should catch config-loading errors separately from experiment-run errors so replay write failures are not labeled as config-read failures.

- [ ] **Step 4: Add example config**

Create `examples/network_herding.json`:

```json
{
  "experiment_name": "network_herding",
  "seed": 42,
  "num_agents": 30,
  "initial_opinion": {
    "type": "bernoulli",
    "probability_a": 0.5
  },
  "topology": {
    "type": "small_world",
    "degree": 4,
    "rewiring_probability": 0.1
  },
  "scheduler": {
    "type": "synchronous_rounds",
    "rounds": 12
  },
  "observation_policy": {
    "type": "neighbor_actions"
  },
  "update_policy": {
    "type": "threshold",
    "adoption_threshold": 0.6
  },
  "output_dir": "runs/network_herding"
}
```

- [ ] **Step 5: Update README**

Modify `README.md` so the Run section contains both commands:

```markdown
## Run

Sequential information cascade:

```bash
python -m society_simulation run examples/sequential_cascade.json
```

Network herding:

```bash
python -m society_simulation run examples/network_herding.json
```

Artifacts are written to `output_dir` in the config. Sequential cascade runs write:

- `config.json`
- `steps.jsonl`
- `metrics.json`
- `summary.txt`

Network herding runs write:

- `config.json`
- `graph.json`
- `steps.jsonl`
- `timeseries.jsonl`
- `metrics.json`
- `summary.txt`
```

- [ ] **Step 6: Run CLI tests**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 7: Run both examples**

Run:

```bash
python -m society_simulation run examples/sequential_cascade.json
python -m society_simulation run examples/network_herding.json
```

Expected: both commands exit 0. The second output includes `experiment=network_herding`, `consensus_reached=`, `edge_disagreement_rate=`, and `output_dir=runs/network_herding`.

- [ ] **Step 8: Commit**

```bash
git add src/society_simulation/cli.py examples/network_herding.json README.md tests/test_cli.py
git commit -m "feat: add network herding CLI example"
```

---

### Task 9: Final Verification and Cleanup

**Files:**
- Inspect all created and modified files.
- Modify only if verification reveals a concrete problem.

- [ ] **Step 1: Run full test suite**

Run:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run both package commands**

Run:

```bash
python -m society_simulation run examples/sequential_cascade.json
python -m society_simulation run examples/network_herding.json
```

Expected: both commands exit 0.

- [ ] **Step 3: Inspect network replay artifacts**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

run_dir = Path("runs/network_herding")
required = [
    "config.json",
    "graph.json",
    "steps.jsonl",
    "timeseries.jsonl",
    "metrics.json",
    "summary.txt",
]
for name in required:
    assert (run_dir / name).exists(), name
metrics = json.loads((run_dir / "metrics.json").read_text())
steps = (run_dir / "steps.jsonl").read_text().splitlines()
timeseries = (run_dir / "timeseries.jsonl").read_text().splitlines()
assert "final_action_counts" in metrics
assert "edge_disagreement_rate" in metrics
assert "consensus_reached" in metrics
assert len(steps) == 30 * 12
assert len(timeseries) == 13
print("network replay ok")
PY
```

Expected: prints `network replay ok`.

- [ ] **Step 4: Verify deterministic replay for the network example**

Run:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from society_simulation.config import NetworkHerdingConfig, load_config
from society_simulation.runner import run_experiment

base = load_config("examples/network_herding.json")
assert isinstance(base, NetworkHerdingConfig)

def run_once(root: Path, name: str) -> dict[str, str]:
    config = NetworkHerdingConfig.from_dict({**base.to_dict(), "output_dir": str(root / name)})
    result = run_experiment(config)
    hashes = {}
    for filename in ("graph.json", "steps.jsonl", "timeseries.jsonl", "metrics.json"):
        data = (result.output_dir / filename).read_bytes()
        hashes[filename] = hashlib.sha256(data).hexdigest()
    return hashes

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    assert run_once(root, "first") == run_once(root, "second")
print("network deterministic ok")
PY
```

Expected: prints `network deterministic ok`.

- [ ] **Step 5: Check git status**

Run:

```bash
git status --short
```

Expected: clean, or only ignored generated `runs/` artifacts absent from status.

- [ ] **Step 6: Commit any verification fixes**

If verification reveals a tracked-file fix, run:

```bash
git add README.md examples src tests .gitignore
git commit -m "chore: finalize network herding runner"
```

If no tracked-file fix is needed, do not create an empty commit.

---

## Final Review Requirement

After Task 9 passes:

1. Run a final code review over the full v1 branch against:
   - `docs/superpowers/specs/2026-06-17-network-herding-v1-design.md`
   - this implementation plan
2. Fix any Critical or Important findings.
3. Re-run full tests.
4. Use `superpowers:finishing-a-development-branch`.
