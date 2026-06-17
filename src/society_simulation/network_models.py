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
