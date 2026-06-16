from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Action = Literal["A", "B"]


@dataclass(frozen=True)
class AgentProfile:
    agent_id: int
    prior_probability: float
    susceptibility: float = 1.0
    label: str | None = None


@dataclass(frozen=True)
class Observation:
    agent_id: int
    private_signal: Action
    observed_actions: list[Action]


@dataclass(frozen=True)
class Decision:
    belief_probability: float
    confidence: float
    action: Action


@dataclass(frozen=True)
class AgentState:
    agent_id: int
    private_signal: Action
    belief_probability: float
    confidence: float
    action: Action
    step_index: int
    observed_actions: list[Action]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
