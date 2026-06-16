from __future__ import annotations

from collections.abc import Iterator, Sequence

from society_simulation.models import Action, AgentState, Observation


class SequentialScheduler:
    def __init__(self, num_agents: int) -> None:
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        self.num_agents = num_agents

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.num_agents))


class PreviousActionsObservation:
    def build(
        self,
        agent_id: int,
        private_signal: Action,
        prior_states: Sequence[AgentState],
    ) -> Observation:
        return Observation(
            agent_id=agent_id,
            private_signal=private_signal,
            observed_actions=tuple(state.action for state in prior_states),
        )
