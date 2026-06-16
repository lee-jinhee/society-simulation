import pytest

from society_simulation.models import AgentState
from society_simulation.scheduling import PreviousActionsObservation, SequentialScheduler


def test_sequential_scheduler_emits_agents_in_order() -> None:
    scheduler = SequentialScheduler(num_agents=4)

    assert list(scheduler) == [0, 1, 2, 3]


def test_previous_actions_observation_exposes_only_public_actions() -> None:
    states = [
        AgentState(
            agent_id=0,
            private_signal="A",
            belief_probability=0.2,
            confidence=0.6,
            action="B",
            step_index=0,
            observed_actions=(),
        ),
        AgentState(
            agent_id=1,
            private_signal="A",
            belief_probability=0.8,
            confidence=0.6,
            action="A",
            step_index=1,
            observed_actions=("B",),
        ),
    ]

    observation = PreviousActionsObservation().build(
        agent_id=2,
        private_signal="A",
        prior_states=states,
    )

    assert observation.agent_id == 2
    assert observation.private_signal == "A"
    assert observation.observed_actions == ("B", "A")
    assert not hasattr(observation, "observed_private_signals")


def test_observed_actions_cannot_be_mutated() -> None:
    observation = PreviousActionsObservation().build(
        agent_id=0,
        private_signal="A",
        prior_states=[],
    )

    with pytest.raises(AttributeError):
        observation.observed_actions.append("B")  # type: ignore[attr-defined]
