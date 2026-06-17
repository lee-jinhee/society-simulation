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
