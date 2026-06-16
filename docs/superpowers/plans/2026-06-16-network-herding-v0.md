# Network Herding v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable sequential information cascade experiment runner with deterministic replay, rule-based policies, metrics, and tests.

**Architecture:** Implement a small Python package under `src/society_simulation`. The run loop owns scheduling, observation construction, policy decisions, state transitions, replay writing, and metrics; agents do not act freely. v0 uses no LLM calls and leaves future graph-local and LLM policies as extension points.

**Tech Stack:** Python 3.11+, standard library, `pytest`; no runtime third-party dependencies for v0.

---

## File Structure

- Create `pyproject.toml`: package metadata, pytest config, console script.
- Modify `README.md`: basic setup and first run instructions.
- Create `examples/sequential_cascade.json`: default runnable experiment config.
- Create `src/society_simulation/__init__.py`: package exports and version.
- Create `src/society_simulation/__main__.py`: `python -m society_simulation` entrypoint.
- Create `src/society_simulation/models.py`: shared dataclasses and action/state type aliases.
- Create `src/society_simulation/config.py`: JSON config loading and validation.
- Create `src/society_simulation/signals.py`: seeded true-state and private-signal generation.
- Create `src/society_simulation/scheduling.py`: sequential scheduler and previous-actions observation policy.
- Create `src/society_simulation/policies.py`: Bayesian and heuristic update policies.
- Create `src/society_simulation/metrics.py`: v0 cascade metrics.
- Create `src/society_simulation/replay.py`: replay artifact writer.
- Create `src/society_simulation/runner.py`: scenario run loop.
- Create `src/society_simulation/cli.py`: command-line interface.
- Create `tests/`: focused pytest files matching the components above.

Keep files small and boring. The interesting thing is the experiment semantics, not framework cleverness.

---

### Task 1: Project Scaffold and Config Validation

**Files:**
- Create: `pyproject.toml`
- Create: `src/society_simulation/__init__.py`
- Create: `src/society_simulation/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from society_simulation.config import ExperimentConfig, load_config


def test_config_accepts_valid_values(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_name="sequential_information_cascade",
        seed=42,
        num_agents=5,
        true_state="A",
        signal_accuracy=0.7,
        prior_probability=0.5,
        scheduler="sequential",
        observation_policy="previous_actions",
        update_policy="bayesian_cascade",
        output_dir=str(tmp_path / "run"),
    )

    config.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_agents", 0, "num_agents must be positive"),
        ("signal_accuracy", 0.49, "signal_accuracy must be between 0.5 and 1.0"),
        ("signal_accuracy", 1.01, "signal_accuracy must be between 0.5 and 1.0"),
        ("prior_probability", 0.0, "prior_probability must be greater than 0 and less than 1"),
        ("prior_probability", 1.0, "prior_probability must be greater than 0 and less than 1"),
        ("experiment_name", "unknown", "unsupported experiment_name"),
        ("scheduler", "random", "unsupported scheduler"),
        ("observation_policy", "global_truth", "unsupported observation_policy"),
        ("update_policy", "llm", "unsupported update_policy"),
        ("true_state", "C", "true_state must be A, B, or null"),
    ],
)
def test_config_rejects_invalid_values(tmp_path: Path, field: str, value: object, message: str) -> None:
    kwargs = {
        "experiment_name": "sequential_information_cascade",
        "seed": 42,
        "num_agents": 5,
        "true_state": "A",
        "signal_accuracy": 0.7,
        "prior_probability": 0.5,
        "scheduler": "sequential",
        "observation_policy": "previous_actions",
        "update_policy": "bayesian_cascade",
        "output_dir": str(tmp_path / "run"),
    }
    kwargs[field] = value
    config = ExperimentConfig(**kwargs)

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_load_config_from_json(tmp_path: Path) -> None:
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

    assert config.seed == 7
    assert config.true_state is None
    assert config.update_policy == "simple_heuristic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`

Expected: FAIL during import with `ModuleNotFoundError: No module named 'society_simulation'`.

- [ ] **Step 3: Add package metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "society-simulation"
version = "0.1.0"
description = "Network-first society simulation experiments"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
society-sim = "society_simulation.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Create `src/society_simulation/__init__.py`:

```python
"""Network-first society simulation experiments."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Implement config loading and validation**

Create `src/society_simulation/config.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

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


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    config = ExperimentConfig(**data)
    config.validate()
    return config
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/society_simulation/__init__.py src/society_simulation/config.py tests/test_config.py
git commit -m "feat: add experiment config validation"
```

---

### Task 2: Core Models, Signal Model, Scheduler, and Observation Policy

**Files:**
- Create: `src/society_simulation/models.py`
- Create: `src/society_simulation/signals.py`
- Create: `src/society_simulation/scheduling.py`
- Create: `tests/test_signals.py`
- Create: `tests/test_scheduling.py`

- [ ] **Step 1: Write failing signal tests**

Create `tests/test_signals.py`:

```python
import random

from society_simulation.signals import BinarySignalModel


def test_signal_generation_is_deterministic_for_seed() -> None:
    first = BinarySignalModel(signal_accuracy=0.7, rng=random.Random(123))
    second = BinarySignalModel(signal_accuracy=0.7, rng=random.Random(123))

    assert first.sample_true_state() == second.sample_true_state()
    assert first.generate_private_signals("A", 10) == second.generate_private_signals("A", 10)


def test_signal_accuracy_is_close_in_large_sample() -> None:
    model = BinarySignalModel(signal_accuracy=0.75, rng=random.Random(4))

    signals = model.generate_private_signals("A", 2000)
    match_rate = signals.count("A") / len(signals)

    assert 0.72 <= match_rate <= 0.78
```

Create `tests/test_scheduling.py`:

```python
from society_simulation.models import AgentState
from society_simulation.scheduling import PreviousActionsObservation, SequentialScheduler


def test_sequential_scheduler_emits_agents_in_order() -> None:
    scheduler = SequentialScheduler(num_agents=4)

    assert list(scheduler) == [0, 1, 2, 3]


def test_previous_actions_observation_exposes_only_public_actions() -> None:
    states = [
        AgentState(
            agent_id=0,
            private_signal="B",
            belief_probability=0.2,
            confidence=0.6,
            action="B",
            step_index=0,
            observed_actions=[],
        ),
        AgentState(
            agent_id=1,
            private_signal="A",
            belief_probability=0.8,
            confidence=0.6,
            action="A",
            step_index=1,
            observed_actions=["B"],
        ),
    ]

    observation = PreviousActionsObservation().build(
        agent_id=2,
        private_signal="A",
        prior_states=states,
    )

    assert observation.agent_id == 2
    assert observation.private_signal == "A"
    assert observation.observed_actions == ["B", "A"]
    assert not hasattr(observation, "observed_private_signals")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_signals.py tests/test_scheduling.py -v`

Expected: FAIL during import because `models.py`, `signals.py`, and `scheduling.py` do not exist.

- [ ] **Step 3: Implement core models**

Create `src/society_simulation/models.py`:

```python
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
```

- [ ] **Step 4: Implement signal generation**

Create `src/society_simulation/signals.py`:

```python
from __future__ import annotations

import random

from society_simulation.models import Action


class BinarySignalModel:
    def __init__(self, signal_accuracy: float, rng: random.Random) -> None:
        if not 0.5 <= signal_accuracy <= 1.0:
            raise ValueError("signal_accuracy must be between 0.5 and 1.0")
        self.signal_accuracy = signal_accuracy
        self.rng = rng

    def sample_true_state(self) -> Action:
        return "A" if self.rng.random() < 0.5 else "B"

    def sample_private_signal(self, true_state: Action) -> Action:
        if true_state not in ("A", "B"):
            raise ValueError("true_state must be A or B")
        if self.rng.random() < self.signal_accuracy:
            return true_state
        return "B" if true_state == "A" else "A"

    def generate_private_signals(self, true_state: Action, num_agents: int) -> list[Action]:
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        return [self.sample_private_signal(true_state) for _ in range(num_agents)]
```

- [ ] **Step 5: Implement scheduler and observation policy**

Create `src/society_simulation/scheduling.py`:

```python
from __future__ import annotations

from collections.abc import Iterator

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
        prior_states: list[AgentState],
    ) -> Observation:
        return Observation(
            agent_id=agent_id,
            private_signal=private_signal,
            observed_actions=[state.action for state in prior_states],
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_signals.py tests/test_scheduling.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/society_simulation/models.py src/society_simulation/signals.py src/society_simulation/scheduling.py tests/test_signals.py tests/test_scheduling.py
git commit -m "feat: add cascade state and observation primitives"
```

---

### Task 3: Bayesian and Heuristic Update Policies

**Files:**
- Create: `src/society_simulation/policies.py`
- Create: `tests/test_policies.py`

- [ ] **Step 1: Write failing policy tests**

Create `tests/test_policies.py`:

```python
import pytest

from society_simulation.models import Observation
from society_simulation.policies import BayesianCascadePolicy, SimpleHeuristicPolicy


def test_bayesian_policy_follows_private_signal_without_history() -> None:
    policy = BayesianCascadePolicy(signal_accuracy=0.7, prior_probability=0.5)

    decision_a = policy.decide(Observation(agent_id=0, private_signal="A", observed_actions=[]))
    decision_b = policy.decide(Observation(agent_id=1, private_signal="B", observed_actions=[]))

    assert decision_a.action == "A"
    assert decision_a.belief_probability == pytest.approx(0.7)
    assert decision_a.confidence == pytest.approx(0.4)
    assert decision_b.action == "B"
    assert decision_b.belief_probability == pytest.approx(0.3)


def test_bayesian_policy_uses_public_action_likelihood_not_hidden_signals() -> None:
    policy = BayesianCascadePolicy(signal_accuracy=0.7, prior_probability=0.5)

    decision = policy.decide(
        Observation(agent_id=2, private_signal="B", observed_actions=["A", "A"])
    )

    assert decision.action == "A"
    assert decision.belief_probability == pytest.approx(0.5)
    assert decision.confidence == pytest.approx(0.0)


def test_bayesian_policy_can_follow_private_signal_after_split_history() -> None:
    policy = BayesianCascadePolicy(signal_accuracy=0.7, prior_probability=0.5)

    decision = policy.decide(
        Observation(agent_id=2, private_signal="A", observed_actions=["A", "B"])
    )

    assert decision.action == "A"
    assert decision.belief_probability > 0.5


def test_simple_heuristic_combines_private_signal_and_majority_actions() -> None:
    policy = SimpleHeuristicPolicy()

    decision = policy.decide(
        Observation(agent_id=3, private_signal="B", observed_actions=["A", "A", "A"])
    )

    assert decision.action == "A"
    assert decision.belief_probability > 0.5
    assert 0.0 <= decision.confidence <= 1.0


def test_simple_heuristic_breaks_ties_toward_private_signal() -> None:
    policy = SimpleHeuristicPolicy()

    decision = policy.decide(
        Observation(agent_id=2, private_signal="B", observed_actions=["A"])
    )

    assert decision.action == "B"
    assert decision.belief_probability == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_policies.py -v`

Expected: FAIL during import because `policies.py` does not exist.

- [ ] **Step 3: Implement policies**

Create `src/society_simulation/policies.py`:

```python
from __future__ import annotations

from functools import lru_cache

from society_simulation.models import Action, Decision, Observation


def confidence_from_belief(belief_probability: float) -> float:
    return abs(belief_probability - 0.5) * 2.0


def _signal_probability(signal: Action, true_state: Action, signal_accuracy: float) -> float:
    if signal == true_state:
        return signal_accuracy
    return 1.0 - signal_accuracy


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


class BayesianCascadePolicy:
    name = "bayesian_cascade"

    def __init__(self, signal_accuracy: float, prior_probability: float) -> None:
        if not 0.5 <= signal_accuracy <= 1.0:
            raise ValueError("signal_accuracy must be between 0.5 and 1.0")
        if not 0.0 < prior_probability < 1.0:
            raise ValueError("prior_probability must be greater than 0 and less than 1")
        self.signal_accuracy = signal_accuracy
        self.prior_probability = prior_probability

    def decide(self, observation: Observation) -> Decision:
        belief = self._posterior_probability(
            tuple(observation.observed_actions),
            observation.private_signal,
        )
        action: Action = "A" if belief >= 0.5 else "B"
        return Decision(
            belief_probability=belief,
            confidence=confidence_from_belief(belief),
            action=action,
        )

    def _posterior_probability(self, observed_actions: tuple[Action, ...], private_signal: Action) -> float:
        likelihood_history_a = self._history_likelihood(observed_actions, "A")
        likelihood_history_b = self._history_likelihood(observed_actions, "B")
        likelihood_signal_a = _signal_probability(private_signal, "A", self.signal_accuracy)
        likelihood_signal_b = _signal_probability(private_signal, "B", self.signal_accuracy)

        numerator = self.prior_probability * likelihood_history_a * likelihood_signal_a
        denominator = numerator + (
            (1.0 - self.prior_probability) * likelihood_history_b * likelihood_signal_b
        )
        if denominator == 0.0:
            return 0.5
        return numerator / denominator

    @lru_cache(maxsize=None)
    def _history_likelihood(self, observed_actions: tuple[Action, ...], true_state: Action) -> float:
        if not observed_actions:
            return 1.0

        prefix = observed_actions[:-1]
        final_action = observed_actions[-1]
        prefix_likelihood = self._history_likelihood(prefix, true_state)

        matching_signal_probability = 0.0
        for private_signal in ("A", "B"):
            predicted = self._action_for_public_history(prefix, private_signal)
            if predicted == final_action:
                matching_signal_probability += _signal_probability(
                    private_signal,
                    true_state,
                    self.signal_accuracy,
                )

        return prefix_likelihood * matching_signal_probability

    def _action_for_public_history(self, observed_actions: tuple[Action, ...], private_signal: Action) -> Action:
        belief = self._posterior_probability(observed_actions, private_signal)
        return "A" if belief >= 0.5 else "B"


class SimpleHeuristicPolicy:
    name = "simple_heuristic"

    def decide(self, observation: Observation) -> Decision:
        score = 1 if observation.private_signal == "A" else -1
        score += observation.observed_actions.count("A")
        score -= observation.observed_actions.count("B")

        if score > 0:
            action: Action = "A"
        elif score < 0:
            action = "B"
        else:
            action = observation.private_signal

        scale = len(observation.observed_actions) + 1
        belief = _clamp_probability(0.5 + 0.5 * (score / scale))
        return Decision(
            belief_probability=belief,
            confidence=confidence_from_belief(belief),
            action=action,
        )


def build_update_policy(name: str, signal_accuracy: float, prior_probability: float) -> BayesianCascadePolicy | SimpleHeuristicPolicy:
    if name == "bayesian_cascade":
        return BayesianCascadePolicy(signal_accuracy=signal_accuracy, prior_probability=prior_probability)
    if name == "simple_heuristic":
        return SimpleHeuristicPolicy()
    raise ValueError("unsupported update_policy")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_policies.py -v`

Expected: PASS.

- [ ] **Step 5: Run existing component tests**

Run: `pytest tests/test_config.py tests/test_signals.py tests/test_scheduling.py tests/test_policies.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/society_simulation/policies.py tests/test_policies.py
git commit -m "feat: add cascade update policies"
```

---

### Task 4: Metrics

**Files:**
- Create: `src/society_simulation/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing metrics tests**

Create `tests/test_metrics.py`:

```python
from society_simulation.metrics import compute_metrics
from society_simulation.models import AgentState


def make_state(
    agent_id: int,
    private_signal: str,
    action: str,
    belief_probability: float = 0.8,
) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        private_signal=private_signal,  # type: ignore[arg-type]
        belief_probability=belief_probability,
        confidence=abs(belief_probability - 0.5) * 2,
        action=action,  # type: ignore[arg-type]
        step_index=agent_id,
        observed_actions=[],
    )


def test_metrics_identify_correct_cascade() -> None:
    states = [
        make_state(0, "B", "B", 0.2),
        make_state(1, "A", "A", 0.8),
        make_state(2, "B", "A", 0.5),
        make_state(3, "B", "A", 0.5),
    ]

    metrics = compute_metrics(states, true_state="A")

    assert metrics["final_accuracy"] == 0.75
    assert metrics["correct_cascade"] is True
    assert metrics["wrong_cascade"] is False
    assert metrics["cascade_start_step"] == 1
    assert metrics["private_signal_ignored_count"] == 2
    assert metrics["action_counts"] == {"A": 3, "B": 1}


def test_metrics_identify_wrong_cascade() -> None:
    states = [
        make_state(0, "A", "A", 0.8),
        make_state(1, "B", "B", 0.2),
        make_state(2, "A", "B", 0.5),
        make_state(3, "A", "B", 0.5),
    ]

    metrics = compute_metrics(states, true_state="A")

    assert metrics["correct_cascade"] is False
    assert metrics["wrong_cascade"] is True
    assert metrics["cascade_start_step"] == 1


def test_metrics_report_no_cascade_when_suffix_has_no_ignored_signal() -> None:
    states = [
        make_state(0, "A", "A", 0.8),
        make_state(1, "B", "B", 0.2),
        make_state(2, "A", "A", 0.8),
        make_state(3, "A", "A", 0.8),
    ]

    metrics = compute_metrics(states, true_state="A")

    assert metrics["correct_cascade"] is False
    assert metrics["wrong_cascade"] is False
    assert metrics["cascade_start_step"] is None


def test_belief_summary_is_computed() -> None:
    states = [
        make_state(0, "A", "A", 0.7),
        make_state(1, "A", "A", 0.9),
    ]

    metrics = compute_metrics(states, true_state="A")

    assert metrics["belief_summary"] == {
        "min": 0.7,
        "max": 0.9,
        "mean": 0.8,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`

Expected: FAIL during import because `metrics.py` does not exist.

- [ ] **Step 3: Implement metrics**

Create `src/society_simulation/metrics.py`:

```python
from __future__ import annotations

from statistics import mean
from typing import Any

from society_simulation.models import Action, AgentState


def compute_metrics(states: list[AgentState], true_state: Action) -> dict[str, Any]:
    if not states:
        raise ValueError("states must not be empty")

    action_counts = {
        "A": sum(1 for state in states if state.action == "A"),
        "B": sum(1 for state in states if state.action == "B"),
    }
    final_accuracy = sum(1 for state in states if state.action == true_state) / len(states)
    private_signal_ignored_count = sum(
        1 for state in states if state.action != state.private_signal
    )
    cascade_start_step, cascade_action = _detect_operational_cascade(states)
    correct_cascade = cascade_action == true_state if cascade_action is not None else False
    wrong_cascade = cascade_action != true_state if cascade_action is not None else False
    beliefs = [state.belief_probability for state in states]

    return {
        "final_accuracy": final_accuracy,
        "correct_cascade": correct_cascade,
        "wrong_cascade": wrong_cascade,
        "cascade_start_step": cascade_start_step,
        "private_signal_ignored_count": private_signal_ignored_count,
        "action_counts": action_counts,
        "belief_summary": {
            "min": min(beliefs),
            "max": max(beliefs),
            "mean": mean(beliefs),
        },
    }


def _detect_operational_cascade(states: list[AgentState]) -> tuple[int | None, Action | None]:
    for start in range(0, len(states) - 1):
        suffix = states[start:]
        suffix_actions = {state.action for state in suffix}
        has_ignored_private_signal = any(state.action != state.private_signal for state in suffix)
        if len(suffix_actions) == 1 and has_ignored_private_signal:
            return states[start].step_index, suffix[0].action
    return None, None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/metrics.py tests/test_metrics.py
git commit -m "feat: add cascade metrics"
```

---

### Task 5: Replay Writer

**Files:**
- Create: `src/society_simulation/replay.py`
- Create: `tests/test_replay.py`

- [ ] **Step 1: Write failing replay tests**

Create `tests/test_replay.py`:

```python
import json
from pathlib import Path

from society_simulation.config import ExperimentConfig
from society_simulation.models import AgentState
from society_simulation.replay import ReplayWriter


def test_replay_writer_outputs_config_steps_metrics_and_summary(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_name="sequential_information_cascade",
        seed=11,
        num_agents=1,
        true_state="A",
        signal_accuracy=0.7,
        prior_probability=0.5,
        scheduler="sequential",
        observation_policy="previous_actions",
        update_policy="simple_heuristic",
        output_dir=str(tmp_path / "run"),
    )
    state = AgentState(
        agent_id=0,
        private_signal="A",
        belief_probability=1.0,
        confidence=1.0,
        action="A",
        step_index=0,
        observed_actions=[],
    )
    metrics = {
        "final_accuracy": 1.0,
        "correct_cascade": False,
        "wrong_cascade": False,
        "cascade_start_step": None,
        "private_signal_ignored_count": 0,
        "action_counts": {"A": 1, "B": 0},
        "belief_summary": {"min": 1.0, "max": 1.0, "mean": 1.0},
    }

    output_path = ReplayWriter(config).write(
        true_state="A",
        states=[state],
        metrics=metrics,
    )

    assert output_path == tmp_path / "run"
    assert json.loads((output_path / "config.json").read_text(encoding="utf-8"))["seed"] == 11
    assert json.loads((output_path / "metrics.json").read_text(encoding="utf-8"))["final_accuracy"] == 1.0
    step_lines = (output_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(step_lines) == 1
    step = json.loads(step_lines[0])
    assert step["agent_id"] == 0
    assert step["true_state"] == "A"
    assert step["random_seed"] == 11
    assert step["update_policy"] == "simple_heuristic"
    assert "experiment_name=sequential_information_cascade" in (output_path / "summary.txt").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_replay.py -v`

Expected: FAIL during import because `replay.py` does not exist.

- [ ] **Step 3: Implement replay writer**

Create `src/society_simulation/replay.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from society_simulation.config import ExperimentConfig
from society_simulation.models import Action, AgentState


class ReplayWriter:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def write(
        self,
        true_state: Action,
        states: list[AgentState],
        metrics: dict[str, Any],
    ) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._write_steps(output_dir / "steps.jsonl", true_state, states)
        self._write_summary(output_dir / "summary.txt", true_state, metrics)
        return output_dir

    def _write_steps(self, path: Path, true_state: Action, states: list[AgentState]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for state in states:
                payload = state.to_dict()
                payload["true_state"] = true_state
                payload["update_policy"] = self.config.update_policy
                payload["random_seed"] = self.config.seed
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _write_summary(self, path: Path, true_state: Action, metrics: dict[str, Any]) -> None:
        lines = [
            f"experiment_name={self.config.experiment_name}",
            f"update_policy={self.config.update_policy}",
            f"seed={self.config.seed}",
            f"true_state={true_state}",
            f"action_counts={metrics['action_counts']}",
            f"correct_cascade={metrics['correct_cascade']}",
            f"wrong_cascade={metrics['wrong_cascade']}",
            f"cascade_start_step={metrics['cascade_start_step']}",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_replay.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/replay.py tests/test_replay.py
git commit -m "feat: add replay writer"
```

---

### Task 6: Scenario Runner

**Files:**
- Create: `src/society_simulation/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing runner tests**

Create `tests/test_runner.py`:

```python
import json
from pathlib import Path

from society_simulation.config import ExperimentConfig
from society_simulation.runner import run_experiment


def make_config(tmp_path: Path, seed: int = 123) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="sequential_information_cascade",
        seed=seed,
        num_agents=6,
        true_state="A",
        signal_accuracy=0.7,
        prior_probability=0.5,
        scheduler="sequential",
        observation_policy="previous_actions",
        update_policy="bayesian_cascade",
        output_dir=str(tmp_path / f"run-{seed}"),
    )


def test_run_experiment_writes_one_step_per_agent(tmp_path: Path) -> None:
    result = run_experiment(make_config(tmp_path))

    assert result.true_state == "A"
    assert result.output_dir == tmp_path / "run-123"
    assert len(result.states) == 6
    assert (result.output_dir / "steps.jsonl").exists()
    assert len((result.output_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()) == 6


def test_run_experiment_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    first_config = make_config(tmp_path, seed=77)
    second_config = ExperimentConfig(
        **{
            **first_config.to_dict(),
            "output_dir": str(tmp_path / "run-77-copy"),
        }
    )

    first = run_experiment(first_config)
    second = run_experiment(second_config)

    first_steps = (first.output_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    second_steps = (second.output_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    normalized_first = [
        {key: value for key, value in json.loads(line).items()}
        for line in first_steps
    ]
    normalized_second = [
        {key: value for key, value in json.loads(line).items()}
        for line in second_steps
    ]
    assert normalized_first == normalized_second
    assert first.metrics == second.metrics
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runner.py -v`

Expected: FAIL during import because `runner.py` does not exist.

- [ ] **Step 3: Implement runner**

Create `src/society_simulation/runner.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

from society_simulation.config import ExperimentConfig
from society_simulation.metrics import compute_metrics
from society_simulation.models import Action, AgentProfile, AgentState
from society_simulation.policies import build_update_policy
from society_simulation.replay import ReplayWriter
from society_simulation.scheduling import PreviousActionsObservation, SequentialScheduler
from society_simulation.signals import BinarySignalModel


@dataclass(frozen=True)
class RunResult:
    true_state: Action
    states: list[AgentState]
    metrics: dict[str, Any]
    output_dir: Path


def run_experiment(config: ExperimentConfig) -> RunResult:
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
        states=states,
        metrics=metrics,
        output_dir=output_dir,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runner.py -v`

Expected: PASS.

- [ ] **Step 5: Run all unit tests so far**

Run: `pytest -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/society_simulation/runner.py tests/test_runner.py
git commit -m "feat: add sequential cascade runner"
```

---

### Task 7: CLI, Example Config, and README

**Files:**
- Create: `src/society_simulation/cli.py`
- Create: `src/society_simulation/__main__.py`
- Create: `examples/sequential_cascade.json`
- Modify: `README.md`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI test**

Create `tests/test_cli.py`:

```python
import json
from pathlib import Path

from society_simulation.cli import main


def test_cli_runs_config_and_prints_summary(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "run"
    config_path.write_text(
        json.dumps(
            {
                "experiment_name": "sequential_information_cascade",
                "seed": 5,
                "num_agents": 4,
                "true_state": "A",
                "signal_accuracy": 0.7,
                "prior_probability": 0.5,
                "scheduler": "sequential",
                "observation_policy": "previous_actions",
                "update_policy": "simple_heuristic",
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["run", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "experiment=sequential_information_cascade" in captured.out
    assert "true_state=A" in captured.out
    assert "output_dir=" in captured.out
    assert (output_dir / "metrics.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`

Expected: FAIL during import because `cli.py` does not exist.

- [ ] **Step 3: Implement CLI entrypoint**

Create `src/society_simulation/cli.py`:

```python
from __future__ import annotations

import argparse
from collections.abc import Sequence

from society_simulation.config import load_config
from society_simulation.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="society-sim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run an experiment config")
    run_parser.add_argument("config", help="path to experiment JSON config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = load_config(args.config)
        result = run_experiment(config)
        print(f"experiment={config.experiment_name}")
        print(f"true_state={result.true_state}")
        print(f"action_counts={result.metrics['action_counts']}")
        print(f"correct_cascade={result.metrics['correct_cascade']}")
        print(f"wrong_cascade={result.metrics['wrong_cascade']}")
        print(f"output_dir={result.output_dir}")
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2
```

Create `src/society_simulation/__main__.py`:

```python
from society_simulation.cli import main

raise SystemExit(main())
```

- [ ] **Step 4: Add example config**

Create `examples/sequential_cascade.json`:

```json
{
  "experiment_name": "sequential_information_cascade",
  "seed": 42,
  "num_agents": 10,
  "true_state": null,
  "signal_accuracy": 0.7,
  "prior_probability": 0.5,
  "scheduler": "sequential",
  "observation_policy": "previous_actions",
  "update_policy": "bayesian_cascade",
  "output_dir": "runs/sequential_cascade"
}
```

- [ ] **Step 5: Update README**

Replace `README.md` with:

````markdown
# society-simulation

Network-first society simulation experiments.

The first runnable experiment is a sequential information cascade. It uses no LLM calls. The goal is to verify the experiment engine, replay logs, and herding metrics before adding graph-local or LLM-based agents.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run the first experiment

```bash
python -m society_simulation run examples/sequential_cascade.json
```

The run writes:

- `config.json`
- `steps.jsonl`
- `metrics.json`
- `summary.txt`

under the configured `output_dir`.

## Test

```bash
pytest -v
```
````

- [ ] **Step 6: Run CLI test**

Run: `pytest tests/test_cli.py -v`

Expected: PASS.

- [ ] **Step 7: Run example command**

Run: `python -m society_simulation run examples/sequential_cascade.json`

Expected output includes:

```text
experiment=sequential_information_cascade
true_state=
action_counts=
output_dir=runs/sequential_cascade
```

- [ ] **Step 8: Commit**

```bash
git add src/society_simulation/cli.py src/society_simulation/__main__.py examples/sequential_cascade.json README.md tests/test_cli.py
git commit -m "feat: add cascade CLI"
```

---

### Task 8: Final Verification and Cleanup

**Files:**
- Inspect all created files.
- Modify only if verification reveals a concrete problem.

- [ ] **Step 1: Run full test suite**

Run: `pytest -v`

Expected: all tests pass.

- [ ] **Step 2: Run package command with example config**

Run: `python -m society_simulation run examples/sequential_cascade.json`

Expected: command exits 0 and writes `runs/sequential_cascade/config.json`, `runs/sequential_cascade/steps.jsonl`, `runs/sequential_cascade/metrics.json`, and `runs/sequential_cascade/summary.txt`.

- [ ] **Step 3: Inspect replay artifacts**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

run_dir = Path("runs/sequential_cascade")
metrics = json.loads((run_dir / "metrics.json").read_text())
steps = (run_dir / "steps.jsonl").read_text().splitlines()
assert "final_accuracy" in metrics
assert "correct_cascade" in metrics
assert "wrong_cascade" in metrics
assert len(steps) == 10
print("replay ok")
PY
```

Expected: prints `replay ok`.

- [ ] **Step 4: Check git status**

Run: `git status --short`

Expected: only intended uncommitted run artifacts may appear under `runs/`. If `runs/` appears, add it to `.gitignore`.

- [ ] **Step 5: Ignore generated runs if needed**

If `git status --short` shows `?? runs/`, append this exact line to `.gitignore`:

```gitignore
runs/
```

Then commit it:

```bash
git add .gitignore
git commit -m "chore: ignore generated run artifacts"
```

- [ ] **Step 6: Final commit if verification required fixes**

If any verification fixes changed tracked source, tests, docs, or examples, commit them:

```bash
git add README.md pyproject.toml examples src tests .gitignore
git commit -m "chore: finalize cascade runner"
```

- [ ] **Step 7: Final status**

Run: `git status --short`

Expected: clean working tree.

---

## Spec Coverage Self-Review

- Sequential information cascade run: Task 6 and Task 7.
- No LLM calls: all tasks use standard-library deterministic policies only.
- Explicit state, observations, update policy, and metrics: Tasks 2, 3, and 4.
- Replay log: Task 5.
- Config file execution: Task 1, Task 6, and Task 7.
- Deterministic same-seed behavior: Task 2 and Task 6 tests.
- Bayesian and heuristic policies: Task 3.
- Core tests: Tasks 1 through 7.
- Future graph-local and LLM extension points: file boundaries and policy interfaces in Tasks 2, 3, and 6.

The plan intentionally does not implement graph topology, LLM providers, public news, polling, or UI because the approved v0 spec defers them.
