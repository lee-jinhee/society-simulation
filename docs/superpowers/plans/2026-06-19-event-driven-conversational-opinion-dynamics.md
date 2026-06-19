# Event-Driven Conversational Opinion Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal runnable `event_driven_opinion_dynamics` scenario where persona-based agents encounter staged events, exchange natural-language messages, update private/public stances, and write auditable replay artifacts.

**Architecture:** Add a new scenario beside `network_herding` instead of stretching the binary network model. Keep event-specific models, config, scheduling, policy, replay, metrics, and runner in focused modules, then connect the scenario through the existing `load_config()` and `run_experiment()` dispatch path. Start with a deterministic mock persona policy for tests, then add OpenAI-compatible persona policy support using the existing LLM transport/cost primitives.

**Tech Stack:** Python 3.11 dataclasses, standard-library JSON/CSV/path utilities, existing pytest suite, existing OpenAI-compatible HTTP transport helpers from `society_simulation.llm_policy`.

---

## Scope Check

This plan implements the trace-validity pilot, not a full generative society. It includes:

- stable agent profiles;
- relationships and channels;
- staged external events;
- deterministic exposure delivery;
- mock persona policy;
- OpenAI-compatible persona policy;
- private/public stance state;
- natural-language messages;
- event replay artifacts;
- summary metrics;
- one runnable congestion-pricing experiment config.

It excludes real news ingestion, recommendation feeds, autonomous schedules, large sweeps, polling prediction, and external validity claims.

## File Structure

Create:

- `src/society_simulation/event_models.py`: event-scenario dataclasses and validation helpers.
- `src/society_simulation/event_config.py`: config parsing and validation for `event_driven_opinion_dynamics`.
- `src/society_simulation/event_policy.py`: mock and OpenAI-compatible persona policies plus JSON parsing.
- `src/society_simulation/event_scheduling.py`: external event exposure and previous-message delivery.
- `src/society_simulation/event_metrics.py`: day-level and final opinion metrics.
- `src/society_simulation/event_replay.py`: replay artifact writer.
- `src/society_simulation/event_runner.py`: day-by-day scenario runner.
- `experiments/event_driven_congestion_pricing.json`: first mock-run experiment.
- `tests/test_event_models.py`
- `tests/test_event_config.py`
- `tests/test_event_policy.py`
- `tests/test_event_scheduling.py`
- `tests/test_event_metrics.py`
- `tests/test_event_replay.py`
- `tests/test_event_runner.py`

Modify:

- `src/society_simulation/config.py`: dispatch `event_driven_opinion_dynamics` configs.
- `src/society_simulation/runner.py`: dispatch event configs to `run_event_driven_opinion_dynamics`.
- `src/society_simulation/cli.py`: print event metrics without requiring A/B action counts.
- `tests/test_cli.py`: cover event scenario CLI output.
- `tests/test_sweep_artifacts.py` and `src/society_simulation/sweep_artifacts.py`: allow event metrics in sweep summary without breaking network fields.

---

### Task 1: Event Domain Models

**Files:**
- Create: `src/society_simulation/event_models.py`
- Test: `tests/test_event_models.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/test_event_models.py`:

```python
import pytest

from society_simulation.event_models import (
    EventAgentProfile,
    EventAgentState,
    EventExposure,
    EventMessage,
    EventRelationship,
    OpinionEvent,
    validate_probability,
    validate_stance,
)


def test_validate_stance_accepts_minus_one_to_one() -> None:
    assert validate_stance(-1.0, "private_stance") == -1.0
    assert validate_stance(0, "private_stance") == 0.0
    assert validate_stance(1.0, "private_stance") == 1.0


@pytest.mark.parametrize("value", [-1.1, 1.1, "0.3", True])
def test_validate_stance_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="private_stance must be a number between -1 and 1"):
        validate_stance(value, "private_stance")


@pytest.mark.parametrize("value", [-0.1, 1.1, "0.3", True])
def test_validate_probability_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="confidence must be a number between 0 and 1"):
        validate_probability(value, "confidence")


def test_agent_profile_from_dict_validates_required_fields() -> None:
    profile = EventAgentProfile.from_dict(
        {
            "agent_id": "jisoo",
            "name": "Jisoo Park",
            "age": 37,
            "occupation": "hospital nurse",
            "household_context": "single parent",
            "neighborhood": "east side",
            "core_values": ["fairness", "public health"],
            "material_interests": ["commute time"],
            "political_trust": 0.45,
            "media_habits": ["local_news", "neighborhood_group_chat"],
            "communication_style": "careful",
            "susceptibilities": ["coworker stories"],
            "initial_private_stance": -0.1,
            "initial_public_stance": 0.0,
            "initial_confidence": 0.35,
            "initial_salience": 0.45,
        }
    )

    assert profile.agent_id == "jisoo"
    assert profile.initial_state(day=0).private_stance == -0.1
    assert profile.to_dict()["name"] == "Jisoo Park"


def test_relationship_from_dict_round_trips() -> None:
    relationship = EventRelationship.from_dict(
        {
            "source_agent_id": "jisoo",
            "target_agent_id": "minho",
            "relationship_type": "coworker",
            "trust": 0.72,
            "conversation_frequency": "high",
            "conflict_sensitivity": 0.4,
            "channels": ["hospital_group_chat"],
        }
    )

    assert relationship.trust == 0.72
    assert relationship.channels == ("hospital_group_chat",)
    assert relationship.to_dict()["target_agent_id"] == "minho"


def test_event_from_dict_supports_audience_filter() -> None:
    event = OpinionEvent.from_dict(
        {
            "event_id": "city_announcement",
            "day": 1,
            "title": "City announces downtown congestion charge",
            "source": "City Hall",
            "source_type": "official",
            "content": "The city proposes a weekday downtown congestion charge.",
            "policy_stance": 0.35,
            "credibility": 0.8,
            "emotional_intensity": 0.2,
            "affected_interests": ["commute time", "public health"],
            "audience_filter": {"media_habits_any": ["local_news"]},
        }
    )

    assert event.day == 1
    assert event.audience_filter == {"media_habits_any": ["local_news"]}


def test_exposure_message_and_state_to_dict_are_json_ready() -> None:
    exposure = EventExposure(
        day=2,
        agent_id="jisoo",
        source_type="event",
        source_id="taxi_story",
        channel="news_feed",
        content="A taxi driver says the fee threatens her income.",
    )
    message = EventMessage(
        day=2,
        sender_agent_id="minho",
        channel="neighborhood_group_chat",
        recipient_agent_id=None,
        text="This fee sounds rough for shift workers.",
    )
    state = EventAgentState(
        agent_id="jisoo",
        day=2,
        private_stance=-0.25,
        public_stance=0.0,
        confidence=0.52,
        salience=0.77,
        emotion="worried",
        memory_summary="Jisoo is worried about commute costs.",
        last_private_reasoning="The taxi story felt concrete.",
    )

    assert exposure.to_dict()["source_id"] == "taxi_story"
    assert message.to_dict()["recipient_agent_id"] is None
    assert state.to_dict()["emotion"] == "worried"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_models.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'society_simulation.event_models'`.

- [ ] **Step 3: Implement event models**

Create `src/society_simulation/event_models.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Literal

RelationshipType = Literal["friend", "coworker", "family", "neighbor", "weak_tie"]
ConversationFrequency = Literal["low", "medium", "high"]
SourceType = Literal["official", "news", "personal_story", "viral_clip", "fact_check", "private_message"]
ExposureSourceType = Literal["event", "message"]


def _require_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _require_string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return tuple(value)


def validate_probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValueError(f"{field} must be a number between 0 and 1")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must be a number between 0 and 1")
    return parsed


def validate_stance(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValueError(f"{field} must be a number between -1 and 1")
    parsed = float(value)
    if not -1.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must be a number between -1 and 1")
    return parsed


@dataclass(frozen=True)
class EventAgentProfile:
    agent_id: str
    name: str
    age: int
    occupation: str
    household_context: str
    neighborhood: str
    core_values: tuple[str, ...]
    material_interests: tuple[str, ...]
    political_trust: float
    media_habits: tuple[str, ...]
    communication_style: str
    susceptibilities: tuple[str, ...]
    initial_private_stance: float
    initial_public_stance: float
    initial_confidence: float
    initial_salience: float

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EventAgentProfile:
        data = _require_mapping(data, "agent")
        return cls(
            agent_id=_require_string(data.get("agent_id"), "agent.agent_id"),
            name=_require_string(data.get("name"), "agent.name"),
            age=_require_int(data.get("age"), "agent.age"),
            occupation=_require_string(data.get("occupation"), "agent.occupation"),
            household_context=_require_string(data.get("household_context"), "agent.household_context"),
            neighborhood=_require_string(data.get("neighborhood"), "agent.neighborhood"),
            core_values=_require_string_list(data.get("core_values"), "agent.core_values"),
            material_interests=_require_string_list(data.get("material_interests"), "agent.material_interests"),
            political_trust=validate_probability(data.get("political_trust"), "agent.political_trust"),
            media_habits=_require_string_list(data.get("media_habits"), "agent.media_habits"),
            communication_style=_require_string(data.get("communication_style"), "agent.communication_style"),
            susceptibilities=_require_string_list(data.get("susceptibilities"), "agent.susceptibilities"),
            initial_private_stance=validate_stance(data.get("initial_private_stance"), "agent.initial_private_stance"),
            initial_public_stance=validate_stance(data.get("initial_public_stance"), "agent.initial_public_stance"),
            initial_confidence=validate_probability(data.get("initial_confidence"), "agent.initial_confidence"),
            initial_salience=validate_probability(data.get("initial_salience"), "agent.initial_salience"),
        )

    def initial_state(self, day: int) -> EventAgentState:
        return EventAgentState(
            agent_id=self.agent_id,
            day=day,
            private_stance=self.initial_private_stance,
            public_stance=self.initial_public_stance,
            confidence=self.initial_confidence,
            salience=self.initial_salience,
            emotion="calm",
            memory_summary=f"{self.name} has not yet discussed the policy.",
            last_private_reasoning="Initial profile state.",
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EventRelationship:
    source_agent_id: str
    target_agent_id: str
    relationship_type: str
    trust: float
    conversation_frequency: str
    conflict_sensitivity: float
    channels: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EventRelationship:
        data = _require_mapping(data, "relationship")
        return cls(
            source_agent_id=_require_string(data.get("source_agent_id"), "relationship.source_agent_id"),
            target_agent_id=_require_string(data.get("target_agent_id"), "relationship.target_agent_id"),
            relationship_type=_require_string(data.get("relationship_type"), "relationship.relationship_type"),
            trust=validate_probability(data.get("trust"), "relationship.trust"),
            conversation_frequency=_require_string(data.get("conversation_frequency"), "relationship.conversation_frequency"),
            conflict_sensitivity=validate_probability(data.get("conflict_sensitivity"), "relationship.conflict_sensitivity"),
            channels=_require_string_list(data.get("channels"), "relationship.channels"),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OpinionEvent:
    event_id: str
    day: int
    title: str
    source: str
    source_type: str
    content: str
    policy_stance: float
    credibility: float
    emotional_intensity: float
    affected_interests: tuple[str, ...]
    audience_filter: dict[str, object]

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> OpinionEvent:
        data = _require_mapping(data, "event")
        return cls(
            event_id=_require_string(data.get("event_id"), "event.event_id"),
            day=_require_int(data.get("day"), "event.day"),
            title=_require_string(data.get("title"), "event.title"),
            source=_require_string(data.get("source"), "event.source"),
            source_type=_require_string(data.get("source_type"), "event.source_type"),
            content=_require_string(data.get("content"), "event.content"),
            policy_stance=validate_stance(data.get("policy_stance"), "event.policy_stance"),
            credibility=validate_probability(data.get("credibility"), "event.credibility"),
            emotional_intensity=validate_probability(data.get("emotional_intensity"), "event.emotional_intensity"),
            affected_interests=_require_string_list(data.get("affected_interests"), "event.affected_interests"),
            audience_filter=dict(_require_mapping(data.get("audience_filter", {}), "event.audience_filter")),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EventExposure:
    day: int
    agent_id: str
    source_type: str
    source_id: str
    channel: str
    content: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EventMessage:
    day: int
    sender_agent_id: str
    channel: str
    recipient_agent_id: str | None
    text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EventAgentState:
    agent_id: str
    day: int
    private_stance: float
    public_stance: float
    confidence: float
    salience: float
    emotion: str
    memory_summary: str
    last_private_reasoning: str

    def __post_init__(self) -> None:
        validate_stance(self.private_stance, "private_stance")
        validate_stance(self.public_stance, "public_stance")
        validate_probability(self.confidence, "confidence")
        validate_probability(self.salience, "salience")
        _require_string(self.emotion, "emotion")
        _require_string(self.memory_summary, "memory_summary")
        _require_string(self.last_private_reasoning, "last_private_reasoning")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_models.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/event_models.py tests/test_event_models.py
git commit -m "feat: add event opinion domain models"
```

---

### Task 2: Event Scenario Config

**Files:**
- Create: `src/society_simulation/event_config.py`
- Modify: `src/society_simulation/config.py`
- Test: `tests/test_event_config.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/test_event_config.py`:

```python
import json
from pathlib import Path

import pytest

from society_simulation.config import load_config
from society_simulation.event_config import EventDrivenOpinionConfig


def valid_event_config(tmp_path: Path) -> dict[str, object]:
    return {
        "experiment_name": "event_driven_opinion_dynamics",
        "seed": 11,
        "scenario_name": "congestion_pricing",
        "days": 3,
        "agents": [
            {
                "agent_id": "jisoo",
                "name": "Jisoo Park",
                "age": 37,
                "occupation": "hospital nurse",
                "household_context": "single parent",
                "neighborhood": "east side",
                "core_values": ["fairness", "public health"],
                "material_interests": ["commute time"],
                "political_trust": 0.45,
                "media_habits": ["local_news", "neighborhood_group_chat"],
                "communication_style": "careful",
                "susceptibilities": ["coworker stories"],
                "initial_private_stance": -0.1,
                "initial_public_stance": 0.0,
                "initial_confidence": 0.35,
                "initial_salience": 0.45,
            },
            {
                "agent_id": "minho",
                "name": "Minho Lee",
                "age": 42,
                "occupation": "delivery driver",
                "household_context": "married with two children",
                "neighborhood": "west side",
                "core_values": ["work", "family budget"],
                "material_interests": ["fuel cost", "delivery time"],
                "political_trust": 0.25,
                "media_habits": ["local_news", "neighborhood_group_chat"],
                "communication_style": "direct",
                "susceptibilities": ["small business stories"],
                "initial_private_stance": -0.45,
                "initial_public_stance": -0.25,
                "initial_confidence": 0.55,
                "initial_salience": 0.7,
            },
        ],
        "relationships": [
            {
                "source_agent_id": "jisoo",
                "target_agent_id": "minho",
                "relationship_type": "neighbor",
                "trust": 0.6,
                "conversation_frequency": "medium",
                "conflict_sensitivity": 0.5,
                "channels": ["neighborhood_group_chat"],
            }
        ],
        "events": [
            {
                "event_id": "city_announcement",
                "day": 1,
                "title": "City announces congestion charge",
                "source": "City Hall",
                "source_type": "official",
                "content": "The city proposes a weekday downtown congestion charge.",
                "policy_stance": 0.35,
                "credibility": 0.8,
                "emotional_intensity": 0.2,
                "affected_interests": ["commute time", "public health"],
                "audience_filter": {"media_habits_any": ["local_news"]},
            }
        ],
        "channels": [{"channel_id": "neighborhood_group_chat", "type": "group_chat"}],
        "update_policy": {"type": "mock_persona", "response_style": "balanced"},
        "output_dir": str(tmp_path / "event-run"),
    }


def test_load_event_driven_config(tmp_path: Path) -> None:
    path = tmp_path / "event.json"
    path.write_text(json.dumps(valid_event_config(tmp_path)), encoding="utf-8")

    config = load_config(path)

    assert isinstance(config, EventDrivenOpinionConfig)
    assert config.experiment_name == "event_driven_opinion_dynamics"
    assert config.scenario_name == "congestion_pricing"
    assert len(config.agents) == 2
    assert config.update_policy["type"] == "mock_persona"


def test_event_config_to_dict_round_trips(tmp_path: Path) -> None:
    config = EventDrivenOpinionConfig.from_dict(valid_event_config(tmp_path))
    payload = config.to_dict()

    round_tripped = EventDrivenOpinionConfig.from_dict(payload)

    assert round_tripped.to_dict() == payload


def test_event_config_rejects_duplicate_agent_ids(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["agents"] = [data["agents"][0], data["agents"][0]]

    config = EventDrivenOpinionConfig.from_dict(data)

    with pytest.raises(ValueError, match="agent ids must be unique"):
        config.validate()


def test_event_config_rejects_relationship_to_missing_agent(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["relationships"][0]["target_agent_id"] = "missing"  # type: ignore[index]

    config = EventDrivenOpinionConfig.from_dict(data)

    with pytest.raises(ValueError, match="relationship target_agent_id is not in agents"):
        config.validate()


def test_event_config_rejects_event_outside_day_range(tmp_path: Path) -> None:
    data = valid_event_config(tmp_path)
    data["events"][0]["day"] = 9  # type: ignore[index]

    config = EventDrivenOpinionConfig.from_dict(data)

    with pytest.raises(ValueError, match="event day must be between 0 and days"):
        config.validate()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_config.py -q
```

Expected: FAIL with missing `society_simulation.event_config`.

- [ ] **Step 3: Implement event config**

Create `src/society_simulation/event_config.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from society_simulation.event_models import EventAgentProfile, EventRelationship, OpinionEvent


def _require_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _require_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


@dataclass(frozen=True)
class EventDrivenOpinionConfig:
    experiment_name: str
    seed: int
    scenario_name: str
    days: int
    agents: tuple[EventAgentProfile, ...]
    relationships: tuple[EventRelationship, ...]
    events: tuple[OpinionEvent, ...]
    channels: tuple[dict[str, object], ...]
    update_policy: dict[str, object]
    output_dir: str

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EventDrivenOpinionConfig:
        data = _require_mapping(data, "event config")
        return cls(
            experiment_name=_require_str(data.get("experiment_name"), "experiment_name"),
            seed=_require_int(data.get("seed"), "seed"),
            scenario_name=_require_str(data.get("scenario_name"), "scenario_name"),
            days=_require_int(data.get("days"), "days"),
            agents=tuple(
                EventAgentProfile.from_dict(_require_mapping(item, "agents[]"))
                for item in _require_list(data.get("agents"), "agents")
            ),
            relationships=tuple(
                EventRelationship.from_dict(_require_mapping(item, "relationships[]"))
                for item in _require_list(data.get("relationships"), "relationships")
            ),
            events=tuple(
                OpinionEvent.from_dict(_require_mapping(item, "events[]"))
                for item in _require_list(data.get("events"), "events")
            ),
            channels=tuple(
                dict(_require_mapping(item, "channels[]"))
                for item in _require_list(data.get("channels"), "channels")
            ),
            update_policy=dict(_require_mapping(data.get("update_policy"), "update_policy")),
            output_dir=_require_str(data.get("output_dir"), "output_dir"),
        )

    def validate(self) -> None:
        if self.experiment_name != "event_driven_opinion_dynamics":
            raise ValueError("unsupported experiment_name")
        if self.days <= 0:
            raise ValueError("days must be positive")
        if not self.agents:
            raise ValueError("agents must not be empty")
        agent_ids = [agent.agent_id for agent in self.agents]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("agent ids must be unique")
        agent_id_set = set(agent_ids)
        channel_ids = {str(channel.get("channel_id")) for channel in self.channels}
        for relationship in self.relationships:
            if relationship.source_agent_id not in agent_id_set:
                raise ValueError("relationship source_agent_id is not in agents")
            if relationship.target_agent_id not in agent_id_set:
                raise ValueError("relationship target_agent_id is not in agents")
            for channel in relationship.channels:
                if channel not in channel_ids:
                    raise ValueError("relationship channel is not in channels")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event ids must be unique")
        for event in self.events:
            if not 0 <= event.day <= self.days:
                raise ValueError("event day must be between 0 and days")
        policy_type = self.update_policy.get("type")
        if policy_type not in ("mock_persona", "llm"):
            raise ValueError("unsupported event update_policy type")
        if policy_type == "mock_persona":
            response_style = self.update_policy.get("response_style", "balanced")
            if response_style not in ("balanced", "silent", "reactive"):
                raise ValueError("unsupported mock persona response_style")
        if policy_type == "llm":
            model = self.update_policy.get("model")
            if not isinstance(model, str) or not model:
                raise ValueError("model must be a non-empty string")

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "scenario_name": self.scenario_name,
            "days": self.days,
            "agents": [agent.to_dict() for agent in self.agents],
            "relationships": [relationship.to_dict() for relationship in self.relationships],
            "events": [event.to_dict() for event in self.events],
            "channels": list(self.channels),
            "update_policy": dict(self.update_policy),
            "output_dir": self.output_dir,
        }
```

Modify `src/society_simulation/config.py`:

```python
from society_simulation.event_config import EventDrivenOpinionConfig
```

Change the `Config` alias:

```python
Config = ExperimentConfig | NetworkHerdingConfig | EventDrivenOpinionConfig
```

Change `load_config()` dispatch:

```python
    if data.get("experiment_name") == "network_herding":
        config = NetworkHerdingConfig.from_dict(data)
    elif data.get("experiment_name") == "event_driven_opinion_dynamics":
        config = EventDrivenOpinionConfig.from_dict(data)
    else:
        config = ExperimentConfig(**data)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_config.py tests/test_network_config.py tests/test_cli.py::test_cli_run_invalid_config_reports_clean_error -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/event_config.py src/society_simulation/config.py tests/test_event_config.py
git commit -m "feat: add event opinion config"
```

---

### Task 3: Event Persona Policy

**Files:**
- Create: `src/society_simulation/event_policy.py`
- Test: `tests/test_event_policy.py`

- [ ] **Step 1: Write failing policy tests**

Create `tests/test_event_policy.py`:

```python
import json

import pytest

from society_simulation.event_models import EventAgentProfile, EventAgentState, EventExposure
from society_simulation.event_policy import (
    MockPersonaPolicy,
    OpenAICompatiblePersonaPolicy,
    parse_event_decision_content,
)


def profile() -> EventAgentProfile:
    return EventAgentProfile.from_dict(
        {
            "agent_id": "jisoo",
            "name": "Jisoo Park",
            "age": 37,
            "occupation": "hospital nurse",
            "household_context": "single parent",
            "neighborhood": "east side",
            "core_values": ["fairness", "public health"],
            "material_interests": ["commute time"],
            "political_trust": 0.45,
            "media_habits": ["local_news", "neighborhood_group_chat"],
            "communication_style": "careful",
            "susceptibilities": ["coworker stories"],
            "initial_private_stance": -0.1,
            "initial_public_stance": 0.0,
            "initial_confidence": 0.35,
            "initial_salience": 0.45,
        }
    )


def state() -> EventAgentState:
    return profile().initial_state(day=0)


def exposure() -> EventExposure:
    return EventExposure(
        day=1,
        agent_id="jisoo",
        source_type="event",
        source_id="city_announcement",
        channel="news_feed",
        content="City Hall says congestion pricing could reduce commute delays and pollution.",
    )


def test_parse_event_decision_content_requires_complete_json() -> None:
    decision = parse_event_decision_content(
        json.dumps(
            {
                "private_stance": 0.2,
                "public_stance": 0.1,
                "confidence": 0.6,
                "salience": 0.7,
                "emotion": "conflicted",
                "private_reasoning": "The policy helps traffic but adds costs.",
                "messages": [
                    {
                        "channel": "neighborhood_group_chat",
                        "recipient": None,
                        "text": "I can see both sides here.",
                    }
                ],
                "memory_update": "Jisoo is more conflicted after the city announcement.",
            }
        )
    )

    assert decision.state.private_stance == 0.2
    assert decision.messages[0].text == "I can see both sides here."


def test_parse_event_decision_content_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="event llm response content must be JSON"):
        parse_event_decision_content("{")


def test_mock_persona_policy_returns_audited_decision() -> None:
    policy = MockPersonaPolicy(response_style="balanced")

    decision = policy.decide(profile(), state(), (exposure(),), day=1)

    assert decision.state.agent_id == "jisoo"
    assert decision.state.day == 1
    assert decision.state.salience >= state().salience
    assert decision.messages
    usage = policy.usage_summary()
    assert usage["provider"] == "mock"
    assert usage["calls"] == 1
    assert policy.audit_records()[0]["agent_id"] == "jisoo"
    assert "Stay in character" in policy.audit_records()[0]["prompt"]


def test_mock_persona_policy_silent_style_posts_no_messages() -> None:
    policy = MockPersonaPolicy(response_style="silent")

    decision = policy.decide(profile(), state(), (exposure(),), day=1)

    assert decision.messages == ()


def test_openai_compatible_persona_policy_sends_human_role_prompt_without_experiment_language() -> None:
    captured: dict[str, object] = {}

    def transport(url: str, headers: dict[str, str], payload: dict[str, object], timeout_seconds: float) -> dict[str, object]:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        captured["timeout_seconds"] = timeout_seconds
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "private_stance": 0.1,
                                "public_stance": 0.0,
                                "confidence": 0.5,
                                "salience": 0.6,
                                "emotion": "conflicted",
                                "private_reasoning": "The announcement is plausible but costly.",
                                "messages": [],
                                "memory_update": "Jisoo remains conflicted.",
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40},
        }

    policy = OpenAICompatiblePersonaPolicy(
        model="cheap-chat",
        api_key="secret-key",
        base_url="https://example.test/v1",
        input_cost_per_1m_tokens=1.0,
        output_cost_per_1m_tokens=2.0,
        transport=transport,
    )

    decision = policy.decide(profile(), state(), (exposure(),), day=1)

    assert decision.state.private_stance == 0.1
    payload = captured["payload"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert isinstance(messages, list)
    prompt_text = json.dumps(messages)
    assert "Stay in character" in prompt_text
    assert "social network experiment" not in prompt_text
    assert "secret-key" not in json.dumps(policy.audit_records(), sort_keys=True)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_policy.py -q
```

Expected: FAIL with missing `event_policy`.

- [ ] **Step 3: Implement event policy**

Create `src/society_simulation/event_policy.py` with this public surface and behavior:

- `EventPolicyDecision`: frozen dataclass with `state: EventAgentState` and `messages: tuple[EventMessage, ...]`.
- `parse_event_decision_content(content, agent_id="agent", day=0)`: parse JSON; require `private_stance`, `public_stance`, `confidence`, `salience`, `emotion`, `private_reasoning`, `messages`, and `memory_update`; validate stance/probability through `EventAgentState`; convert each message object into `EventMessage(day=day, sender_agent_id=agent_id, channel=message["channel"], recipient_agent_id=message["recipient"], text=message["text"])`.
- `MockPersonaPolicy`: constructor accepts `response_style`, `provider`, `model`, `input_cost_per_1m_tokens`, and `output_cost_per_1m_tokens`; `decide()` returns `EventPolicyDecision`; `usage_summary()` returns the same shape as existing LLM usage summaries; `audit_records()` returns defensive copies.
- `OpenAICompatiblePersonaPolicy`: constructor mirrors `OpenAICompatibleLLMPolicy` fields; `decide()` sends chat-completion messages through `OpenAICompatibleClient`; response content is parsed by `parse_event_decision_content()`; audit records must not include API keys or authorization headers.

Implementation rules:

- Reuse `LLMPricing`, `LLMUsage`, `OpenAICompatibleClient`, and `estimate_tokens` from `society_simulation.llm_policy`.
- Build prompts with a role message that says `Stay in character. Do not mention being an AI, a model, a simulation, or an experiment.`
- Do not include the phrase `social network experiment` anywhere in event persona prompts.
- `MockPersonaPolicy(response_style="balanced")` should move private stance slightly toward the average event stance implied by exposure text keywords:
  - if exposure contains `pollution`, `traffic`, `health`, or `asthma`, add `+0.10`;
  - if exposure contains `fee`, `cost`, `burden`, `income`, or `livelihood`, add `-0.10`;
  - clamp stance to `[-1, 1]`;
  - raise salience by `0.10` when exposures are present;
  - post one group-chat message when response style is not `silent`.
- Audit rows must include `agent_id`, `day`, `provider`, `model`, `policy_type`, `prompt`, `raw_response`, parsed state fields, token counts, costs, and latency.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_policy.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/event_policy.py tests/test_event_policy.py
git commit -m "feat: add event persona policies"
```

---

### Task 4: Event Exposure Scheduling

**Files:**
- Create: `src/society_simulation/event_scheduling.py`
- Test: `tests/test_event_scheduling.py`

- [ ] **Step 1: Write failing scheduling tests**

Create `tests/test_event_scheduling.py`:

```python
from society_simulation.event_models import EventAgentProfile, EventMessage, OpinionEvent
from society_simulation.event_scheduling import build_day_exposures


def agent(agent_id: str, media_habits: list[str]) -> EventAgentProfile:
    return EventAgentProfile.from_dict(
        {
            "agent_id": agent_id,
            "name": agent_id.title(),
            "age": 30,
            "occupation": "resident",
            "household_context": "apartment",
            "neighborhood": "central",
            "core_values": ["fairness"],
            "material_interests": ["commute time"],
            "political_trust": 0.5,
            "media_habits": media_habits,
            "communication_style": "plain",
            "susceptibilities": ["friends"],
            "initial_private_stance": 0.0,
            "initial_public_stance": 0.0,
            "initial_confidence": 0.4,
            "initial_salience": 0.4,
        }
    )


def test_build_day_exposures_delivers_events_by_media_habit() -> None:
    agents = (agent("a", ["local_news"]), agent("b", ["podcast"]))
    event = OpinionEvent.from_dict(
        {
            "event_id": "news",
            "day": 1,
            "title": "City proposal",
            "source": "Local News",
            "source_type": "news",
            "content": "The fee may reduce traffic.",
            "policy_stance": 0.2,
            "credibility": 0.7,
            "emotional_intensity": 0.3,
            "affected_interests": ["commute time"],
            "audience_filter": {"media_habits_any": ["local_news"]},
        }
    )

    exposures = build_day_exposures(
        day=1,
        agents=agents,
        events=(event,),
        previous_messages=(),
        channel_members={"neighborhood": {"a", "b"}},
    )

    assert [exposure.agent_id for exposure in exposures] == ["a"]
    assert exposures[0].source_id == "news"


def test_build_day_exposures_delivers_previous_group_chat_messages_to_channel_members_except_sender() -> None:
    agents = (agent("a", ["local_news"]), agent("b", ["local_news"]))
    message = EventMessage(
        day=1,
        sender_agent_id="a",
        channel="neighborhood",
        recipient_agent_id=None,
        text="I worry this fee hurts workers.",
    )

    exposures = build_day_exposures(
        day=2,
        agents=agents,
        events=(),
        previous_messages=(message,),
        channel_members={"neighborhood": {"a", "b"}},
    )

    assert len(exposures) == 1
    assert exposures[0].agent_id == "b"
    assert exposures[0].source_type == "message"


def test_build_day_exposures_delivers_private_dm_only_to_recipient() -> None:
    agents = (agent("a", ["local_news"]), agent("b", ["local_news"]), agent("c", ["local_news"]))
    message = EventMessage(
        day=1,
        sender_agent_id="a",
        channel="private_dm",
        recipient_agent_id="c",
        text="Can we talk about the charge?",
    )

    exposures = build_day_exposures(
        day=2,
        agents=agents,
        events=(),
        previous_messages=(message,),
        channel_members={},
    )

    assert [exposure.agent_id for exposure in exposures] == ["c"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_scheduling.py -q
```

Expected: FAIL with missing `event_scheduling`.

- [ ] **Step 3: Implement event scheduling**

Create `src/society_simulation/event_scheduling.py`:

```python
from __future__ import annotations

from collections.abc import Mapping

from society_simulation.event_models import EventAgentProfile, EventExposure, EventMessage, OpinionEvent


def build_day_exposures(
    *,
    day: int,
    agents: tuple[EventAgentProfile, ...],
    events: tuple[OpinionEvent, ...],
    previous_messages: tuple[EventMessage, ...],
    channel_members: Mapping[str, set[str]],
) -> tuple[EventExposure, ...]:
    agent_by_id = {agent.agent_id: agent for agent in agents}
    exposures: list[EventExposure] = []
    for event in events:
        if event.day != day:
            continue
        for agent in agents:
            if _event_visible_to_agent(event, agent):
                exposures.append(
                    EventExposure(
                        day=day,
                        agent_id=agent.agent_id,
                        source_type="event",
                        source_id=event.event_id,
                        channel="news_feed",
                        content=f"{event.title}: {event.content}",
                    )
                )
    for message in previous_messages:
        if message.day >= day:
            continue
        if message.recipient_agent_id is not None:
            if message.recipient_agent_id in agent_by_id:
                exposures.append(
                    EventExposure(
                        day=day,
                        agent_id=message.recipient_agent_id,
                        source_type="message",
                        source_id=f"{message.sender_agent_id}:{message.day}:{message.channel}",
                        channel=message.channel,
                        content=f"{message.sender_agent_id}: {message.text}",
                    )
                )
            continue
        for agent_id in sorted(channel_members.get(message.channel, set())):
            if agent_id == message.sender_agent_id:
                continue
            exposures.append(
                EventExposure(
                    day=day,
                    agent_id=agent_id,
                    source_type="message",
                    source_id=f"{message.sender_agent_id}:{message.day}:{message.channel}",
                    channel=message.channel,
                    content=f"{message.sender_agent_id}: {message.text}",
                )
            )
    return tuple(exposures)


def _event_visible_to_agent(event: OpinionEvent, agent: EventAgentProfile) -> bool:
    media_any = event.audience_filter.get("media_habits_any")
    if isinstance(media_any, list):
        return any(item in agent.media_habits for item in media_any if isinstance(item, str))
    agent_ids = event.audience_filter.get("agent_ids")
    if isinstance(agent_ids, list):
        return agent.agent_id in {item for item in agent_ids if isinstance(item, str)}
    return True
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_scheduling.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/event_scheduling.py tests/test_event_scheduling.py
git commit -m "feat: add event exposure scheduling"
```

---

### Task 5: Event Metrics

**Files:**
- Create: `src/society_simulation/event_metrics.py`
- Test: `tests/test_event_metrics.py`

- [ ] **Step 1: Write failing metrics tests**

Create `tests/test_event_metrics.py`:

```python
import pytest

from society_simulation.event_metrics import compute_event_metrics, compute_event_timeseries
from society_simulation.event_models import EventAgentState, EventMessage


def state(agent_id: str, day: int, private: float, public: float, emotion: str = "calm") -> EventAgentState:
    return EventAgentState(
        agent_id=agent_id,
        day=day,
        private_stance=private,
        public_stance=public,
        confidence=0.5,
        salience=0.6,
        emotion=emotion,
        memory_summary="memory",
        last_private_reasoning="reason",
    )


def test_compute_event_timeseries_tracks_private_public_gap_and_messages() -> None:
    states_by_day = (
        (state("a", 0, -0.2, 0.0), state("b", 0, 0.2, 0.0)),
        (state("a", 1, -0.4, -0.1, "worried"), state("b", 1, 0.3, 0.1, "hopeful")),
    )
    messages = (
        EventMessage(day=1, sender_agent_id="a", channel="chat", recipient_agent_id=None, text="text"),
    )

    rows = compute_event_timeseries(states_by_day, messages)

    assert rows[0]["day"] == 0
    assert rows[1]["mean_private_stance"] == pytest.approx(-0.05)
    assert rows[1]["mean_public_stance"] == pytest.approx(0.0)
    assert rows[1]["mean_private_public_gap"] == pytest.approx(0.25)
    assert rows[1]["message_count"] == 1
    assert rows[1]["emotion_counts"] == {"hopeful": 1, "worried": 1}


def test_compute_event_metrics_uses_final_day() -> None:
    states_by_day = (
        (state("a", 0, 0.0, 0.0),),
        (state("a", 1, 0.5, 0.25),),
    )

    metrics = compute_event_metrics(states_by_day, messages=())

    assert metrics["final_private_stance_mean"] == pytest.approx(0.5)
    assert metrics["final_public_stance_mean"] == pytest.approx(0.25)
    assert metrics["agent_count"] == 1
    assert metrics["day_count"] == 2
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_metrics.py -q
```

Expected: FAIL with missing `event_metrics`.

- [ ] **Step 3: Implement event metrics**

Create `src/society_simulation/event_metrics.py` with:

```python
from __future__ import annotations

from collections import Counter
from statistics import mean, pvariance
from typing import Any

from society_simulation.event_models import EventAgentState, EventMessage


def compute_event_timeseries(
    states_by_day: tuple[tuple[EventAgentState, ...], ...],
    messages: tuple[EventMessage, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for states in states_by_day:
        if not states:
            raise ValueError("states_by_day must not contain empty days")
        day = states[0].day
        private_values = [state.private_stance for state in states]
        public_values = [state.public_stance for state in states]
        gaps = [abs(state.private_stance - state.public_stance) for state in states]
        day_messages = [message for message in messages if message.day == day]
        rows.append(
            {
                "day": day,
                "mean_private_stance": mean(private_values),
                "mean_public_stance": mean(public_values),
                "mean_private_public_gap": mean(gaps),
                "private_stance_variance": pvariance(private_values) if len(private_values) > 1 else 0.0,
                "public_stance_variance": pvariance(public_values) if len(public_values) > 1 else 0.0,
                "mean_confidence": mean(state.confidence for state in states),
                "mean_salience": mean(state.salience for state in states),
                "message_count": len(day_messages),
                "private_message_count": sum(1 for message in day_messages if message.recipient_agent_id is not None),
                "public_message_count": sum(1 for message in day_messages if message.recipient_agent_id is None),
                "emotion_counts": dict(sorted(Counter(state.emotion for state in states).items())),
            }
        )
    return rows


def compute_event_metrics(
    states_by_day: tuple[tuple[EventAgentState, ...], ...],
    messages: tuple[EventMessage, ...],
) -> dict[str, Any]:
    if not states_by_day:
        raise ValueError("states_by_day must not be empty")
    rows = compute_event_timeseries(states_by_day, messages)
    final = rows[-1]
    return {
        "agent_count": len(states_by_day[-1]),
        "day_count": len(states_by_day),
        "message_count": len(messages),
        "final_private_stance_mean": final["mean_private_stance"],
        "final_public_stance_mean": final["mean_public_stance"],
        "final_private_public_gap": final["mean_private_public_gap"],
        "final_private_stance_variance": final["private_stance_variance"],
        "final_public_stance_variance": final["public_stance_variance"],
        "final_mean_confidence": final["mean_confidence"],
        "final_mean_salience": final["mean_salience"],
        "timeseries": rows,
    }
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/event_metrics.py tests/test_event_metrics.py
git commit -m "feat: add event opinion metrics"
```

---

### Task 6: Event Replay Writer

**Files:**
- Create: `src/society_simulation/event_replay.py`
- Test: `tests/test_event_replay.py`

- [ ] **Step 1: Write failing replay tests**

Create `tests/test_event_replay.py`:

```python
import json
from pathlib import Path

from tests.test_event_config import valid_event_config

from society_simulation.event_config import EventDrivenOpinionConfig
from society_simulation.event_models import EventExposure, EventMessage
from society_simulation.event_replay import EventReplayWriter


def test_event_replay_writer_writes_required_artifacts(tmp_path: Path) -> None:
    config = EventDrivenOpinionConfig.from_dict(valid_event_config(tmp_path))
    states_by_day = (
        tuple(agent.initial_state(day=0) for agent in config.agents),
        tuple(agent.initial_state(day=1) for agent in config.agents),
    )
    exposures = (
        EventExposure(
            day=1,
            agent_id="jisoo",
            source_type="event",
            source_id="city_announcement",
            channel="news_feed",
            content="City announcement.",
        ),
    )
    messages = (
        EventMessage(
            day=1,
            sender_agent_id="jisoo",
            channel="neighborhood_group_chat",
            recipient_agent_id=None,
            text="I am thinking about the fee.",
        ),
    )
    metrics = {"agent_count": 2, "day_count": 2, "final_private_stance_mean": 0.0}
    llm_decisions = ({"agent_id": "jisoo", "day": 1, "prompt": "prompt", "raw_response": {"content": "{}"}},)

    output_dir = EventReplayWriter(config).write(
        states_by_day=states_by_day,
        exposures=exposures,
        messages=messages,
        metrics=metrics,
        llm_decisions=llm_decisions,
    )

    assert output_dir == Path(config.output_dir)
    for name in [
        "config.json",
        "agents.json",
        "relationships.json",
        "events.json",
        "exposures.jsonl",
        "messages.jsonl",
        "agent_states.jsonl",
        "metrics.json",
        "summary.md",
        "llm_decisions.jsonl",
    ]:
        assert (output_dir / name).exists()
    state_rows = [json.loads(line) for line in (output_dir / "agent_states.jsonl").read_text(encoding="utf-8").splitlines()]
    assert state_rows[0]["agent_id"] == "jisoo"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_replay.py -q
```

Expected: FAIL with missing `event_replay`.

- [ ] **Step 3: Implement replay writer**

Create `src/society_simulation/event_replay.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from society_simulation.event_config import EventDrivenOpinionConfig
from society_simulation.event_models import EventAgentState, EventExposure, EventMessage


class EventReplayWriter:
    def __init__(self, config: EventDrivenOpinionConfig) -> None:
        self.config = config

    def write(
        self,
        *,
        states_by_day: tuple[tuple[EventAgentState, ...], ...],
        exposures: tuple[EventExposure, ...],
        messages: tuple[EventMessage, ...],
        metrics: dict[str, Any],
        llm_decisions: tuple[dict[str, Any], ...],
    ) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(output_dir / "config.json", self.config.to_dict())
        self._write_json(output_dir / "agents.json", {"agents": [agent.to_dict() for agent in self.config.agents]})
        self._write_json(output_dir / "relationships.json", {"relationships": [relationship.to_dict() for relationship in self.config.relationships]})
        self._write_json(output_dir / "events.json", {"events": [event.to_dict() for event in self.config.events]})
        self._write_jsonl(output_dir / "exposures.jsonl", tuple(exposure.to_dict() for exposure in exposures))
        self._write_jsonl(output_dir / "messages.jsonl", tuple(message.to_dict() for message in messages))
        self._write_jsonl(
            output_dir / "agent_states.jsonl",
            tuple(state.to_dict() for day_states in states_by_day for state in day_states),
        )
        self._write_json(output_dir / "metrics.json", metrics)
        if llm_decisions:
            self._write_jsonl(output_dir / "llm_decisions.jsonl", llm_decisions)
        self._write_summary(output_dir / "summary.md", metrics)
        return output_dir

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: tuple[dict[str, Any], ...]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _write_summary(self, path: Path, metrics: dict[str, Any]) -> None:
        lines = [
            f"# {self.config.scenario_name}",
            "",
            f"- experiment_name: `{self.config.experiment_name}`",
            f"- agents: `{metrics.get('agent_count')}`",
            f"- days: `{metrics.get('day_count')}`",
            f"- final_private_stance_mean: `{metrics.get('final_private_stance_mean')}`",
            f"- final_public_stance_mean: `{metrics.get('final_public_stance_mean')}`",
            f"- final_private_public_gap: `{metrics.get('final_private_public_gap')}`",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_replay.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/event_replay.py tests/test_event_replay.py
git commit -m "feat: add event replay artifacts"
```

---

### Task 7: Event Runner With Mock Persona Policy

**Files:**
- Create: `src/society_simulation/event_runner.py`
- Modify: `src/society_simulation/runner.py`
- Test: `tests/test_event_runner.py`

- [ ] **Step 1: Write failing runner tests**

Create `tests/test_event_runner.py`:

```python
import json
from pathlib import Path

from tests.test_event_config import valid_event_config

from society_simulation.config import load_config
from society_simulation.event_config import EventDrivenOpinionConfig
from society_simulation.runner import run_experiment


def test_run_experiment_dispatches_event_driven_opinion_dynamics(tmp_path: Path) -> None:
    path = tmp_path / "event.json"
    path.write_text(json.dumps(valid_event_config(tmp_path)), encoding="utf-8")
    config = load_config(path)
    assert isinstance(config, EventDrivenOpinionConfig)

    result = run_experiment(config)

    assert result.output_dir == tmp_path / "event-run"
    assert len(result.states_by_day) == config.days + 1
    assert len(result.states_by_day[0]) == len(config.agents)
    assert result.metrics["agent_count"] == len(config.agents)
    assert "llm_usage" in result.metrics
    assert (result.output_dir / "agent_states.jsonl").exists()
    assert (result.output_dir / "messages.jsonl").exists()
    assert (result.output_dir / "llm_decisions.jsonl").exists()


def test_event_runner_is_deterministic_for_mock_policy(tmp_path: Path) -> None:
    first_data = valid_event_config(tmp_path)
    second_data = valid_event_config(tmp_path)
    first_data["output_dir"] = str(tmp_path / "first")
    second_data["output_dir"] = str(tmp_path / "second")

    first = run_experiment(EventDrivenOpinionConfig.from_dict(first_data))
    second = run_experiment(EventDrivenOpinionConfig.from_dict(second_data))

    assert first.metrics == second.metrics
    assert (first.output_dir / "agent_states.jsonl").read_text(encoding="utf-8") == (
        second.output_dir / "agent_states.jsonl"
    ).read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_runner.py -q
```

Expected: FAIL with missing event runner dispatch.

- [ ] **Step 3: Implement event runner and dispatch**

Create `src/society_simulation/event_runner.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from society_simulation.event_config import EventDrivenOpinionConfig
from society_simulation.event_metrics import compute_event_metrics
from society_simulation.event_models import EventAgentState, EventExposure, EventMessage
from society_simulation.event_policy import MockPersonaPolicy, OpenAICompatiblePersonaPolicy
from society_simulation.event_replay import EventReplayWriter
from society_simulation.event_scheduling import build_day_exposures


@dataclass(frozen=True)
class EventRunResult:
    states_by_day: tuple[tuple[EventAgentState, ...], ...]
    exposures: tuple[EventExposure, ...]
    messages: tuple[EventMessage, ...]
    metrics: dict[str, Any]
    output_dir: Path


def run_event_driven_opinion_dynamics(config: EventDrivenOpinionConfig) -> EventRunResult:
    config.validate()
    policy = _build_event_policy(config)
    channel_members = _channel_members(config)
    states_by_day: list[tuple[EventAgentState, ...]] = [
        tuple(agent.initial_state(day=0) for agent in config.agents)
    ]
    all_exposures: list[EventExposure] = []
    all_messages: list[EventMessage] = []
    for day in range(1, config.days + 1):
        previous_states = {state.agent_id: state for state in states_by_day[-1]}
        day_exposures = build_day_exposures(
            day=day,
            agents=config.agents,
            events=config.events,
            previous_messages=tuple(all_messages),
            channel_members=channel_members,
        )
        all_exposures.extend(day_exposures)
        exposures_by_agent: dict[str, list[EventExposure]] = {}
        for exposure in day_exposures:
            exposures_by_agent.setdefault(exposure.agent_id, []).append(exposure)
        next_states: list[EventAgentState] = []
        for profile in config.agents:
            decision = policy.decide(
                profile,
                previous_states[profile.agent_id],
                tuple(exposures_by_agent.get(profile.agent_id, [])),
                day=day,
            )
            next_states.append(decision.state)
            all_messages.extend(decision.messages)
        states_by_day.append(tuple(next_states))
    frozen_states = tuple(states_by_day)
    frozen_messages = tuple(all_messages)
    metrics = compute_event_metrics(frozen_states, frozen_messages)
    usage_summary = getattr(policy, "usage_summary", None)
    if callable(usage_summary):
        metrics["llm_usage"] = usage_summary()
    audit_records = getattr(policy, "audit_records", None)
    llm_decisions = audit_records() if callable(audit_records) else ()
    output_dir = EventReplayWriter(config).write(
        states_by_day=frozen_states,
        exposures=tuple(all_exposures),
        messages=frozen_messages,
        metrics=metrics,
        llm_decisions=llm_decisions,
    )
    return EventRunResult(
        states_by_day=frozen_states,
        exposures=tuple(all_exposures),
        messages=frozen_messages,
        metrics=metrics,
        output_dir=output_dir,
    )


def _channel_members(config: EventDrivenOpinionConfig) -> dict[str, set[str]]:
    members: dict[str, set[str]] = {}
    for relationship in config.relationships:
        for channel in relationship.channels:
            members.setdefault(channel, set()).add(relationship.source_agent_id)
            members.setdefault(channel, set()).add(relationship.target_agent_id)
    return members


def _build_event_policy(config: EventDrivenOpinionConfig) -> object:
    policy = config.update_policy
    if policy["type"] == "mock_persona":
        return MockPersonaPolicy(
            response_style=str(policy.get("response_style", "balanced")),
            input_cost_per_1m_tokens=float(policy.get("input_cost_per_1m_tokens", 0.0)),
            output_cost_per_1m_tokens=float(policy.get("output_cost_per_1m_tokens", 0.0)),
        )
    api_key_env = str(policy.get("api_key_env", "SOCIETY_SIM_LLM_API_KEY"))
    import os

    return OpenAICompatiblePersonaPolicy(
        model=str(policy["model"]),
        api_key=os.environ.get(api_key_env, ""),
        base_url=str(policy.get("base_url", "https://api.openai.com/v1")),
        temperature=float(policy.get("temperature", 0.0)),
        max_completion_tokens=int(policy.get("max_completion_tokens", 160)),
        token_limit_parameter=str(policy.get("token_limit_parameter", "max_completion_tokens")),
        timeout_seconds=float(policy.get("timeout_seconds", 30.0)),
        input_cost_per_1m_tokens=float(policy.get("input_cost_per_1m_tokens", 0.0)),
        output_cost_per_1m_tokens=float(policy.get("output_cost_per_1m_tokens", 0.0)),
        max_estimated_cost_usd=(
            None
            if "max_estimated_cost_usd" not in policy
            else float(policy["max_estimated_cost_usd"])
        ),
    )
```

Modify `src/society_simulation/runner.py`:

```python
from society_simulation.event_config import EventDrivenOpinionConfig
from society_simulation.event_runner import EventRunResult, run_event_driven_opinion_dynamics
```

Change the return type and dispatch:

```python
def run_experiment(config: Config) -> RunResult | NetworkRunResult | EventRunResult:
    if isinstance(config, NetworkHerdingConfig):
        return run_network_herding(config)
    if isinstance(config, EventDrivenOpinionConfig):
        return run_event_driven_opinion_dynamics(config)
    return run_sequential_information_cascade(config)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_runner.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/society_simulation/event_runner.py src/society_simulation/runner.py tests/test_event_runner.py
git commit -m "feat: run event opinion dynamics"
```

---

### Task 8: CLI, Example Experiment, And Sweep Summary Compatibility

**Files:**
- Create: `experiments/event_driven_congestion_pricing.json`
- Modify: `src/society_simulation/cli.py`
- Modify: `src/society_simulation/sweep_artifacts.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_sweep_artifacts.py`

- [ ] **Step 1: Write failing CLI test**

Append to `tests/test_cli.py`:

```python
def test_cli_runs_event_driven_opinion_config_and_prints_event_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.test_event_config import valid_event_config

    config_path = tmp_path / "event.json"
    output_dir = tmp_path / "event-run"
    data = valid_event_config(tmp_path)
    data["output_dir"] = str(output_dir)
    config_path.write_text(json.dumps(data), encoding="utf-8")

    exit_code = cli.main(["run", str(config_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "experiment=event_driven_opinion_dynamics" in output
    assert "final_private_stance_mean=" in output
    assert "final_public_stance_mean=" in output
    assert "final_private_public_gap=" in output
    assert "llm_calls=" in output
    assert f"output_dir={output_dir}" in output
    assert (output_dir / "summary.md").exists()


def test_event_driven_congestion_pricing_experiment_exists_and_is_valid() -> None:
    from society_simulation.event_config import EventDrivenOpinionConfig
    from society_simulation.config import load_config

    config = load_config("experiments/event_driven_congestion_pricing.json")

    assert isinstance(config, EventDrivenOpinionConfig)
    assert config.experiment_name == "event_driven_opinion_dynamics"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cli.py::test_cli_runs_event_driven_opinion_config_and_prints_event_summary tests/test_cli.py::test_event_driven_congestion_pricing_experiment_exists_and_is_valid -q
```

Expected: FAIL because CLI requires action counts and the experiment file does not exist.

- [ ] **Step 3: Update CLI and add experiment config**

Modify `src/society_simulation/cli.py`:

```python
def _print_event_metrics(metrics: dict[str, object]) -> None:
    if "final_private_stance_mean" not in metrics:
        return
    print(f"final_private_stance_mean={metrics['final_private_stance_mean']}")
    print(f"final_public_stance_mean={metrics['final_public_stance_mean']}")
    print(f"final_private_public_gap={metrics['final_private_public_gap']}")
    print(f"message_count={metrics.get('message_count', 0)}")
```

Change `_run_single_config()` so `action_counts` is optional for event metrics:

```python
        result = run_experiment(config)
        metrics = result.metrics
        action_counts = None
        if "final_private_stance_mean" not in metrics:
            action_counts = _require_action_counts(metrics)
```

Change printing:

```python
    if action_counts is not None:
        print(f"action_counts={action_counts}")
```

Add after network metric printing:

```python
    _print_event_metrics(metrics)
```

Create `experiments/event_driven_congestion_pricing.json` with 8 agents, 7 days, one group channel, relationships, and these events:

- day 1: city announcement;
- day 2: taxi driver livelihood story;
- day 3: public-health report;
- day 4: local business owner interview;
- day 5: viral clip alleging revenue motives;
- day 6: local newspaper fact-check;
- day 7: final neighborhood discussion prompt.

Use:

```json
"update_policy": {
  "type": "mock_persona",
  "response_style": "balanced",
  "input_cost_per_1m_tokens": 0.0,
  "output_cost_per_1m_tokens": 0.0
}
```

Set:

```json
"output_dir": "runs/event_driven_congestion_pricing"
```

- [ ] **Step 4: Extend sweep artifacts for event metrics**

Modify `src/society_simulation/sweep_artifacts.py`:

- Add event metric fields to `METRIC_FIELDS`:

```python
    "final_private_stance_mean",
    "final_public_stance_mean",
    "final_private_public_gap",
    "final_private_stance_variance",
    "final_public_stance_variance",
    "final_mean_confidence",
    "final_mean_salience",
    "message_count",
    "agent_count",
    "day_count",
```

- Add numeric event fields to `NUMERIC_MEAN_FIELDS`.
- Keep `_flatten_metrics()` filling missing fields with `None`.

Add one test to `tests/test_sweep_artifacts.py` that passes metrics with `final_private_stance_mean` and asserts `summary.csv` contains that column and value.

- [ ] **Step 5: Run tests to verify pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_cli.py::test_cli_runs_event_driven_opinion_config_and_prints_event_summary tests/test_cli.py::test_event_driven_congestion_pricing_experiment_exists_and_is_valid tests/test_sweep_artifacts.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/society_simulation/cli.py src/society_simulation/sweep_artifacts.py tests/test_cli.py tests/test_sweep_artifacts.py experiments/event_driven_congestion_pricing.json
git commit -m "feat: expose event opinion scenario in cli"
```

---

### Task 9: End-To-End Verification And Research Run Protocol

**Files:**
- Create: `docs/research/2026-06-19-event-driven-congestion-pricing-pilot.md`
- Modify: none unless verification finds a real issue.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
./.venv/bin/python -m pytest tests/test_event_models.py tests/test_event_config.py tests/test_event_policy.py tests/test_event_scheduling.py tests/test_event_metrics.py tests/test_event_replay.py tests/test_event_runner.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```bash
./.venv/bin/python -m pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run the mock experiment**

Run:

```bash
./.venv/bin/python -m society_simulation run experiments/event_driven_congestion_pricing.json
```

Expected output includes:

```text
experiment=event_driven_opinion_dynamics
final_private_stance_mean=
final_public_stance_mean=
final_private_public_gap=
llm_calls=
output_dir=runs/event_driven_congestion_pricing
```

- [ ] **Step 4: Inspect replay artifacts**

Run:

```bash
ls runs/event_driven_congestion_pricing
head -n 3 runs/event_driven_congestion_pricing/agent_states.jsonl
head -n 3 runs/event_driven_congestion_pricing/messages.jsonl
head -n 3 runs/event_driven_congestion_pricing/exposures.jsonl
```

Expected:

- `agent_states.jsonl` contains one row per agent per day;
- `messages.jsonl` contains natural-language messages;
- `exposures.jsonl` shows event/message exposure provenance;
- `summary.md` exists.

- [ ] **Step 5: Write pilot note**

Create `docs/research/2026-06-19-event-driven-congestion-pricing-pilot.md` with:

```markdown
# Event-Driven Congestion Pricing Pilot

Date: 2026-06-19

## Status

Mock-policy pilot completed. This is not a claim about real human opinion. It verifies that the event-driven scenario can produce auditable traces with agents, relationships, staged events, exposures, messages, private/public stance, memory, and metrics.

## Purpose

The purpose is to replace the previous binary neighbor-action toy with a richer trace-validity experiment. The key question is whether the simulator can represent people encountering events and conversations over time, not whether a population converges to a label.

## Configuration

- Scenario: congestion pricing in a fictional mid-sized city.
- Agents: 8.
- Days: 7.
- Policy: mock persona.
- Output: `runs/event_driven_congestion_pricing`.

## What To Inspect

- `exposures.jsonl`: what each agent saw.
- `messages.jsonl`: what each agent said.
- `agent_states.jsonl`: private stance, public stance, confidence, salience, emotion, memory.
- `llm_decisions.jsonl`: prompt and structured decision audit.
- `metrics.json`: aggregate public/private stance and message metrics.

## Next Paid Pilot

Run the same config with `update_policy.type = "llm"` and a strict cost cap after reviewing mock traces. The paid pilot should use 8 agents, 7 days, max completion tokens around 160, and one seed.
```

- [ ] **Step 6: Commit**

```bash
git add docs/research/2026-06-19-event-driven-congestion-pricing-pilot.md
git commit -m "docs: record event congestion pricing pilot protocol"
```

---

## Final Verification

Run:

```bash
./.venv/bin/python -m pytest -q
git diff --check
git status --short
```

Expected:

- all tests pass;
- `git diff --check` prints no whitespace errors;
- `git status --short` is empty after final commit.

## Execution Recommendation

Use `superpowers:subagent-driven-development`.

Reason: tasks are independent enough for fresh context per unit, and each task has a test boundary. The best sequence is strict task order because later tasks depend on earlier modules:

1. models;
2. config;
3. policy;
4. scheduling;
5. metrics;
6. replay;
7. runner;
8. CLI/example/sweep compatibility;
9. verification and pilot note.
