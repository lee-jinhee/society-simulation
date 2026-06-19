from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field as dataclass_field
from math import isfinite
from types import MappingProxyType


def validate_probability(value: object, field: str) -> float:
    message = f"{field} must be a number between 0 and 1"
    return _validate_number(value, 0.0, 1.0, message)


def validate_stance(value: object, field: str) -> float:
    message = f"{field} must be a number between -1 and 1"
    return _validate_number(value, -1.0, 1.0, message)


def _validate_number(
    value: object,
    min_value: float,
    max_value: float,
    message: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)

    parsed = float(value)
    if not isfinite(parsed) or not min_value <= parsed <= max_value:
        raise ValueError(message)

    return parsed


def _require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_field(data: dict[str, object], field_name: str) -> object:
    if field_name not in data:
        raise ValueError(f"{field_name} is required")
    return data[field_name]


def _require_non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    parsed = _require_int(value, field_name)
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return parsed


def _require_str_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list")

    return tuple(
        _require_non_empty_str(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


def _freeze_audience_filter_value(value: object) -> object:
    if isinstance(value, dict):
        copied = {key: _freeze_audience_filter_value(item) for key, item in value.items()}
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_audience_filter_value(item) for item in value)
    return value


def _to_json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _to_json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_to_json_ready(item) for item in value]
    return value


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
        object.__setattr__(self, "agent_id", _require_non_empty_str(self.agent_id, "agent_id"))
        object.__setattr__(self, "day", _require_non_negative_int(self.day, "day"))
        object.__setattr__(
            self,
            "private_stance",
            validate_stance(self.private_stance, "private_stance"),
        )
        object.__setattr__(
            self,
            "public_stance",
            validate_stance(self.public_stance, "public_stance"),
        )
        object.__setattr__(self, "confidence", validate_probability(self.confidence, "confidence"))
        object.__setattr__(self, "salience", validate_probability(self.salience, "salience"))
        object.__setattr__(self, "emotion", _require_non_empty_str(self.emotion, "emotion"))
        object.__setattr__(
            self,
            "memory_summary",
            _require_non_empty_str(self.memory_summary, "memory_summary"),
        )
        object.__setattr__(
            self,
            "last_private_reasoning",
            _require_non_empty_str(self.last_private_reasoning, "last_private_reasoning"),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
        data = _require_mapping(data, "agent_profile")
        return cls(
            agent_id=_require_non_empty_str(_require_field(data, "agent_id"), "agent_id"),
            name=_require_non_empty_str(_require_field(data, "name"), "name"),
            age=_require_non_negative_int(_require_field(data, "age"), "age"),
            occupation=_require_non_empty_str(_require_field(data, "occupation"), "occupation"),
            household_context=_require_non_empty_str(
                _require_field(data, "household_context"),
                "household_context",
            ),
            neighborhood=_require_non_empty_str(
                _require_field(data, "neighborhood"),
                "neighborhood",
            ),
            core_values=_require_str_tuple(_require_field(data, "core_values"), "core_values"),
            material_interests=_require_str_tuple(
                _require_field(data, "material_interests"),
                "material_interests",
            ),
            political_trust=validate_probability(
                _require_field(data, "political_trust"),
                "political_trust",
            ),
            media_habits=_require_str_tuple(_require_field(data, "media_habits"), "media_habits"),
            communication_style=_require_non_empty_str(
                _require_field(data, "communication_style"),
                "communication_style",
            ),
            susceptibilities=_require_str_tuple(
                _require_field(data, "susceptibilities"),
                "susceptibilities",
            ),
            initial_private_stance=validate_stance(
                _require_field(data, "initial_private_stance"),
                "initial_private_stance",
            ),
            initial_public_stance=validate_stance(
                _require_field(data, "initial_public_stance"),
                "initial_public_stance",
            ),
            initial_confidence=validate_probability(
                _require_field(data, "initial_confidence"),
                "initial_confidence",
            ),
            initial_salience=validate_probability(
                _require_field(data, "initial_salience"),
                "initial_salience",
            ),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _require_non_empty_str(self.agent_id, "agent_id"))
        object.__setattr__(self, "name", _require_non_empty_str(self.name, "name"))
        object.__setattr__(self, "age", _require_non_negative_int(self.age, "age"))
        object.__setattr__(
            self,
            "occupation",
            _require_non_empty_str(self.occupation, "occupation"),
        )
        object.__setattr__(
            self,
            "household_context",
            _require_non_empty_str(self.household_context, "household_context"),
        )
        object.__setattr__(
            self,
            "neighborhood",
            _require_non_empty_str(self.neighborhood, "neighborhood"),
        )
        object.__setattr__(self, "core_values", _require_str_tuple(self.core_values, "core_values"))
        object.__setattr__(
            self,
            "material_interests",
            _require_str_tuple(self.material_interests, "material_interests"),
        )
        object.__setattr__(
            self,
            "political_trust",
            validate_probability(self.political_trust, "political_trust"),
        )
        object.__setattr__(
            self,
            "media_habits",
            _require_str_tuple(self.media_habits, "media_habits"),
        )
        object.__setattr__(
            self,
            "communication_style",
            _require_non_empty_str(self.communication_style, "communication_style"),
        )
        object.__setattr__(
            self,
            "susceptibilities",
            _require_str_tuple(self.susceptibilities, "susceptibilities"),
        )
        object.__setattr__(
            self,
            "initial_private_stance",
            validate_stance(self.initial_private_stance, "initial_private_stance"),
        )
        object.__setattr__(
            self,
            "initial_public_stance",
            validate_stance(self.initial_public_stance, "initial_public_stance"),
        )
        object.__setattr__(
            self,
            "initial_confidence",
            validate_probability(self.initial_confidence, "initial_confidence"),
        )
        object.__setattr__(
            self,
            "initial_salience",
            validate_probability(self.initial_salience, "initial_salience"),
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
            memory_summary=f"{self.name} starts with their existing perspective.",
            last_private_reasoning=f"{self.name}'s initial view reflects their profile.",
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
            source_agent_id=_require_non_empty_str(
                _require_field(data, "source_agent_id"),
                "source_agent_id",
            ),
            target_agent_id=_require_non_empty_str(
                _require_field(data, "target_agent_id"),
                "target_agent_id",
            ),
            relationship_type=_require_non_empty_str(
                _require_field(data, "relationship_type"),
                "relationship_type",
            ),
            trust=validate_probability(_require_field(data, "trust"), "trust"),
            conversation_frequency=_require_non_empty_str(
                _require_field(data, "conversation_frequency"),
                "conversation_frequency",
            ),
            conflict_sensitivity=validate_probability(
                _require_field(data, "conflict_sensitivity"),
                "conflict_sensitivity",
            ),
            channels=_require_str_tuple(_require_field(data, "channels"), "channels"),
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_agent_id",
            _require_non_empty_str(self.source_agent_id, "source_agent_id"),
        )
        object.__setattr__(
            self,
            "target_agent_id",
            _require_non_empty_str(self.target_agent_id, "target_agent_id"),
        )
        object.__setattr__(
            self,
            "relationship_type",
            _require_non_empty_str(self.relationship_type, "relationship_type"),
        )
        object.__setattr__(self, "trust", validate_probability(self.trust, "trust"))
        object.__setattr__(
            self,
            "conversation_frequency",
            _require_non_empty_str(self.conversation_frequency, "conversation_frequency"),
        )
        object.__setattr__(
            self,
            "conflict_sensitivity",
            validate_probability(self.conflict_sensitivity, "conflict_sensitivity"),
        )
        object.__setattr__(self, "channels", _require_str_tuple(self.channels, "channels"))

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
    audience_filter: Mapping[str, object] = dataclass_field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> OpinionEvent:
        data = _require_mapping(data, "event")
        audience_filter = data.get("audience_filter", {})
        if not isinstance(audience_filter, dict):
            raise ValueError("audience_filter must be an object")

        return cls(
            event_id=_require_non_empty_str(_require_field(data, "event_id"), "event_id"),
            day=_require_non_negative_int(_require_field(data, "day"), "day"),
            title=_require_non_empty_str(_require_field(data, "title"), "title"),
            source=_require_non_empty_str(_require_field(data, "source"), "source"),
            source_type=_require_non_empty_str(_require_field(data, "source_type"), "source_type"),
            content=_require_non_empty_str(_require_field(data, "content"), "content"),
            policy_stance=validate_stance(_require_field(data, "policy_stance"), "policy_stance"),
            credibility=validate_probability(_require_field(data, "credibility"), "credibility"),
            emotional_intensity=validate_probability(
                _require_field(data, "emotional_intensity"),
                "emotional_intensity",
            ),
            affected_interests=_require_str_tuple(
                _require_field(data, "affected_interests"),
                "affected_interests",
            ),
            audience_filter=audience_filter,
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _require_non_empty_str(self.event_id, "event_id"))
        object.__setattr__(self, "day", _require_non_negative_int(self.day, "day"))
        object.__setattr__(self, "title", _require_non_empty_str(self.title, "title"))
        object.__setattr__(self, "source", _require_non_empty_str(self.source, "source"))
        object.__setattr__(
            self,
            "source_type",
            _require_non_empty_str(self.source_type, "source_type"),
        )
        object.__setattr__(self, "content", _require_non_empty_str(self.content, "content"))
        object.__setattr__(
            self,
            "policy_stance",
            validate_stance(self.policy_stance, "policy_stance"),
        )
        object.__setattr__(
            self,
            "credibility",
            validate_probability(self.credibility, "credibility"),
        )
        object.__setattr__(
            self,
            "emotional_intensity",
            validate_probability(self.emotional_intensity, "emotional_intensity"),
        )
        object.__setattr__(
            self,
            "affected_interests",
            _require_str_tuple(self.affected_interests, "affected_interests"),
        )
        if not isinstance(self.audience_filter, dict):
            raise ValueError("audience_filter must be an object")
        object.__setattr__(
            self,
            "audience_filter",
            _freeze_audience_filter_value(self.audience_filter),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "day": self.day,
            "title": self.title,
            "source": self.source,
            "source_type": self.source_type,
            "content": self.content,
            "policy_stance": self.policy_stance,
            "credibility": self.credibility,
            "emotional_intensity": self.emotional_intensity,
            "affected_interests": list(self.affected_interests),
            "audience_filter": _to_json_ready(self.audience_filter),
        }


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
