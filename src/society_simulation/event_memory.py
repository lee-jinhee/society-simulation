from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
import re

from society_simulation.event_models import EventExposure, validate_probability, validate_stance

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_VALID_MEMORY_KINDS = {
    "event_exposure",
    "social_message",
    "self_reasoning",
    "self_message",
    "reflection",
}


def _require_non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _str_tuple(value: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    return tuple(
        _require_non_empty_str(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


@dataclass(frozen=True)
class SocialMemory:
    memory_id: str
    agent_id: str
    day: int
    kind: str
    text: str
    source_id: str
    source_type: str
    channel: str
    related_agent_ids: tuple[str, ...]
    related_event_ids: tuple[str, ...]
    stance_signal: float
    emotional_intensity: float
    source_trust: float
    identity_relevance: float
    importance: float
    private: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_id", _require_non_empty_str(self.memory_id, "memory_id"))
        object.__setattr__(self, "agent_id", _require_non_empty_str(self.agent_id, "agent_id"))
        object.__setattr__(self, "day", _require_non_negative_int(self.day, "day"))
        object.__setattr__(self, "kind", _require_non_empty_str(self.kind, "kind"))
        if self.kind not in _VALID_MEMORY_KINDS:
            raise ValueError("kind must be a supported social memory kind")
        object.__setattr__(self, "text", _require_non_empty_str(self.text, "text"))
        object.__setattr__(self, "source_id", _require_non_empty_str(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "source_type",
            _require_non_empty_str(self.source_type, "source_type"),
        )
        object.__setattr__(self, "channel", _require_non_empty_str(self.channel, "channel"))
        object.__setattr__(
            self,
            "related_agent_ids",
            _str_tuple(self.related_agent_ids, "related_agent_ids"),
        )
        object.__setattr__(
            self,
            "related_event_ids",
            _str_tuple(self.related_event_ids, "related_event_ids"),
        )
        object.__setattr__(
            self,
            "stance_signal",
            validate_stance(self.stance_signal, "stance_signal"),
        )
        object.__setattr__(
            self,
            "emotional_intensity",
            validate_probability(self.emotional_intensity, "emotional_intensity"),
        )
        object.__setattr__(
            self,
            "source_trust",
            validate_probability(self.source_trust, "source_trust"),
        )
        object.__setattr__(
            self,
            "identity_relevance",
            validate_probability(self.identity_relevance, "identity_relevance"),
        )
        object.__setattr__(
            self,
            "importance",
            validate_probability(self.importance, "importance"),
        )
        if not isinstance(self.private, bool):
            raise ValueError("private must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryQuery:
    agent_id: str
    day: int
    text: str
    related_agent_ids: tuple[str, ...]
    related_event_ids: tuple[str, ...]
    stance_hint: float
    affected_interests: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _require_non_empty_str(self.agent_id, "agent_id"))
        object.__setattr__(self, "day", _require_non_negative_int(self.day, "day"))
        object.__setattr__(self, "text", _require_non_empty_str(self.text, "text"))
        object.__setattr__(
            self,
            "related_agent_ids",
            _str_tuple(self.related_agent_ids, "related_agent_ids"),
        )
        object.__setattr__(
            self,
            "related_event_ids",
            _str_tuple(self.related_event_ids, "related_event_ids"),
        )
        object.__setattr__(self, "stance_hint", validate_stance(self.stance_hint, "stance_hint"))
        object.__setattr__(
            self,
            "affected_interests",
            _str_tuple(self.affected_interests, "affected_interests"),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedMemory:
    memory: SocialMemory
    score: float
    recency_score: float
    relevance_score: float
    importance_score: float
    trust_score: float
    emotion_score: float
    identity_score: float

    def __post_init__(self) -> None:
        for field_name in (
            "score",
            "recency_score",
            "relevance_score",
            "importance_score",
            "trust_score",
            "emotion_score",
            "identity_score",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_probability(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
            "recency_score": self.recency_score,
            "relevance_score": self.relevance_score,
            "importance_score": self.importance_score,
            "trust_score": self.trust_score,
            "emotion_score": self.emotion_score,
            "identity_score": self.identity_score,
        }


def build_memory_query(
    *,
    agent_id: str,
    day: int,
    exposures: tuple[EventExposure, ...],
    affected_interests: tuple[str, ...],
) -> MemoryQuery:
    text = " ".join(exposure.content for exposure in exposures).strip()
    related_agent_ids = tuple(
        sorted(
            {
                exposure.source_id.split(":", 1)[0]
                for exposure in exposures
                if exposure.source_type == "message" and ":" in exposure.source_id
            }
        )
    )
    related_event_ids = tuple(
        sorted(
            exposure.source_id
            for exposure in exposures
            if exposure.source_type == "event"
        )
    )
    return MemoryQuery(
        agent_id=agent_id,
        day=day,
        text=text or "No new information.",
        related_agent_ids=related_agent_ids,
        related_event_ids=related_event_ids,
        stance_hint=0.0,
        affected_interests=affected_interests,
    )


def retrieve_memories(
    memories: tuple[SocialMemory, ...],
    query: MemoryQuery,
    *,
    limit: int,
) -> tuple[RetrievedMemory, ...]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be positive")
    scored = [
        _score_memory(memory, query)
        for memory in memories
        if memory.agent_id == query.agent_id and memory.day <= query.day
    ]
    return tuple(sorted(scored, key=lambda item: (-item.score, item.memory.memory_id))[:limit])


def _score_memory(memory: SocialMemory, query: MemoryQuery) -> RetrievedMemory:
    recency_score = exp(-0.35 * max(0, query.day - memory.day))
    relevance_score = _relevance_score(memory, query)
    importance_score = memory.importance
    trust_score = memory.source_trust
    emotion_score = memory.emotional_intensity
    identity_score = memory.identity_relevance
    score = (
        0.20 * recency_score
        + 0.25 * relevance_score
        + 0.20 * importance_score
        + 0.10 * trust_score
        + 0.10 * emotion_score
        + 0.15 * identity_score
    )
    return RetrievedMemory(
        memory=memory,
        score=score,
        recency_score=recency_score,
        relevance_score=relevance_score,
        importance_score=importance_score,
        trust_score=trust_score,
        emotion_score=emotion_score,
        identity_score=identity_score,
    )


def _relevance_score(memory: SocialMemory, query: MemoryQuery) -> float:
    query_tokens = _tokens(query.text) | _tokens(" ".join(query.affected_interests))
    memory_tokens = _tokens(memory.text)
    token_score = (
        len(query_tokens & memory_tokens) / len(query_tokens | memory_tokens)
        if query_tokens and memory_tokens
        else 0.0
    )
    agent_score = 0.25 if set(memory.related_agent_ids) & set(query.related_agent_ids) else 0.0
    event_score = 0.35 if set(memory.related_event_ids) & set(query.related_event_ids) else 0.0
    return min(1.0, token_score + agent_score + event_score)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))
