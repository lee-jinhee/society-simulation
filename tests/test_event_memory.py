import pytest

from society_simulation.event_memory import (
    MemoryQuery,
    SocialMemory,
    build_memory_query,
    retrieve_memories,
)
from society_simulation.event_models import EventExposure


def memory(
    memory_id: str,
    *,
    day: int,
    text: str,
    importance: float = 0.5,
    source_trust: float = 0.5,
    emotional_intensity: float = 0.5,
    identity_relevance: float = 0.5,
) -> SocialMemory:
    return SocialMemory(
        memory_id=memory_id,
        agent_id="jisoo",
        day=day,
        kind="event_exposure",
        text=text,
        source_id=memory_id,
        source_type="event",
        channel="news_feed",
        related_agent_ids=(),
        related_event_ids=(memory_id,),
        stance_signal=0.2,
        emotional_intensity=emotional_intensity,
        source_trust=source_trust,
        identity_relevance=identity_relevance,
        importance=importance,
        private=False,
    )


def test_social_memory_validates_probability_fields() -> None:
    with pytest.raises(ValueError, match="importance must be a number between 0 and 1"):
        memory("bad", day=1, text="bad", importance=1.2)


def test_social_memory_validates_stance_signal() -> None:
    with pytest.raises(ValueError, match="stance_signal must be a number between -1 and 1"):
        SocialMemory(
            memory_id="bad",
            agent_id="jisoo",
            day=1,
            kind="event_exposure",
            text="bad",
            source_id="event",
            source_type="event",
            channel="news_feed",
            related_agent_ids=(),
            related_event_ids=(),
            stance_signal=1.2,
            emotional_intensity=0.2,
            source_trust=0.5,
            identity_relevance=0.5,
            importance=0.5,
            private=False,
        )


def test_retrieve_memories_returns_top_scored_memories_with_components() -> None:
    query = MemoryQuery(
        agent_id="jisoo",
        day=4,
        text="hospital commute asthma traffic",
        related_agent_ids=(),
        related_event_ids=("health_report",),
        stance_hint=0.4,
        affected_interests=("hospital access", "public health"),
    )
    memories = (
        memory(
            "health_report",
            day=3,
            text="public health report links traffic to asthma visits",
            importance=0.8,
            source_trust=0.9,
            emotional_intensity=0.4,
            identity_relevance=0.9,
        ),
        memory(
            "old_parking",
            day=1,
            text="downtown parking was expensive",
            importance=0.2,
            source_trust=0.4,
            emotional_intensity=0.2,
            identity_relevance=0.1,
        ),
    )

    retrieved = retrieve_memories(memories, query, limit=1)

    assert len(retrieved) == 1
    assert retrieved[0].memory.memory_id == "health_report"
    assert retrieved[0].score > 0.0
    assert retrieved[0].relevance_score > retrieved[0].recency_score / 2
    assert retrieved[0].to_dict()["memory"]["memory_id"] == "health_report"


def test_retrieve_memories_filters_to_agent_and_excludes_future_memories() -> None:
    query = MemoryQuery(
        agent_id="jisoo",
        day=2,
        text="traffic",
        related_agent_ids=(),
        related_event_ids=(),
        stance_hint=0.0,
        affected_interests=(),
    )
    other_agent = SocialMemory(
        memory_id="other",
        agent_id="minho",
        day=1,
        kind="event_exposure",
        text="traffic",
        source_id="event",
        source_type="event",
        channel="news_feed",
        related_agent_ids=(),
        related_event_ids=(),
        stance_signal=0.0,
        emotional_intensity=0.5,
        source_trust=0.5,
        identity_relevance=0.5,
        importance=0.5,
        private=False,
    )
    future = memory("future", day=3, text="traffic")

    assert retrieve_memories((other_agent, future), query, limit=5) == ()


def test_retrieved_memory_validates_score_components() -> None:
    retrieved = retrieve_memories(
        (memory("traffic", day=1, text="traffic"),),
        MemoryQuery(
            agent_id="jisoo",
            day=1,
            text="traffic",
            related_agent_ids=(),
            related_event_ids=(),
            stance_hint=0.0,
            affected_interests=(),
        ),
        limit=1,
    )[0]

    with pytest.raises(ValueError, match="score must be a number between 0 and 1"):
        type(retrieved)(
            memory=retrieved.memory,
            score=1.2,
            recency_score=retrieved.recency_score,
            relevance_score=retrieved.relevance_score,
            importance_score=retrieved.importance_score,
            trust_score=retrieved.trust_score,
            emotion_score=retrieved.emotion_score,
            identity_score=retrieved.identity_score,
        )


def test_build_memory_query_combines_exposure_text_and_metadata() -> None:
    exposure = EventExposure(
        day=2,
        agent_id="jisoo",
        source_type="event",
        source_id="public_health_report",
        channel="news_feed",
        content="Report links traffic to asthma near schools.",
    )

    query = build_memory_query(
        agent_id="jisoo",
        day=2,
        exposures=(exposure,),
        affected_interests=("public health",),
    )

    assert query.text == "Report links traffic to asthma near schools."
    assert query.related_event_ids == ("public_health_report",)
    assert query.affected_interests == ("public health",)
