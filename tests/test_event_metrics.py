import json

import pytest

from society_simulation.event_metrics import compute_event_metrics, compute_event_timeseries
from society_simulation.event_memory import SocialMemory
from society_simulation.event_models import EventAgentState, EventMessage


def state(
    agent_id: str,
    day: int,
    private: float,
    public: float,
    emotion: str = "calm",
    confidence: float = 0.5,
    salience: float = 0.6,
    willingness: float = 0.5,
    perceived_majority: float = 0.0,
    fairness: float = 0.3,
    official_trust: float = 0.5,
    speech_action: str = "read_only",
    silence_reason: str = "not_silent",
) -> EventAgentState:
    return EventAgentState(
        agent_id=agent_id,
        day=day,
        private_stance=private,
        public_stance=public,
        confidence=confidence,
        salience=salience,
        willingness_to_speak=willingness,
        perceived_majority=perceived_majority,
        fairness_concern=fairness,
        trust_in_official_info=official_trust,
        speech_action=speech_action,
        emotion=emotion,
        silence_reason=silence_reason,
        memory_summary="memory",
        last_private_reasoning="reason",
    )


def test_compute_event_timeseries_tracks_private_public_gap_and_messages() -> None:
    states_by_day = (
        (state("a", 0, -0.2, 0.0), state("b", 0, 0.2, 0.0)),
        (
            state(
                "a",
                1,
                -0.4,
                -0.1,
                "worried",
                confidence=0.7,
                salience=0.8,
                willingness=0.2,
                perceived_majority=0.4,
                fairness=0.9,
                official_trust=0.25,
                speech_action="read_only",
                silence_reason="I do not want to pile on publicly.",
            ),
            state(
                "b",
                1,
                0.3,
                0.1,
                "hopeful",
                confidence=0.9,
                salience=0.4,
                willingness=0.8,
                perceived_majority=0.2,
                fairness=0.4,
                official_trust=0.7,
                speech_action="public_post",
            ),
        ),
    )
    messages = (
        EventMessage(
            day=1,
            sender_agent_id="b",
            channel="chat",
            recipient_agent_id=None,
            text="text",
        ),
        EventMessage(
            day=1,
            sender_agent_id="b",
            channel="dm",
            recipient_agent_id="a",
            text="private text",
        ),
    )

    rows = compute_event_timeseries(states_by_day, messages)

    assert rows[0]["day"] == 0
    assert rows[1]["mean_private_stance"] == pytest.approx(-0.05)
    assert rows[1]["mean_public_stance"] == pytest.approx(0.0)
    assert rows[1]["mean_private_public_gap"] == pytest.approx(0.25)
    assert rows[1]["private_stance_variance"] == pytest.approx(0.1225)
    assert rows[1]["public_stance_variance"] == pytest.approx(0.01)
    assert rows[1]["mean_confidence"] == pytest.approx(0.8)
    assert rows[1]["mean_salience"] == pytest.approx(0.6)
    assert rows[1]["mean_willingness_to_speak"] == pytest.approx(0.5)
    assert rows[1]["silent_agent_count"] == 1
    assert rows[1]["silent_agent_rate"] == pytest.approx(0.5)
    assert rows[1]["mean_perceived_majority"] == pytest.approx(0.3)
    assert rows[1]["perceived_majority_error"] == pytest.approx(0.35)
    assert rows[1]["mean_fairness_concern"] == pytest.approx(0.65)
    assert rows[1]["mean_trust_in_official_info"] == pytest.approx(0.475)
    assert rows[1]["public_expression_bias"] == pytest.approx(0.05)
    assert rows[1]["speech_action_counts"] == {"public_post": 1, "read_only": 1}
    assert rows[1]["message_count"] == 2
    assert rows[1]["private_message_count"] == 1
    assert rows[1]["public_message_count"] == 1
    assert rows[1]["emotion_counts"] == {"hopeful": 1, "worried": 1}


def test_compute_event_metrics_uses_final_day() -> None:
    states_by_day = (
        (state("a", 0, 0.0, 0.0),),
        (
            state(
                "a",
                1,
                0.5,
                0.25,
                confidence=0.8,
                salience=0.4,
                willingness=0.3,
                perceived_majority=0.1,
                fairness=0.7,
                official_trust=0.2,
                speech_action="private_message",
                silence_reason="I would wait before posting.",
            ),
        ),
    )
    messages = (
        EventMessage(
            day=1,
            sender_agent_id="a",
            channel="chat",
            recipient_agent_id="b",
            text="text",
        ),
    )

    metrics = compute_event_metrics(states_by_day, messages=messages)

    assert metrics["agent_count"] == 1
    assert metrics["day_count"] == 2
    assert metrics["message_count"] == 1
    assert metrics["final_private_stance_mean"] == pytest.approx(0.5)
    assert metrics["final_public_stance_mean"] == pytest.approx(0.25)
    assert metrics["final_private_public_gap"] == pytest.approx(0.25)
    assert metrics["final_private_stance_variance"] == pytest.approx(0.0)
    assert metrics["final_public_stance_variance"] == pytest.approx(0.0)
    assert metrics["final_mean_confidence"] == pytest.approx(0.8)
    assert metrics["final_mean_salience"] == pytest.approx(0.4)
    assert metrics["final_mean_willingness_to_speak"] == pytest.approx(0.3)
    assert metrics["final_silent_agent_count"] == 1
    assert metrics["final_silent_agent_rate"] == pytest.approx(1.0)
    assert metrics["final_mean_perceived_majority"] == pytest.approx(0.1)
    assert metrics["final_perceived_majority_error"] == pytest.approx(0.4)
    assert metrics["final_mean_fairness_concern"] == pytest.approx(0.7)
    assert metrics["final_mean_trust_in_official_info"] == pytest.approx(0.2)
    assert metrics["final_public_expression_bias"] == pytest.approx(-0.25)
    assert metrics["final_speech_action_counts"] == {"private_message": 1}
    assert metrics["final_public_post_rate"] == pytest.approx(0.0)
    assert metrics["final_private_message_rate"] == pytest.approx(1.0)
    assert metrics["final_read_only_rate"] == pytest.approx(0.0)
    assert metrics["final_avoid_discussion_rate"] == pytest.approx(0.0)
    assert metrics["timeseries"][-1]["message_count"] == 1


def test_compute_event_metrics_includes_memory_summary() -> None:
    states_by_day = ((state("jisoo", 0, -0.1, 0.0),),)
    memories = (
        SocialMemory(
            memory_id="m1",
            agent_id="jisoo",
            day=0,
            kind="self_reasoning",
            text="private concern",
            source_id="decision",
            source_type="self",
            channel="internal",
            related_agent_ids=(),
            related_event_ids=(),
            stance_signal=-0.1,
            emotional_intensity=0.4,
            source_trust=1.0,
            identity_relevance=0.8,
            importance=0.7,
            private=True,
        ),
    )
    retrievals = (
        {
            "agent_id": "jisoo",
            "day": 0,
            "query": {},
            "retrieved": [{"score": 0.75, "memory": {"kind": "self_reasoning"}}],
        },
    )

    metrics = compute_event_metrics(states_by_day, (), memories=memories, retrievals=retrievals)

    assert metrics["memory_count"] == 1
    assert metrics["private_memory_count"] == 1
    assert metrics["public_memory_count"] == 0
    assert metrics["retrieval_count"] == 1
    assert metrics["mean_retrieved_memories_per_decision"] == pytest.approx(1.0)
    assert metrics["mean_retrieval_score"] == pytest.approx(0.75)


def test_compute_event_timeseries_rejects_mixed_day_bucket() -> None:
    states_by_day = (
        (
            state("a", 1, 0.1, 0.0),
            state("b", 2, 0.2, 0.1),
        ),
    )

    with pytest.raises(ValueError, match="states within a day bucket must share the same day"):
        compute_event_timeseries(states_by_day, messages=())


def test_compute_event_timeseries_rejects_unordered_day_buckets() -> None:
    states_by_day = (
        (state("a", 1, 0.1, 0.0),),
        (state("a", 0, 0.2, 0.1),),
    )

    with pytest.raises(ValueError, match="states_by_day must be ordered by day"):
        compute_event_timeseries(states_by_day, messages=())


def test_compute_event_timeseries_rejects_duplicate_day_buckets() -> None:
    states_by_day = (
        (state("a", 0, 0.1, 0.0),),
        (state("a", 0, 0.2, 0.1),),
    )

    with pytest.raises(ValueError, match="states_by_day must be ordered by day"):
        compute_event_timeseries(states_by_day, messages=())


def test_compute_event_metrics_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="states_by_day must not be empty"):
        compute_event_metrics((), messages=())


def test_compute_event_metrics_rejects_empty_day() -> None:
    with pytest.raises(ValueError, match="states_by_day must not contain empty days"):
        compute_event_metrics(((),), messages=())


def test_compute_event_metrics_returns_json_serializable_values() -> None:
    states_by_day = (
        (
            state("a", 0, -0.2, -0.1, "calm"),
            state("b", 0, 0.4, 0.2, "alert"),
        ),
    )
    messages = (
        EventMessage(
            day=0,
            sender_agent_id="a",
            channel="chat",
            recipient_agent_id="b",
            text="text",
        ),
    )

    metrics = compute_event_metrics(states_by_day, messages)

    json.dumps(metrics)
