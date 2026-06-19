import pytest

from society_simulation.event_metrics import compute_event_metrics, compute_event_timeseries
from society_simulation.event_models import EventAgentState, EventMessage


def state(
    agent_id: str,
    day: int,
    private: float,
    public: float,
    emotion: str = "calm",
) -> EventAgentState:
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
        EventMessage(
            day=1,
            sender_agent_id="a",
            channel="chat",
            recipient_agent_id=None,
            text="text",
        ),
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
