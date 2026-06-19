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
    previous_day: int | None = None
    for states in states_by_day:
        if not states:
            raise ValueError("states_by_day must not contain empty days")
        day = states[0].day
        if any(state.day != day for state in states[1:]):
            raise ValueError("states within a day bucket must share the same day")
        if previous_day is not None and day <= previous_day:
            raise ValueError("states_by_day must be ordered by day")
        previous_day = day
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
                "private_stance_variance": (
                    pvariance(private_values) if len(private_values) > 1 else 0.0
                ),
                "public_stance_variance": (
                    pvariance(public_values) if len(public_values) > 1 else 0.0
                ),
                "mean_confidence": mean(state.confidence for state in states),
                "mean_salience": mean(state.salience for state in states),
                "message_count": len(day_messages),
                "private_message_count": sum(
                    1 for message in day_messages if message.recipient_agent_id is not None
                ),
                "public_message_count": sum(
                    1 for message in day_messages if message.recipient_agent_id is None
                ),
                "emotion_counts": dict(sorted(Counter(state.emotion for state in states).items())),
            }
        )
    return rows


def compute_event_metrics(
    states_by_day: tuple[tuple[EventAgentState, ...], ...],
    messages: tuple[EventMessage, ...],
    *,
    memories: tuple[object, ...] = (),
    retrievals: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    if not states_by_day:
        raise ValueError("states_by_day must not be empty")
    rows = compute_event_timeseries(states_by_day, messages)
    final = rows[-1]
    retrieval_scores = tuple(
        float(item["score"])
        for row in retrievals
        for item in _retrieved_items(row)
        if isinstance(item.get("score"), (int, float))
    )
    return {
        "agent_count": len(states_by_day[-1]),
        "day_count": len(states_by_day),
        "message_count": len(messages),
        "memory_count": len(memories),
        "retrieval_count": len(retrievals),
        "private_memory_count": sum(1 for memory in memories if getattr(memory, "private", False)),
        "public_memory_count": sum(
            1 for memory in memories if not getattr(memory, "private", False)
        ),
        "mean_retrieved_memories_per_decision": (
            mean(len(_retrieved_items(row)) for row in retrievals) if retrievals else 0.0
        ),
        "mean_retrieval_score": mean(retrieval_scores) if retrieval_scores else 0.0,
        "retrieval_kind_counts": _retrieval_kind_counts(retrievals),
        "final_private_stance_mean": final["mean_private_stance"],
        "final_public_stance_mean": final["mean_public_stance"],
        "final_private_public_gap": final["mean_private_public_gap"],
        "final_private_stance_variance": final["private_stance_variance"],
        "final_public_stance_variance": final["public_stance_variance"],
        "final_mean_confidence": final["mean_confidence"],
        "final_mean_salience": final["mean_salience"],
        "timeseries": rows,
    }


def _retrieved_items(row: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    retrieved = row.get("retrieved", ())
    if not isinstance(retrieved, (list, tuple)):
        return ()
    return tuple(item for item in retrieved if isinstance(item, dict))


def _retrieval_kind_counts(retrievals: tuple[dict[str, Any], ...]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in retrievals:
        for item in _retrieved_items(row):
            memory = item.get("memory")
            if not isinstance(memory, dict):
                continue
            kind = memory.get("kind")
            if isinstance(kind, str):
                counts[kind] += 1
    return dict(sorted(counts.items()))
