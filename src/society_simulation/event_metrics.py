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
        perceived_majority_values = [state.perceived_majority for state in states]
        gaps = [abs(state.private_stance - state.public_stance) for state in states]
        day_messages = [message for message in messages if message.day == day]
        public_senders = {
            message.sender_agent_id for message in day_messages if message.recipient_agent_id is None
        }
        silent_agent_count = sum(1 for state in states if state.agent_id not in public_senders)
        mean_private_stance = mean(private_values)
        mean_public_stance = mean(public_values)
        mean_perceived_majority = mean(perceived_majority_values)
        rows.append(
            {
                "day": day,
                "mean_private_stance": mean_private_stance,
                "mean_public_stance": mean_public_stance,
                "mean_private_public_gap": mean(gaps),
                "private_stance_variance": (
                    pvariance(private_values) if len(private_values) > 1 else 0.0
                ),
                "public_stance_variance": (
                    pvariance(public_values) if len(public_values) > 1 else 0.0
                ),
                "mean_confidence": mean(state.confidence for state in states),
                "mean_salience": mean(state.salience for state in states),
                "mean_willingness_to_speak": mean(
                    state.willingness_to_speak for state in states
                ),
                "silent_agent_count": silent_agent_count,
                "silent_agent_rate": silent_agent_count / len(states),
                "mean_perceived_majority": mean_perceived_majority,
                "perceived_majority_error": abs(mean_perceived_majority - mean_private_stance),
                "mean_fairness_concern": mean(state.fairness_concern for state in states),
                "mean_trust_in_official_info": mean(
                    state.trust_in_official_info for state in states
                ),
                "public_expression_bias": mean_public_stance - mean_private_stance,
                "speech_action_counts": dict(
                    sorted(Counter(state.speech_action for state in states).items())
                ),
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
        "final_mean_willingness_to_speak": final["mean_willingness_to_speak"],
        "final_silent_agent_count": final["silent_agent_count"],
        "final_silent_agent_rate": final["silent_agent_rate"],
        "final_mean_perceived_majority": final["mean_perceived_majority"],
        "final_perceived_majority_error": final["perceived_majority_error"],
        "final_mean_fairness_concern": final["mean_fairness_concern"],
        "final_mean_trust_in_official_info": final["mean_trust_in_official_info"],
        "final_public_expression_bias": final["public_expression_bias"],
        "final_speech_action_counts": final["speech_action_counts"],
        "final_public_post_rate": _speech_action_rate(final, "public_post"),
        "final_private_message_rate": _speech_action_rate(final, "private_message"),
        "final_read_only_rate": _speech_action_rate(final, "read_only"),
        "final_avoid_discussion_rate": _speech_action_rate(final, "avoid_discussion"),
        "timeseries": rows,
    }


def _retrieved_items(row: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    retrieved = row.get("retrieved", ())
    if not isinstance(retrieved, (list, tuple)):
        return ()
    return tuple(item for item in retrieved if isinstance(item, dict))


def _speech_action_rate(row: dict[str, Any], action: str) -> float:
    counts = row.get("speech_action_counts")
    if not isinstance(counts, dict):
        return 0.0
    total = sum(value for value in counts.values() if isinstance(value, int))
    if total == 0:
        return 0.0
    count = counts.get(action, 0)
    if not isinstance(count, int):
        return 0.0
    return count / total


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
