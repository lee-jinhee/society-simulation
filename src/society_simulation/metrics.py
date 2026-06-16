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
