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
        observed_actions=(),
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
