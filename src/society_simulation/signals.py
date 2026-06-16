from __future__ import annotations

import random

from society_simulation.models import Action


class BinarySignalModel:
    def __init__(self, signal_accuracy: float, rng: random.Random) -> None:
        if not 0.5 <= signal_accuracy <= 1.0:
            raise ValueError("signal_accuracy must be between 0.5 and 1.0")
        self.signal_accuracy = signal_accuracy
        self.rng = rng

    def sample_true_state(self) -> Action:
        return "A" if self.rng.random() < 0.5 else "B"

    def sample_private_signal(self, true_state: Action) -> Action:
        if true_state not in ("A", "B"):
            raise ValueError("true_state must be A or B")
        if self.rng.random() < self.signal_accuracy:
            return true_state
        return "B" if true_state == "A" else "A"

    def generate_private_signals(self, true_state: Action, num_agents: int) -> list[Action]:
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        return [self.sample_private_signal(true_state) for _ in range(num_agents)]
