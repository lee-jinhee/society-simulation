import random

from society_simulation.signals import BinarySignalModel


def test_signal_generation_is_deterministic_for_seed() -> None:
    first = BinarySignalModel(signal_accuracy=0.7, rng=random.Random(123))
    second = BinarySignalModel(signal_accuracy=0.7, rng=random.Random(123))

    assert first.sample_true_state() == second.sample_true_state()
    assert first.generate_private_signals("A", 10) == second.generate_private_signals("A", 10)


def test_signal_accuracy_is_close_in_large_sample() -> None:
    model = BinarySignalModel(signal_accuracy=0.75, rng=random.Random(4))

    signals = model.generate_private_signals("A", 2000)
    match_rate = signals.count("A") / len(signals)

    assert 0.72 <= match_rate <= 0.78
