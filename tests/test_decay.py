import pytest
import math
from magda_agent.learning.decay import (
    decay_weight_exponential,
    decay_weight_step,
    decay_skill_weights,
    SkillWeightDecayer,
)


def test_decay_weight_exponential_above_baseline():
    # Weight starts at 2.0, decays towards 1.0 with rate 0.5 over 2.0 seconds
    # Expected: 1.0 + (2.0 - 1.0) * exp(-0.5 * 2.0) = 1.0 + 1.0 * exp(-1) = 1.0 + 1.0 / e
    weight = 2.0
    elapsed_time = 2.0
    decay_rate = 0.5
    baseline = 1.0

    result = decay_weight_exponential(weight, elapsed_time, decay_rate, baseline)
    expected = baseline + (weight - baseline) * math.exp(-decay_rate * elapsed_time)
    assert pytest.approx(result) == expected
    assert result > 1.0
    assert result < 2.0


def test_decay_weight_exponential_below_baseline():
    # Weight starts at 0.5, decays towards 1.0 with rate 0.5 over 2.0 seconds
    # Expected: 1.0 + (0.5 - 1.0) * exp(-1) = 1.0 - 0.5 / e
    weight = 0.5
    elapsed_time = 2.0
    decay_rate = 0.5
    baseline = 1.0

    result = decay_weight_exponential(weight, elapsed_time, decay_rate, baseline)
    expected = baseline + (weight - baseline) * math.exp(-decay_rate * elapsed_time)
    assert pytest.approx(result) == expected
    assert result > 0.5
    assert result < 1.0


def test_decay_weight_exponential_edge_cases():
    # Negative elapsed time -> should return original weight
    assert decay_weight_exponential(2.0, -1.0, 0.5, 1.0) == 2.0
    # Zero elapsed time -> should return original weight
    assert decay_weight_exponential(2.0, 0.0, 0.5, 1.0) == 2.0
    # Zero decay rate -> should return original weight
    assert decay_weight_exponential(2.0, 5.0, 0.0, 1.0) == 2.0
    # Weight already at baseline -> should return baseline
    assert decay_weight_exponential(1.0, 5.0, 0.5, 1.0) == 1.0


def test_decay_weight_step_above_baseline():
    # Weight starts at 2.0, decays towards 1.0 with step rate 0.1 over 3 steps
    # Expected: 1.0 + (2.0 - 1.0) * (1 - 0.1) ** 3 = 1.0 + 1.0 * 0.9 ** 3 = 1.0 + 0.729 = 1.729
    weight = 2.0
    steps = 3
    decay_rate = 0.1
    baseline = 1.0

    result = decay_weight_step(weight, steps, decay_rate, baseline)
    assert pytest.approx(result) == 1.729


def test_decay_weight_step_below_baseline():
    # Weight starts at 0.2, decays towards 1.0 with step rate 0.2 over 2 steps
    # Expected: 1.0 + (0.2 - 1.0) * (1 - 0.2) ** 2 = 1.0 - 0.8 * 0.64 = 1.0 - 0.512 = 0.488
    weight = 0.2
    steps = 2
    decay_rate = 0.2
    baseline = 1.0

    result = decay_weight_step(weight, steps, decay_rate, baseline)
    assert pytest.approx(result) == 0.488


def test_decay_weight_step_edge_cases():
    # Negative steps -> should return original weight
    assert decay_weight_step(2.0, -5, 0.1, 1.0) == 2.0
    # Zero steps -> should return original weight
    assert decay_weight_step(2.0, 0, 0.1, 1.0) == 2.0
    # Zero decay rate -> should return original weight
    assert decay_weight_step(2.0, 3, 0.0, 1.0) == 2.0


def test_decay_skill_weights():
    skill_weights = {"skill_a": 1.8, "skill_b": 0.4, "skill_c": 1.0}
    last_updated = {"skill_a": 1000.0, "skill_b": 1050.0}  # skill_c has no last updated timestamp
    current_time = 1100.0
    decay_rate = 0.01
    baseline = 1.0

    result = decay_skill_weights(
        skill_weights, last_updated, current_time, decay_rate, baseline
    )

    # For skill_a: elapsed = 100.0, expected = 1.0 + 0.8 * exp(-0.01 * 100) = 1.0 + 0.8 / e
    expected_a = 1.0 + 0.8 * math.exp(-1.0)
    assert pytest.approx(result["skill_a"]) == expected_a

    # For skill_b: elapsed = 50.0, expected = 1.0 + (-0.6) * exp(-0.01 * 50) = 1.0 - 0.6 * exp(-0.5)
    expected_b = 1.0 - 0.6 * math.exp(-0.5)
    assert pytest.approx(result["skill_b"]) == expected_b

    # For skill_c: no timestamp, should return original weight
    assert result["skill_c"] == 1.0


def test_skill_weight_decayer_class_time():
    decayer = SkillWeightDecayer(decay_rate=0.02, baseline=1.0)
    skill_weights = {"skill_a": 1.5}
    last_updated = {"skill_a": 100.0}
    current_time = 150.0

    # Elapsed = 50.0. Expected: 1.0 + 0.5 * exp(-0.02 * 50) = 1.0 + 0.5 * exp(-1.0) = 1.0 + 0.5 / e
    result = decayer.decay_by_time(skill_weights, last_updated, current_time)
    expected = 1.0 + 0.5 * math.exp(-1.0)
    assert pytest.approx(result["skill_a"]) == expected


def test_skill_weight_decayer_class_steps():
    decayer = SkillWeightDecayer(decay_rate=0.1, baseline=1.0)
    skill_weights = {"skill_a": 1.5, "skill_b": 0.5}

    result = decayer.decay_by_steps(skill_weights, steps=2)

    # For skill_a: 1.0 + 0.5 * (0.9) ** 2 = 1.0 + 0.5 * 0.81 = 1.405
    assert pytest.approx(result["skill_a"]) == 1.405
    # For skill_b: 1.0 - 0.5 * (0.9) ** 2 = 1.0 - 0.5 * 0.81 = 0.595
    assert pytest.approx(result["skill_b"]) == 0.595
