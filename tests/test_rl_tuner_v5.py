import pytest
from magda_agent.learning.rl_tuner_v5 import OnlineRLParameterTunerV5

def test_initial_learning_rate():
    tuner = OnlineRLParameterTunerV5(base_learning_rate=0.05)
    assert tuner.learning_rate == 0.05

def test_tune_learning_rate_high_uncertainty():
    tuner = OnlineRLParameterTunerV5(base_learning_rate=0.01)
    metrics = {"uncertainty": 0.8, "reward": 5.0}

    new_lr = tuner.tune_learning_rate(metrics)
    assert new_lr > 0.01  # Increased due to high uncertainty
    assert new_lr == pytest.approx(0.01 * 1.1)

def test_tune_learning_rate_high_reward():
    tuner = OnlineRLParameterTunerV5(base_learning_rate=0.01)
    metrics = {"uncertainty": 0.1, "reward": 9.0}

    new_lr = tuner.tune_learning_rate(metrics)
    assert new_lr < 0.01  # Decreased to stabilize because of high reward
    assert new_lr == pytest.approx(0.01 * 0.95)

def test_tune_learning_rate_low_reward():
    tuner = OnlineRLParameterTunerV5(base_learning_rate=0.01)
    metrics = {"uncertainty": 0.1, "reward": 2.0}

    new_lr = tuner.tune_learning_rate(metrics)
    assert new_lr > 0.01  # Increased to escape local minimum
    assert new_lr == pytest.approx(0.01 * 1.05)

def test_tune_learning_rate_boundaries():
    # Test max boundary
    tuner_max = OnlineRLParameterTunerV5(base_learning_rate=0.095)
    metrics_max = {"uncertainty": 0.9, "reward": 2.0}
    new_lr_max = tuner_max.tune_learning_rate(metrics_max)
    assert new_lr_max == 0.1 # Should be capped at max_lr (0.1)

    # Test min boundary
    tuner_min = OnlineRLParameterTunerV5(base_learning_rate=0.00105)
    metrics_min = {"uncertainty": 0.1, "reward": 9.0}
    new_lr_min = tuner_min.tune_learning_rate(metrics_min)
    assert new_lr_min == 0.001 # Should be capped at min_lr (0.001)

def test_tune_learning_rate_empty_metrics():
    tuner = OnlineRLParameterTunerV5(base_learning_rate=0.02)
    new_lr = tuner.tune_learning_rate({})
    assert new_lr == 0.02 # No change
