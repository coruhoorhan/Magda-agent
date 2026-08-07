"""
Tests for RL Metrics System V5
"""
import pytest
from magda_agent.learning.metrics_v5 import RLMetricsSystemV5

def test_log_signal() -> None:
    """
    Test logging a signal correctly pushes it to the store.
    """
    system = RLMetricsSystemV5()
    system.log_signal("agent_1", 12345.6, 0.95, {"action": "test"})
    metrics = system.get_metrics()

    assert len(metrics) == 1
    assert metrics[0]["agent_id"] == "agent_1"
    assert metrics[0]["timestamp"] == 12345.6
    assert metrics[0]["reward"] == 0.95
    assert metrics[0]["metadata"] == {"action": "test"}
