import pytest
from unittest.mock import MagicMock
from magda_agent.learning.skill_pruner_v2 import SkillPrunerV2

@pytest.fixture
def mock_registry() -> MagicMock:
    """Fixture providing a mock SkillRegistry populated with fake skills and telemetry data."""
    registry = MagicMock()
    # Populate mock skills and descriptions
    registry.skills = {
        "good_skill": lambda x: x,
        "bad_skill": lambda x: x,
        "new_skill": lambda x: x,
        "unused_skill": lambda x: x
    }
    registry.descriptions = {
        "good_skill": "A good skill",
        "bad_skill": "A bad skill",
        "new_skill": "A new skill with few calls",
        "unused_skill": "A skill with zero calls"
    }

    # Setup telemetry tracker mock
    tracker = MagicMock()

    # Mock get_aggregated_metrics to return predefined test data
    def mock_get_metrics():
        return {
            "good_skill": {"total_calls": 10, "success_rate": 0.9, "total_execution_time_ms": 100},
            "bad_skill": {"total_calls": 6, "success_rate": 0.3, "total_execution_time_ms": 100},
            "new_skill": {"total_calls": 2, "success_rate": 0.0, "total_execution_time_ms": 100},
            "unused_skill": {"total_calls": 0, "success_rate": 0.0, "total_execution_time_ms": 0}
        }

    tracker.get_aggregated_metrics.side_effect = mock_get_metrics
    registry.telemetry_tracker = tracker

    return registry

def test_skill_pruner_high_failure(mock_registry: MagicMock) -> None:
    """Test that skills with high failure rates and zero usage are pruned when prune_zero_usage is True."""
    pruner = SkillPrunerV2(mock_registry)
    pruner.evaluate_and_prune(min_calls=5, failure_threshold=0.5, prune_zero_usage=True)

    assert "good_skill" in mock_registry.skills
    assert "good_skill" in mock_registry.descriptions

    assert "new_skill" in mock_registry.skills
    assert "new_skill" in mock_registry.descriptions

    assert "bad_skill" not in mock_registry.skills
    assert "bad_skill" not in mock_registry.descriptions

    assert "unused_skill" not in mock_registry.skills
    assert "unused_skill" not in mock_registry.descriptions

def test_skill_pruner_keep_zero_usage(mock_registry: MagicMock) -> None:
    """Test that unused skills are kept when prune_zero_usage is False."""
    pruner = SkillPrunerV2(mock_registry)
    pruner.evaluate_and_prune(min_calls=5, failure_threshold=0.5, prune_zero_usage=False)

    assert "unused_skill" in mock_registry.skills
    assert "unused_skill" in mock_registry.descriptions

    assert "bad_skill" not in mock_registry.skills
    assert "bad_skill" not in mock_registry.descriptions

def test_skill_pruner_no_telemetry_tracker() -> None:
    """Test that pruning is skipped safely if the registry lacks a telemetry tracker."""
    registry = MagicMock()
    del registry.telemetry_tracker
    registry.skills = {"skill1": lambda x: x}

    pruner = SkillPrunerV2(registry)
    pruner.evaluate_and_prune()

    assert "skill1" in registry.skills
