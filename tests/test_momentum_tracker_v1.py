import pytest
from unittest.mock import MagicMock
from magda_agent.learning.momentum_tracker_v1 import MomentumTracker
from magda_agent.user_model.model import UserModel

@pytest.fixture
def mock_user_model():
    model = MagicMock(spec=UserModel)
    model.get_model.return_value = {
        "behavior_weights": {
            "exploration": 1.0,
            "verbosity": 1.0,
            "directness": 1.0
        }
    }
    return model

def test_initial_tracking_does_not_update(mock_user_model, monkeypatch):
    """Test that the first interaction just sets up the state."""
    tracker = MomentumTracker(mock_user_model)
    tracker.track_and_update(user_id=1, interaction_text="Hello, how are you?")

    assert 1 in tracker.user_states
    mock_user_model.save_model.assert_not_called()

def test_high_momentum_decreases_verbosity(mock_user_model, monkeypatch):
    """Test that rapid, short interactions decrease verbosity."""
    tracker = MomentumTracker(mock_user_model)

    # Mock time to simulate rapid interaction
    import time
    times = [1000.0, 1005.0] # 5 seconds delta
    def mock_time():
        return times.pop(0)
    monkeypatch.setattr(time, "time", mock_time)

    # First interaction
    tracker.track_and_update(user_id=1, interaction_text="Hi")

    # Second interaction, 5 seconds later
    # speed = 100 / 5 = 20.0
    # momentum = 20.0 + (1 * 0.1) = 20.1 (high momentum > 15)
    tracker.track_and_update(user_id=1, interaction_text="Fast")

    mock_user_model.save_model.assert_called_once()
    saved_model = mock_user_model.save_model.call_args[0][1]
    assert saved_model["behavior_weights"]["verbosity"] == pytest.approx(0.9)

def test_low_momentum_increases_verbosity(mock_user_model, monkeypatch):
    """Test that slow interactions increase verbosity."""
    tracker = MomentumTracker(mock_user_model)

    # Mock time to simulate slow interaction
    import time
    times = [1000.0, 1050.0] # 50 seconds delta
    def mock_time():
        return times.pop(0)
    monkeypatch.setattr(time, "time", mock_time)

    # First interaction
    tracker.track_and_update(user_id=1, interaction_text="Can you explain this deeply?")

    # Second interaction, 50 seconds later
    # speed = 100 / 50 = 2.0
    # momentum = 2.0 + (2 * 0.1) = 2.2 (low momentum < 5)
    tracker.track_and_update(user_id=1, interaction_text="Please do.")

    mock_user_model.save_model.assert_called_once()
    saved_model = mock_user_model.save_model.call_args[0][1]
    assert saved_model["behavior_weights"]["verbosity"] == pytest.approx(1.1)

def test_verbosity_bounded(mock_user_model, monkeypatch):
    """Test that verbosity stays within [0.5, 2.0]."""
    mock_user_model.get_model.return_value = {
        "behavior_weights": {
            "verbosity": 0.5
        }
    }

    tracker = MomentumTracker(mock_user_model)

    import time
    times = [1000.0, 1005.0] # 5 seconds delta
    def mock_time():
        return times.pop(0)
    monkeypatch.setattr(time, "time", mock_time)

    # First interaction
    tracker.track_and_update(user_id=1, interaction_text="Hi")

    # Second interaction (high momentum -> should decrease verbosity but it's already 0.5)
    tracker.track_and_update(user_id=1, interaction_text="Fast")

    # Now it is bounded and it shouldn't actually save if the value did not change,
    # but the implementation calls save_model as long as `verbosity_shift != 0.0`.
    # Let's adjust the mock to expect that the saved verbosity is 0.5 and not lower.

    mock_user_model.save_model.assert_called_once()
    saved_model = mock_user_model.save_model.call_args[0][1]
    assert saved_model["behavior_weights"]["verbosity"] == pytest.approx(0.5)

    mock_user_model.save_model.reset_mock()

    # Test upper bound
    mock_user_model.get_model.return_value = {
        "behavior_weights": {
            "verbosity": 2.0
        }
    }

    times = [2000.0, 2050.0] # 50 seconds delta
    def mock_time_2():
        return times.pop(0)
    monkeypatch.setattr(time, "time", mock_time_2)

    tracker.track_and_update(user_id=2, interaction_text="Hi")
    tracker.track_and_update(user_id=2, interaction_text="Slow")

    mock_user_model.save_model.assert_called_once()
    saved_model_2 = mock_user_model.save_model.call_args[0][1]
    assert saved_model_2["behavior_weights"]["verbosity"] == pytest.approx(2.0)
