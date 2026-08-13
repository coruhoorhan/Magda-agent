import pytest
from magda_agent.learning.policy_gradient import RealtimePolicyGradientUpdater

def test_initial_weights_empty() -> None:
    """Test that initially there are no weights for a new user."""
    updater = RealtimePolicyGradientUpdater()
    weights = updater.get_dynamic_action_weights("user_1")
    assert weights == {}

def test_positive_emotion_shift_increases_weight() -> None:
    """Test that a positive pleasure shift increases the action weight."""
    updater = RealtimePolicyGradientUpdater()

    updater.process_mid_session_emotion_shift(
        user_id="user_1",
        action_taken="tell_joke",
        pleasure_shift=0.5,
        arousal_shift=0.0,
        dominance_shift=0.0
    )

    weights = updater.get_dynamic_action_weights("user_1")
    assert "tell_joke" in weights
    assert weights["tell_joke"] > 1.0
    assert abs(weights["tell_joke"] - 1.1) < 1e-6

def test_negative_emotion_shift_decreases_weight() -> None:
    """Test that a negative pleasure shift decreases the action weight."""
    updater = RealtimePolicyGradientUpdater()

    updater.process_mid_session_emotion_shift(
        user_id="user_2",
        action_taken="send_alert",
        pleasure_shift=-0.5,
        arousal_shift=0.0,
        dominance_shift=0.0
    )

    weights = updater.get_dynamic_action_weights("user_2")
    assert "send_alert" in weights
    assert weights["send_alert"] < 1.0
    assert abs(weights["send_alert"] - 0.9) < 1e-6

def test_arousal_magnifies_shift() -> None:
    """Test that high arousal magnifies the effect of the pleasure shift."""
    updater1 = RealtimePolicyGradientUpdater()
    updater2 = RealtimePolicyGradientUpdater()

    updater1.process_mid_session_emotion_shift(
        user_id="user_3",
        action_taken="dance",
        pleasure_shift=0.5,
        arousal_shift=0.0,
        dominance_shift=0.0
    )

    updater2.process_mid_session_emotion_shift(
        user_id="user_3",
        action_taken="dance",
        pleasure_shift=0.5,
        arousal_shift=1.0,
        dominance_shift=0.0
    )

    weights1 = updater1.get_dynamic_action_weights("user_3")
    weights2 = updater2.get_dynamic_action_weights("user_3")

    assert weights2["dance"] > weights1["dance"]
    assert abs(weights2["dance"] - 1.2) < 1e-6

def test_weight_bounding() -> None:
    """Test that weights remain within the 0.1 to 3.0 bounds."""
    updater = RealtimePolicyGradientUpdater()

    # Try to push below 0.1
    for _ in range(20):
        updater.process_mid_session_emotion_shift(
            user_id="user_4",
            action_taken="annoy",
            pleasure_shift=-1.0,
            arousal_shift=1.0,
            dominance_shift=0.0
        )

    weights = updater.get_dynamic_action_weights("user_4")
    assert weights["annoy"] == 0.1

    # Try to push above 3.0
    for _ in range(40):
        updater.process_mid_session_emotion_shift(
            user_id="user_5",
            action_taken="praise",
            pleasure_shift=1.0,
            arousal_shift=1.0,
            dominance_shift=0.0
        )

    weights_high = updater.get_dynamic_action_weights("user_5")
    assert weights_high["praise"] == 3.0
