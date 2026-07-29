import pytest
from magda_agent.learning.openclaw_rl_delayed_rewards import DelayedRewardTracker

def test_add_step():
    """Test that steps are correctly added to the trajectory."""
    tracker = DelayedRewardTracker()
    tracker.add_step("state1", "action1")
    tracker.add_step("state2", "action2")

    assert len(tracker.trajectory) == 2
    assert tracker.trajectory[0] == {"state": "state1", "action": "action1"}
    assert tracker.trajectory[1] == {"state": "state2", "action": "action2"}

def test_apply_reward_with_decay():
    """Test that delayed rewards are correctly backpropagated with exponential discounting."""
    tracker = DelayedRewardTracker()
    tracker.add_step("state1", "action1")
    tracker.add_step("state2", "action2")
    tracker.add_step("state3", "action3")

    final_reward = 1.0
    discount_factor = 0.9

    adjustments = tracker.apply_reward(final_reward, discount_factor)

    assert len(adjustments) == 3
    # The last step should receive the full reward
    assert adjustments[2]["reward"] == pytest.approx(1.0)
    # The second to last step should receive reward * discount_factor
    assert adjustments[1]["reward"] == pytest.approx(0.9)
    # The first step should receive reward * discount_factor^2
    assert adjustments[0]["reward"] == pytest.approx(0.81)

def test_empty_trajectory():
    """Test that applying rewards to an empty trajectory returns an empty list."""
    tracker = DelayedRewardTracker()
    adjustments = tracker.apply_reward(1.0)
    assert adjustments == []

def test_clear_trajectory():
    """Test that the trajectory can be cleared."""
    tracker = DelayedRewardTracker()
    tracker.add_step("state1", "action1")
    tracker.clear()

    assert len(tracker.trajectory) == 0
    assert tracker.apply_reward(1.0) == []
