import pytest
import time
from unittest.mock import patch
from magda_agent.learning.habits_v4 import HabitTrackerV4


@pytest.fixture
def habit_tracker_v4(tmp_path):
    persist_dir = str(tmp_path / "test_habit_decay_v4_db")
    return HabitTrackerV4(persist_directory=persist_dir)


def test_habit_decay_with_momentum(habit_tracker_v4: HabitTrackerV4):
    """Test that explicit time decay works with momentum in V4."""
    current_time = time.time()
    one_day = 24 * 3600

    # Store records with mocked time (as if recorded 10 days ago)
    with patch("time.time", return_value=current_time - (10 * one_day)):
        habit_tracker_v4.record_usage("how do I deploy?", "deploy_skill", 9.0)
        habit_tracker_v4.record_usage("how do I deploy?", "deploy_skill", 9.0)
        habit_tracker_v4.record_usage("find files", "search_skill", 9.0)

    # Initial setup verification
    assert habit_tracker_v4.collection.count() == 3

    # All initial weights should be 1.0
    results = habit_tracker_v4.collection.get(include=["metadatas"])
    for meta in results["metadatas"]:
        assert meta["weight"] == 1.0

    # Test suggesting strategy before decay
    # Should suggest deploy_skill (momentum sum = 2.0)
    assert habit_tracker_v4.suggest_strategy("how do I deploy?") == "deploy_skill"

    # Now let's decay the habits by 10 days
    # Decay rate = 0.05 per day. Expected new weight = 1.0 - (0.05 * 10) = 0.5
    # Since 0.5 > min_weight (0.2), they should NOT be deleted, but weights should update
    with patch("time.time", return_value=current_time):
        decayed_count = habit_tracker_v4.decay_habits_with_momentum(decay_rate=0.05, min_weight=0.2)

    assert decayed_count == 0
    assert habit_tracker_v4.collection.count() == 3

    results = habit_tracker_v4.collection.get(include=["metadatas"])
    for meta in results["metadatas"]:
        assert 0.49 <= meta["weight"] <= 0.51  # Approx 0.5
        # The timestamp should have been updated to current_time
        assert meta["timestamp"] == current_time

    # Now, what if another 10 days pass?
    # New weight should be 0.5 - (0.05 * 10) = 0.0, which is < 0.2 (min_weight)
    # Thus, they should be deleted!
    future_time = current_time + (10 * one_day)
    with patch("time.time", return_value=future_time):
        decayed_count = habit_tracker_v4.decay_habits_with_momentum(decay_rate=0.05, min_weight=0.2)

    assert decayed_count == 3
    assert habit_tracker_v4.collection.count() == 0

def test_suggest_strategy_with_momentum(habit_tracker_v4: HabitTrackerV4):
    """Test that suggestion logic correctly factors in cumulative momentum."""

    # Let's say skill_A has 1 usage (weight 1.0), and skill_B has 3 usages but decayed a lot.
    # We can inject data manually or just use standard recording and time mocking.

    current_time = time.time()

    with patch("time.time", return_value=current_time):
        # A single usage of deploy_skill (weight 1.0)
        habit_tracker_v4.record_usage("how do I deploy?", "deploy_skill", 9.0)

        # A single usage of search_skill (weight 1.0)
        habit_tracker_v4.record_usage("how do I deploy?", "search_skill", 9.0)
        habit_tracker_v4.record_usage("how do I deploy?", "search_skill", 9.0)

    # Total momentum for search_skill = 2.0 (>= 1.5)
    assert habit_tracker_v4.suggest_strategy("how do I deploy?") == "search_skill"
