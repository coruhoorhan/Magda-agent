import pytest
import time
from unittest.mock import patch
from magda_agent.learning.habits import HabitTracker


@pytest.fixture
def habit_tracker(tmp_path):
    persist_dir = str(tmp_path / "test_habit_decay_db")
    return HabitTracker(persist_directory=persist_dir)


def test_habit_decay(habit_tracker: HabitTracker):
    """Test that older habits decay appropriately."""
    current_time = time.time()
    one_day = 24 * 3600

    # Store records with mocked time
    with patch("time.time", return_value=current_time - (40 * one_day)):
        # 40 days old (should decay)
        habit_tracker.record_usage("how do I deploy?", "deploy_skill", 9.0)
        habit_tracker.record_usage("how do I deploy?", "deploy_skill", 9.0)

    with patch("time.time", return_value=current_time - (10 * one_day)):
        # 10 days old (should not decay)
        habit_tracker.record_usage("find files", "search_skill", 9.0)
        habit_tracker.record_usage("find files", "search_skill", 9.0)

    # Initially we have 4 records in the collection
    assert habit_tracker.collection.count() == 4

    # Decay habits older than 30 days
    # The first two records (deploy_skill) are 40 days old, they should be deleted.
    with patch("time.time", return_value=current_time):
        decayed_count = habit_tracker.decay_habits(days=30.0)

    assert decayed_count == 2
    assert habit_tracker.collection.count() == 2

    # Let's verify suggest_strategy
    # The old habit (deploy_skill) has been decayed and shouldn't be suggested
    # Note: ChromaDB might return empty or not depending on what's left. Let's see what suggest_strategy returns.
    # We didn't leave any 'deploy_skill' records, so suggestion should fail to find 2 occurrences.
    assert habit_tracker.suggest_strategy("how do I deploy?") is None

    # The new habit (search_skill) should still be suggested
    assert habit_tracker.suggest_strategy("find files") == "search_skill"

    # Decay remaining records (using days=5)
    with patch("time.time", return_value=current_time):
        decayed_count = habit_tracker.decay_habits(days=5.0)

    assert decayed_count == 2
    assert habit_tracker.collection.count() == 0

    assert habit_tracker.suggest_strategy("find files") is None
