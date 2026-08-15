import pytest
from unittest.mock import MagicMock
from magda_agent.learning.habits import HabitTracker
from magda_agent.learning.interactive_rl_v6 import InteractiveRLLoopV6

@pytest.fixture
def mock_habit_tracker():
    tracker = MagicMock(spec=HabitTracker)
    return tracker

@pytest.fixture
def interactive_rl(mock_habit_tracker):
    return InteractiveRLLoopV6(habit_tracker=mock_habit_tracker)

def test_analyze_feedback_positive(interactive_rl):
    """Test that positive replies yield a high score."""
    assert interactive_rl.analyze_feedback("Thanks Magda!") == 9.0
    assert interactive_rl.analyze_feedback("that's perfect, yes") == 9.0
    assert interactive_rl.analyze_feedback("Great job") == 9.0

def test_analyze_feedback_negative(interactive_rl):
    """Test that negative replies yield a low score."""
    assert interactive_rl.analyze_feedback("no that's wrong") == 2.0
    assert interactive_rl.analyze_feedback("bad output") == 2.0
    assert interactive_rl.analyze_feedback("stop failing") == 2.0

def test_analyze_feedback_neutral(interactive_rl):
    """Test that neutral or ambiguous replies yield a default neutral score."""
    assert interactive_rl.analyze_feedback("I see") == 5.0
    assert interactive_rl.analyze_feedback("okay") == 5.0
    assert interactive_rl.analyze_feedback("hmm") == 5.0

def test_process_interaction_records_usage(interactive_rl, mock_habit_tracker):
    """Test that process_interaction correctly delegates to habit_tracker."""
    interactive_rl.process_interaction(
        input_text="What's the weather like?",
        skill_used="weather_skill",
        user_reply="thanks!",
        user_id=123
    )

    # analyze_feedback("thanks!") should return 9.0
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="What's the weather like?",
        skill_used="weather_skill",
        evaluation_score=9.0,
        user_id=123
    )

def test_process_interaction_negative(interactive_rl, mock_habit_tracker):
    """Test process_interaction with a negative reply."""
    interactive_rl.process_interaction(
        input_text="Set a timer for 10 minutes.",
        skill_used="timer_skill",
        user_reply="no, that's incorrect.",
        user_id=456
    )

    # analyze_feedback("no, that's incorrect.") should return 2.0
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="Set a timer for 10 minutes.",
        skill_used="timer_skill",
        evaluation_score=2.0,
        user_id=456
    )
