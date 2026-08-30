import pytest
from unittest.mock import MagicMock

from magda_agent.learning.openclaw_rl_v6 import OpenClawRLV6

@pytest.fixture
def mock_habit_tracker() -> MagicMock:
    """Mocks the HabitTracker."""
    return MagicMock()

@pytest.fixture
def mock_mirror_neurons() -> MagicMock:
    """Mocks the MirrorNeurons."""
    return MagicMock()

@pytest.fixture
def learner(mock_habit_tracker: MagicMock, mock_mirror_neurons: MagicMock) -> OpenClawRLV6:
    """Provides a fresh OpenClawRLV6 instance."""
    return OpenClawRLV6(
        habit_tracker=mock_habit_tracker,
        mirror_neurons=mock_mirror_neurons
    )

def test_calculate_reward(learner: OpenClawRLV6) -> None:
    """Tests the reward calculation logic directly."""
    # Positive empathy shift, with tool output
    # base = (0.5 + 1.0) * 5.0 = 7.5
    # bonus = 2.0 (since p_shift > 0 and tool_output is given)
    # total = 9.5
    assert learner._calculate_reward(0.5, "Some output") == 9.5

    # Negative empathy shift, with tool output
    # base = (-0.5 + 1.0) * 5.0 = 2.5
    # no bonus since p_shift <= 0
    assert learner._calculate_reward(-0.5, "Some output") == 2.5

    # Capped at 10.0
    assert learner._calculate_reward(1.0, "output") == 10.0

    # Capped at 0.0
    assert learner._calculate_reward(-1.5, None) == 0.0

@pytest.mark.asyncio
async def test_process_positive_signal(
    learner: OpenClawRLV6,
    mock_habit_tracker: MagicMock,
    mock_mirror_neurons: MagicMock
) -> None:
    """Tests positive signal reinforces habits."""
    # Arrange
    user_id = 42
    user_reply = "This is very helpful!"
    action_context = "Executed search"
    tool_output = "Search results found"

    mock_mirror_neurons.empathize.return_value = (0.5, 0.1, 0.0) # p_shift = 0.5, a_shift = 0.1

    # Act
    await learner.process_next_state_signal(
        user_reply=user_reply,
        action_context=action_context,
        user_id=user_id,
        tool_output=tool_output,
        skills_used=["search_skill"]
    )

    # Assert
    mock_mirror_neurons.empathize.assert_called_once_with(f"{user_reply} [Tool Output: {tool_output}]")
    assert mock_habit_tracker.record_usage.call_count == 1

    # We don't test for EXACT 9.5 anymore due to learning rate scaling
    call_args = mock_habit_tracker.record_usage.call_args[1]
    assert call_args['input_text'] == action_context
    assert call_args['skill_used'] == "search_skill"
    assert call_args['evaluation_score'] > 9.0  # Just make sure it is scaled appropriately
    assert call_args['user_id'] == user_id

@pytest.mark.asyncio
async def test_process_negative_signal(
    learner: OpenClawRLV6,
    mock_habit_tracker: MagicMock,
    mock_mirror_neurons: MagicMock
) -> None:
    """Tests negative signal does not reinforce habits."""
    # Arrange
    user_id = 42
    user_reply = "That's wrong."
    action_context = "Executed search"
    tool_output = "Search results"

    mock_mirror_neurons.empathize.return_value = (-0.8, -0.2, 0.0) # p_shift = -0.8

    # Act
    await learner.process_next_state_signal(
        user_reply=user_reply,
        action_context=action_context,
        user_id=user_id,
        tool_output=tool_output
    )

    # Assert
    mock_mirror_neurons.empathize.assert_called_once_with(f"{user_reply} [Tool Output: {tool_output}]")
    mock_habit_tracker.record_usage.assert_not_called()

@pytest.mark.asyncio
async def test_empty_signals(
    learner: OpenClawRLV6,
    mock_habit_tracker: MagicMock,
    mock_mirror_neurons: MagicMock
) -> None:
    """Tests empty signals do not process anything."""
    # Act
    await learner.process_next_state_signal(
        user_reply="",
        action_context="Something",
        user_id=1
    )

    # Assert
    mock_mirror_neurons.empathize.assert_not_called()
    mock_habit_tracker.record_usage.assert_not_called()
