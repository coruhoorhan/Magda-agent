import pytest
from unittest.mock import MagicMock
from magda_agent.learning.online_rl_loop_v7 import OnlineRLFeedbackLoopV7
from magda_agent.emotions.mirror_neurons import MirrorNeurons

@pytest.fixture
def mock_mirror_neurons():
    mock = MagicMock(spec=MirrorNeurons)
    # Default to neutral feedback
    mock.empathize.return_value = (0.0, 0.0, 0.0)
    return mock

@pytest.fixture
def rl_loop(mock_mirror_neurons):
    return OnlineRLFeedbackLoopV7(mirror_neurons=mock_mirror_neurons)

@pytest.mark.asyncio
async def test_adjust_behavior_positive(rl_loop, mock_mirror_neurons):
    """Test weight adjustments for positive feedback."""
    mock_mirror_neurons.empathize.return_value = (0.2, 0.1, 0.0)

    await rl_loop.adjust_behavior("Great job!", "Hello, how are you?")

    weights = rl_loop.get_weights()
    assert weights["verbosity"] == 1.05
    assert weights["empathy"] == 1.05
    assert weights["directness"] == 1.0

    assert len(rl_loop.trajectory_log) == 1
    assert rl_loop.trajectory_log[0]["reward"] == 0.2

@pytest.mark.asyncio
async def test_adjust_behavior_negative(rl_loop, mock_mirror_neurons):
    """Test weight adjustments for negative feedback."""
    mock_mirror_neurons.empathize.return_value = (-0.2, 0.1, 0.0)

    await rl_loop.adjust_behavior("This is terrible.", "Here is the result.")

    weights = rl_loop.get_weights()
    assert weights["verbosity"] == 0.95
    assert weights["directness"] == 1.05
    assert weights["empathy"] == 1.0

    assert len(rl_loop.trajectory_log) == 1
    assert rl_loop.trajectory_log[0]["reward"] == -0.2

@pytest.mark.asyncio
async def test_delayed_feedback_positive(rl_loop):
    """Test delayed positive feedback via registered interaction."""
    # Register an interaction
    interaction_id = rl_loop.register_interaction(
        state_context="Recommended link to user.",
        expected_action_type="click",
        user_id="user123"
    )

    assert interaction_id in rl_loop.pending_interactions

    # Apply delayed positive feedback
    await rl_loop.apply_delayed_feedback(interaction_id, reward=0.3)

    # Interaction should be removed
    assert interaction_id not in rl_loop.pending_interactions

    weights = rl_loop.get_weights()
    assert weights["verbosity"] == 1.05
    assert weights["empathy"] == 1.05

    assert len(rl_loop.trajectory_log) == 1
    assert rl_loop.trajectory_log[0]["reward"] == 0.3
    assert rl_loop.trajectory_log[0]["next_state"] == "Delayed Action: click"

@pytest.mark.asyncio
async def test_delayed_feedback_negative(rl_loop):
    """Test delayed negative feedback via registered interaction."""
    interaction_id = rl_loop.register_interaction(
        state_context="Shown ad.",
        expected_action_type="dismiss"
    )

    # Apply delayed negative feedback
    await rl_loop.apply_delayed_feedback(interaction_id, reward=-0.5)

    weights = rl_loop.get_weights()
    assert weights["verbosity"] == 0.95
    assert weights["directness"] == 1.05

@pytest.mark.asyncio
async def test_delayed_feedback_not_found(rl_loop):
    """Test applying feedback for an unknown interaction id does not error."""
    # Should not raise exception
    await rl_loop.apply_delayed_feedback("invalid-id", reward=0.5)

    # Weights should be unchanged
    weights = rl_loop.get_weights()
    assert weights["verbosity"] == 1.0
    assert weights["empathy"] == 1.0
    assert weights["directness"] == 1.0
