import pytest
from unittest.mock import MagicMock
from magda_agent.learning.online_rl_loop_v6 import OnlineRLFeedbackLoopV6

@pytest.fixture
def mock_mirror_neurons():
    mock = MagicMock()
    # Default to neutral feedback
    mock.empathize.return_value = (0.0, 0.0, 0.0)
    return mock

@pytest.fixture
def rl_loop(mock_mirror_neurons):
    return OnlineRLFeedbackLoopV6(mirror_neurons=mock_mirror_neurons)

@pytest.mark.asyncio
async def test_initialization(rl_loop):
    weights = rl_loop.get_weights()
    assert weights["verbosity"] == 1.0
    assert weights["directness"] == 1.0
    assert weights["empathy"] == 1.0
    assert len(rl_loop.trajectory_log) == 0

@pytest.mark.asyncio
async def test_positive_feedback_adjusts_weights(rl_loop, mock_mirror_neurons):
    # Mock a positive sentiment shift (P-shift > 0.1)
    mock_mirror_neurons.empathize.return_value = (0.5, 0.0, 0.0)

    await rl_loop.adjust_behavior("This is great!", "context_xyz", "user_1")

    weights = rl_loop.get_weights()
    assert weights["verbosity"] > 1.0
    assert weights["empathy"] > 1.0
    assert weights["directness"] == 1.0

    assert len(rl_loop.trajectory_log) == 1
    assert rl_loop.trajectory_log[0]["reward"] == 0.5
    assert rl_loop.trajectory_log[0]["next_state"] == "This is great!"

@pytest.mark.asyncio
async def test_negative_feedback_adjusts_weights(rl_loop, mock_mirror_neurons):
    # Mock a negative sentiment shift (P-shift < -0.1)
    mock_mirror_neurons.empathize.return_value = (-0.5, 0.0, 0.0)

    await rl_loop.adjust_behavior("This is terrible.", "context_abc", "user_1")

    weights = rl_loop.get_weights()
    assert weights["verbosity"] < 1.0
    assert weights["directness"] > 1.0
    assert weights["empathy"] == 1.0

    assert len(rl_loop.trajectory_log) == 1
    assert rl_loop.trajectory_log[0]["reward"] == -0.5

@pytest.mark.asyncio
async def test_neutral_feedback_does_not_adjust_weights(rl_loop, mock_mirror_neurons):
    # Mock a neutral sentiment shift
    mock_mirror_neurons.empathize.return_value = (0.05, 0.0, 0.0)

    await rl_loop.adjust_behavior("Okay.", "context_123", "user_1")

    weights = rl_loop.get_weights()
    assert weights["verbosity"] == 1.0
    assert weights["directness"] == 1.0
    assert weights["empathy"] == 1.0

    assert len(rl_loop.trajectory_log) == 1
    assert rl_loop.trajectory_log[0]["reward"] == 0.05

@pytest.mark.asyncio
async def test_bounds_on_weights(rl_loop, mock_mirror_neurons):
    # Force max weight
    mock_mirror_neurons.empathize.return_value = (0.5, 0.0, 0.0)

    # Run multiple times to exceed bounds
    for _ in range(50):
        await rl_loop.adjust_behavior("Great!", "context", "user_1")

    weights = rl_loop.get_weights()
    assert weights["verbosity"] == 2.0
    assert weights["empathy"] == 2.0

    # Force min weight
    mock_mirror_neurons.empathize.return_value = (-0.5, 0.0, 0.0)

    # Run multiple times to exceed bounds
    for _ in range(50):
        await rl_loop.adjust_behavior("Bad!", "context", "user_1")

    weights = rl_loop.get_weights()
    assert weights["verbosity"] == 0.5
    assert weights["directness"] == 2.0
