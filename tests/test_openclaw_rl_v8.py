import pytest
from unittest.mock import MagicMock
from magda_agent.learning.openclaw_rl_v8 import RLSignalsProcessorV8

@pytest.fixture
def mock_mirror_neurons() -> MagicMock:
    """Provides a mocked MirrorNeurons instance."""
    return MagicMock()

@pytest.fixture
def processor(mock_mirror_neurons: MagicMock) -> RLSignalsProcessorV8:
    """Provides an instance of RLSignalsProcessorV8 with mocked dependencies."""
    return RLSignalsProcessorV8(mirror_neurons=mock_mirror_neurons)

def test_process_next_state_signals_positive(
    processor: RLSignalsProcessorV8, mock_mirror_neurons: MagicMock
) -> None:
    """Tests the correct mapping of a positive signal to a reward dictionary."""
    user_reply = "Good job"
    tool_output = "Done"

    # Mocking positive p_shift (0.5), a_shift (0.1), d_shift (0.0)
    mock_mirror_neurons.empathize.return_value = (0.5, 0.1, 0.0)

    result = processor.process_next_state_signals(user_reply, tool_output)

    mock_mirror_neurons.empathize.assert_called_once_with(f"{user_reply} {tool_output}")

    assert result["p_shift"] == 0.5
    assert result["a_shift"] == 0.1
    assert result["d_shift"] == 0.0
    # Reward = (0.5 + 1.0) * 5.0 = 7.5
    assert result["reward"] == 7.5

def test_process_next_state_signals_negative(
    processor: RLSignalsProcessorV8, mock_mirror_neurons: MagicMock
) -> None:
    """Tests the correct mapping of a negative signal to a reward dictionary."""
    user_reply = "Bad answer"
    tool_output = "Error"

    # Mocking negative p_shift (-0.5), a_shift (0.1), d_shift (0.2)
    mock_mirror_neurons.empathize.return_value = (-0.5, 0.1, 0.2)

    result = processor.process_next_state_signals(user_reply, tool_output)

    mock_mirror_neurons.empathize.assert_called_once_with(f"{user_reply} {tool_output}")

    assert result["p_shift"] == -0.5
    assert result["a_shift"] == 0.1
    assert result["d_shift"] == 0.2
    # Reward = (-0.5 + 1.0) * 5.0 = 2.5
    assert result["reward"] == 2.5
