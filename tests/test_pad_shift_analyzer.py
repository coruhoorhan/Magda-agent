import pytest
from unittest.mock import MagicMock
from typing import Dict, Optional

from magda_agent.emotions.mirror_neurons import MirrorNeurons
from magda_agent.learning.pad_shift_analyzer import PADShiftAnalyzer

@pytest.fixture
def mock_mirror_neurons() -> MagicMock:
    """Provides a mocked MirrorNeurons instance."""
    return MagicMock(spec=MirrorNeurons)

@pytest.fixture
def analyzer(mock_mirror_neurons: MagicMock) -> PADShiftAnalyzer:
    """Provides a PADShiftAnalyzer with a mocked MirrorNeurons dependency."""
    return PADShiftAnalyzer(mirror_neurons=mock_mirror_neurons)

def test_analyze_interaction_without_tool_output(analyzer: PADShiftAnalyzer, mock_mirror_neurons: MagicMock) -> None:
    """Tests analyzing a single interaction without tool output."""
    user_reply = "I am happy"
    mock_mirror_neurons.empathize.return_value = (0.2, 0.1, 0.0)

    result = analyzer.analyze_interaction(user_reply)

    mock_mirror_neurons.empathize.assert_called_once_with(user_reply)
    assert result == {"p_shift": 0.2, "a_shift": 0.1, "d_shift": 0.0}

def test_analyze_interaction_with_tool_output(analyzer: PADShiftAnalyzer, mock_mirror_neurons: MagicMock) -> None:
    """Tests analyzing a single interaction with tool output."""
    user_reply = "I am sad"
    tool_output = "No results found"
    mock_mirror_neurons.empathize.return_value = (-0.2, 0.2, 0.1)

    result = analyzer.analyze_interaction(user_reply, tool_output)

    expected_text = f"{user_reply} [Tool Output: {tool_output}]"
    mock_mirror_neurons.empathize.assert_called_once_with(expected_text)
    assert result == {"p_shift": -0.2, "a_shift": 0.2, "d_shift": 0.1}

def test_aggregate_shifts_empty(analyzer: PADShiftAnalyzer) -> None:
    """Tests aggregating empty interaction logs returns zeros."""
    result = analyzer.aggregate_shifts([])
    assert result["total_p_shift"] == 0.0
    assert result["adaptability_score"] == 0.0

def test_aggregate_shifts(analyzer: PADShiftAnalyzer, mock_mirror_neurons: MagicMock) -> None:
    """Tests aggregating a list of interaction logs."""
    # Setup mock to return different values for different calls
    mock_mirror_neurons.empathize.side_effect = [
        (0.2, 0.1, 0.0),   # Call 1
        (-0.4, 0.3, 0.1),  # Call 2
    ]

    logs: list[Dict[str, Optional[str]]] = [
        {"user_reply": "Good job", "tool_output": "Success"},
        {"user_reply": "Bad job", "tool_output": None}
    ]

    result = analyzer.aggregate_shifts(logs)

    assert mock_mirror_neurons.empathize.call_count == 2

    # Total calculations
    # p: 0.2 - 0.4 = -0.2
    # a: 0.1 + 0.3 = 0.4
    # d: 0.0 + 0.1 = 0.1
    assert result["total_p_shift"] == pytest.approx(-0.2)
    assert result["total_a_shift"] == pytest.approx(0.4)
    assert result["total_d_shift"] == pytest.approx(0.1)

    # Averages
    assert result["avg_p_shift"] == pytest.approx(-0.1)
    assert result["avg_a_shift"] == pytest.approx(0.2)
    assert result["avg_d_shift"] == pytest.approx(0.05)

    # Adaptability Score = abs(-0.1) + abs(0.2) + abs(0.05) = 0.35
    assert result["adaptability_score"] == pytest.approx(0.35)
