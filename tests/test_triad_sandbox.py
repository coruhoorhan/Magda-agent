import pytest
from unittest.mock import MagicMock
from magda_agent.architecture.triad_sandbox import TriadSandbox


def test_triad_sandbox_successful_execution() -> None:
    """Test the triad sandbox when the evaluator approves the output."""
    mock_planner = MagicMock()
    mock_planner.plan.return_value = "Step 1, Step 2"

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Completed result"

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.return_value = True

    sandbox = TriadSandbox(
        planner=mock_planner,
        generator=mock_generator,
        evaluator=mock_evaluator
    )

    task_prompt = "Do the thing"
    result = sandbox.execute(task_prompt)

    # Verify calls
    mock_planner.plan.assert_called_once_with(task_prompt)
    mock_generator.generate.assert_called_once_with("Step 1, Step 2")
    mock_evaluator.evaluate.assert_called_once_with(task_prompt, "Step 1, Step 2", "Completed result")

    assert result == "Completed result"


def test_triad_sandbox_failed_evaluation() -> None:
    """Test the triad sandbox when the evaluator rejects the output."""
    mock_planner = MagicMock()
    mock_planner.plan.return_value = "Step A"

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Bad result"

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.return_value = False

    sandbox = TriadSandbox(
        planner=mock_planner,
        generator=mock_generator,
        evaluator=mock_evaluator
    )

    task_prompt = "Do task A"

    with pytest.raises(ValueError, match="Generator output failed Evaluator validation."):
        sandbox.execute(task_prompt)

    # Verify calls
    mock_planner.plan.assert_called_once_with(task_prompt)
    mock_generator.generate.assert_called_once_with("Step A")
    mock_evaluator.evaluate.assert_called_once_with(task_prompt, "Step A", "Bad result")
