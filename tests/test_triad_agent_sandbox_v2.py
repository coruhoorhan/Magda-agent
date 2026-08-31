import pytest
from unittest.mock import MagicMock
from magda_agent.architecture.triad_agent_sandbox_v2 import TriadAgentSandboxV2


def test_triad_agent_sandbox_v2_successful_execution() -> None:
    """Test the triad sandbox v2 when the evaluator approves the output."""
    mock_planner = MagicMock()
    mock_planner.plan.return_value = "Step 1, Step 2"

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Completed result"

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.return_value = True

    sandbox = TriadAgentSandboxV2(
        planner=mock_planner,
        generator=mock_generator,
        evaluator=mock_evaluator
    )

    # Pre-existing state
    sandbox.set_state("initial", "state")

    task_prompt = "Do the thing"
    result = sandbox.execute(task_prompt)

    # Verify calls
    mock_planner.plan.assert_called_once_with(task_prompt)
    mock_generator.generate.assert_called_once_with("Step 1, Step 2")
    mock_evaluator.evaluate.assert_called_once_with(task_prompt, "Step 1, Step 2", "Completed result")

    assert result == "Completed result"

    # State should contain the latest execution details
    assert sandbox.get_state("initial") == "state"
    assert sandbox.get_state("task_prompt") == task_prompt
    assert sandbox.get_state("plan") == "Step 1, Step 2"
    assert sandbox.get_state("output") == "Completed result"


def test_triad_agent_sandbox_v2_failed_evaluation_rollback() -> None:
    """Test the triad sandbox v2 when the evaluator rejects the output, verifying rollback."""
    mock_planner = MagicMock()
    mock_planner.plan.return_value = "Step A"

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Bad result"

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.return_value = False

    sandbox = TriadAgentSandboxV2(
        planner=mock_planner,
        generator=mock_generator,
        evaluator=mock_evaluator
    )

    sandbox.set_state("previous_data", 123)

    task_prompt = "Do task A"

    with pytest.raises(ValueError, match="Generator output failed Evaluator validation."):
        sandbox.execute(task_prompt)

    # Verify calls
    mock_planner.plan.assert_called_once_with(task_prompt)
    mock_generator.generate.assert_called_once_with("Step A")
    mock_evaluator.evaluate.assert_called_once_with(task_prompt, "Step A", "Bad result")

    # State should have rolled back to before execution
    assert sandbox.get_state("previous_data") == 123
    assert sandbox.get_state("task_prompt") is None
    assert sandbox.get_state("plan") is None
    assert sandbox.get_state("output") is None

def test_triad_agent_sandbox_v2_generator_error_rollback() -> None:
    """Test the triad sandbox v2 when the generator raises an error, verifying rollback."""
    mock_planner = MagicMock()
    mock_planner.plan.return_value = "Step A"

    mock_generator = MagicMock()
    mock_generator.generate.side_effect = RuntimeError("Generation failed")

    mock_evaluator = MagicMock()

    sandbox = TriadAgentSandboxV2(
        planner=mock_planner,
        generator=mock_generator,
        evaluator=mock_evaluator
    )

    sandbox.set_state("valid_state", True)

    task_prompt = "Do task A"

    with pytest.raises(RuntimeError, match="Generation failed"):
        sandbox.execute(task_prompt)

    # State should have rolled back
    assert sandbox.get_state("valid_state") is True
    assert sandbox.get_state("task_prompt") is None
    assert sandbox.get_state("plan") is None
    assert sandbox.get_state("output") is None
