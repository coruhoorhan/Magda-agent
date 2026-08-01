import pytest
from unittest.mock import MagicMock
from typing import Dict, Any

from magda_agent.safety.acs_workflow import AgenticWorkflow
from magda_agent.safety.acs import ACSWorkflowGuard

@pytest.fixture
def mock_guard():
    guard = MagicMock(spec=ACSWorkflowGuard)
    # Default to passing all checkpoints
    guard.checkpoint_1_input_validation.return_value = (True, "Passed")
    guard.checkpoint_2_intent_authorization.return_value = (True, "Passed")
    guard.checkpoint_3_tool_policy.return_value = (True, "Passed")
    guard.checkpoint_4_state_transition.return_value = (True, "Passed")
    guard.checkpoint_5_output_sanitization.return_value = (True, "Passed")
    return guard

@pytest.fixture
def workflow(mock_guard):
    return AgenticWorkflow(guard=mock_guard)

def test_successful_execution(workflow, mock_guard):
    mock_tool = MagicMock(return_value="Valid output")

    result = workflow.execute_action(
        action="execute",
        tool_name="test_tool",
        tool_func=mock_tool,
        kwargs={"arg1": "val1"},
        current_state="idle",
        next_state="executing"
    )

    assert result["status"] == "success"
    assert result["current_state"] == "executing"
    assert result["output"] == "Valid output"
    mock_tool.assert_called_once_with(arg1="val1")

    # Ensure all 5 checkpoints were called
    assert mock_guard.checkpoint_1_input_validation.called
    assert mock_guard.checkpoint_2_intent_authorization.called
    assert mock_guard.checkpoint_3_tool_policy.called
    assert mock_guard.checkpoint_4_state_transition.called
    assert mock_guard.checkpoint_5_output_sanitization.called


def test_failure_checkpoint_1_input_validation(workflow, mock_guard):
    mock_guard.checkpoint_1_input_validation.return_value = (False, "Invalid input")
    mock_tool = MagicMock()

    result = workflow.execute_action(
        action="execute",
        tool_name="test_tool",
        tool_func=mock_tool,
        kwargs={},
        current_state="idle",
        next_state="executing"
    )

    assert result["status"] == "error"
    assert result["current_state"] == "error"
    assert "Invalid input" in result["error_reason"]
    mock_tool.assert_not_called()

def test_failure_checkpoint_2_intent_authorization(workflow, mock_guard):
    mock_guard.checkpoint_2_intent_authorization.return_value = (False, "Unauthorized")
    mock_tool = MagicMock()

    result = workflow.execute_action(
        action="execute",
        tool_name="test_tool",
        tool_func=mock_tool,
        kwargs={},
        current_state="idle",
        next_state="executing"
    )

    assert result["status"] == "error"
    assert "Unauthorized" in result["error_reason"]
    mock_tool.assert_not_called()

def test_failure_checkpoint_3_tool_policy(workflow, mock_guard):
    mock_guard.checkpoint_3_tool_policy.return_value = (False, "Policy violation")
    mock_tool = MagicMock()

    result = workflow.execute_action(
        action="execute",
        tool_name="test_tool",
        tool_func=mock_tool,
        kwargs={},
        current_state="idle",
        next_state="executing"
    )

    assert result["status"] == "error"
    assert "Policy violation" in result["error_reason"]
    mock_tool.assert_not_called()

def test_failure_checkpoint_4_state_transition(workflow, mock_guard):
    mock_guard.checkpoint_4_state_transition.return_value = (False, "Invalid transition")
    mock_tool = MagicMock()

    result = workflow.execute_action(
        action="execute",
        tool_name="test_tool",
        tool_func=mock_tool,
        kwargs={},
        current_state="idle",
        next_state="executing"
    )

    assert result["status"] == "error"
    assert "Invalid transition" in result["error_reason"]
    mock_tool.assert_not_called()

def test_tool_execution_exception(workflow, mock_guard):
    mock_tool = MagicMock(side_effect=ValueError("Simulated tool crash"))

    result = workflow.execute_action(
        action="execute",
        tool_name="test_tool",
        tool_func=mock_tool,
        kwargs={},
        current_state="idle",
        next_state="executing"
    )

    assert result["status"] == "error"
    assert "Simulated tool crash" in result["error_reason"]
    assert mock_guard.checkpoint_5_output_sanitization.called is False

def test_failure_checkpoint_5_output_sanitization(workflow, mock_guard):
    mock_guard.checkpoint_5_output_sanitization.return_value = (False, "Sensitive data found")
    mock_tool = MagicMock(return_value="Secret Key 123")

    result = workflow.execute_action(
        action="execute",
        tool_name="test_tool",
        tool_func=mock_tool,
        kwargs={},
        current_state="idle",
        next_state="executing"
    )

    assert result["status"] == "error"
    assert "Sensitive data found" in result["error_reason"]
    mock_tool.assert_called_once()
