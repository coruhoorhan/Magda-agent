"""
Tests for the MCP Action Tool Governance V5 Module.
"""
import logging
import pytest
from unittest.mock import MagicMock

from magda_agent.skills.mcp_governance_v5 import ActionToolGovernance, GovernanceError, ToolExecutor


@pytest.fixture
def mock_backend():
    """Fixture to provide a mocked underlying tool executor."""
    backend = MagicMock(spec=ToolExecutor)
    return backend


@pytest.fixture
def mock_logger():
    """Fixture to provide a mocked logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def governance(mock_backend, mock_logger):
    """Fixture to provide a configured ActionToolGovernance instance."""
    return ActionToolGovernance(backend=mock_backend, logger=mock_logger)


def test_execute_tool_success_logging(governance, mock_backend, mock_logger):
    """Test that a successful tool execution is properly intercepted and logged."""
    mock_backend.execute_tool.return_value = "success_result"

    result = governance.execute_tool("my_tool", {"arg1": "val1"}, auth_token="token123")

    # Verify execution was forwarded correctly
    mock_backend.execute_tool.assert_called_once_with("my_tool", {"arg1": "val1"}, "token123")
    assert result == "success_result"

    # Verify logging
    assert mock_logger.info.call_count == 2

    # Check pre-execution log
    pre_exec_log = mock_logger.info.call_args_list[0][0][0]
    assert "[GOVERNANCE] Intercepted execution attempt for tool 'my_tool'" in pre_exec_log
    assert "['arg1']" in pre_exec_log
    assert "Yes" in pre_exec_log

    # Check post-execution log
    post_exec_log = mock_logger.info.call_args_list[1][0][0]
    assert "[GOVERNANCE] Successfully executed tool 'my_tool'" in post_exec_log


def test_execute_tool_success_logging_no_auth(governance, mock_backend, mock_logger):
    """Test interception and logging when no auth token is provided."""
    mock_backend.execute_tool.return_value = "success"

    governance.execute_tool("public_tool", {})

    mock_backend.execute_tool.assert_called_once_with("public_tool", {}, None)

    pre_exec_log = mock_logger.info.call_args_list[0][0][0]
    assert "Auth provided: No" in pre_exec_log


def test_execute_tool_backend_exception_interception(governance, mock_backend, mock_logger):
    """Test that if the backend raises an error, governance catches, logs, and wraps it."""
    mock_backend.execute_tool.side_effect = ValueError("Invalid arguments")

    with pytest.raises(GovernanceError, match="Backend execution failed for tool 'failing_tool': Invalid arguments"):
        governance.execute_tool("failing_tool", {})

    # Verify error logging
    mock_logger.error.assert_called_once()
    error_log = mock_logger.error.call_args[0][0]
    assert "[GOVERNANCE] Tool execution failed for 'failing_tool'" in error_log
    assert "Invalid arguments" in error_log


def test_execute_tool_propagates_governance_error(governance, mock_backend, mock_logger):
    """Test that existing GovernanceErrors from the backend are propagated as-is without extra wrapping."""
    mock_backend.execute_tool.side_effect = GovernanceError("Policy violation")

    with pytest.raises(GovernanceError, match="Policy violation") as exc_info:
        governance.execute_tool("restricted_tool", {})

    # Ensure it's not double-wrapped (i.e. "Backend execution failed... Policy violation")
    assert "Backend execution failed" not in str(exc_info.value)

    # Error should still be logged
    mock_logger.error.assert_called_once()
