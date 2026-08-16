import pytest
from unittest.mock import MagicMock

from magda_agent.safety.mcp_enforcer_v7 import MCPActionEnforcer
from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.acs import SecurityViolationError

def test_mcp_enforcer_safe_tool_passes():
    enforcer = MCPActionEnforcer()
    mock_policy = MagicMock(spec=PolicyLayer)

    # "read_file" is not in the default sensitive prefixes list
    result = enforcer.enforce("read_file", {"path": "test.txt"}, mock_policy)

    assert result is True
    # Policy evaluation should not even be called for safe tools
    mock_policy.evaluate.assert_not_called()

def test_mcp_enforcer_high_risk_tool_approved():
    enforcer = MCPActionEnforcer()
    mock_policy = MagicMock(spec=PolicyLayer)

    # Mock evaluate to return allowed
    mock_policy.evaluate.return_value = (True, "Action 'write_file' is allowed.")

    # "write_file" starts with "write_" which is a sensitive prefix
    result = enforcer.enforce("write_file", {"path": "test.txt", "content": "hello"}, mock_policy)

    assert result is True
    mock_policy.evaluate.assert_called_once_with("write_file", path="test.txt", content="hello")

def test_mcp_enforcer_high_risk_tool_blocked():
    enforcer = MCPActionEnforcer()
    mock_policy = MagicMock(spec=PolicyLayer)

    # Mock evaluate to return denied
    mock_policy.evaluate.return_value = (False, "Action denied: access to sensitive path.")

    with pytest.raises(SecurityViolationError) as exc_info:
        enforcer.enforce("delete_file", {"path": ".env"}, mock_policy)

    assert "High-risk MCP action 'delete_file' blocked by policy: Action denied: access to sensitive path." in str(exc_info.value)
    mock_policy.evaluate.assert_called_once_with("delete_file", path=".env")

def test_mcp_enforcer_custom_sensitive_prefixes():
    enforcer = MCPActionEnforcer(sensitive_prefixes=["custom_"])
    mock_policy = MagicMock(spec=PolicyLayer)

    # Mock evaluate to return allowed
    mock_policy.evaluate.return_value = (True, "Action 'custom_action' is allowed.")

    result = enforcer.enforce("custom_action", {}, mock_policy)

    assert result is True
    mock_policy.evaluate.assert_called_once_with("custom_action")

    mock_policy.reset_mock()

    # "write_file" is no longer high risk because we overrode the default prefixes
    result = enforcer.enforce("write_file", {}, mock_policy)
    assert result is True
    mock_policy.evaluate.assert_not_called()
