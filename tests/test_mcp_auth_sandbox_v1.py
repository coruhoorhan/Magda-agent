"""
Tests for the MCP Dynamic Capability Auth Sandbox.
"""
import pytest
from unittest.mock import MagicMock

from magda_agent.security.mcp_auth_sandbox_v1 import MCPAuthSandbox, AuthSecurityError


@pytest.fixture
def sandbox():
    """Fixture to provide a clean MCPAuthSandbox instance."""
    return MCPAuthSandbox()


def test_execute_tool_success_no_auth_required(sandbox):
    """Test execution of a tool that does not require any auth bindings."""
    mock_func = MagicMock(return_value="success")
    sandbox.register_tool("public_tool", mock_func)

    result = sandbox.execute_tool("public_tool", {"param": "value"})

    assert result == "success"
    mock_func.assert_called_once_with(param="value")


def test_execute_tool_success_with_valid_auth(sandbox):
    """Test execution of a tool with a valid exact token binding."""
    mock_func = MagicMock(return_value="protected_data")
    sandbox.register_tool("private_tool", mock_func, required_token_binding="valid-token")

    result = sandbox.execute_tool("private_tool", {"user_id": 123}, auth_token="valid-token")

    assert result == "protected_data"
    mock_func.assert_called_once_with(user_id=123)


def test_execute_tool_success_with_valid_prefix_auth(sandbox):
    """Test execution of a tool with a valid prefix token binding."""
    mock_func = MagicMock(return_value="prefixed_data")
    sandbox.register_tool("prefixed_tool", mock_func, required_token_binding="oauth2")

    result = sandbox.execute_tool("prefixed_tool", {}, auth_token="oauth2:user-token-xyz")

    assert result == "prefixed_data"
    mock_func.assert_called_once()


def test_execute_tool_denied_missing_token(sandbox):
    """Test that the sandbox denies execution if the token is missing but required."""
    mock_func = MagicMock()
    sandbox.register_tool("secure_tool", mock_func, required_token_binding="secret")

    with pytest.raises(AuthSecurityError, match="requires an auth token, but none was provided"):
        sandbox.execute_tool("secure_tool", {})

    mock_func.assert_not_called()


def test_execute_tool_denied_invalid_token(sandbox):
    """Test that the sandbox denies execution if the provided token is invalid."""
    mock_func = MagicMock()
    sandbox.register_tool("highly_secure_tool", mock_func, required_token_binding="admin-token")

    with pytest.raises(AuthSecurityError, match="Invalid token binding"):
        sandbox.execute_tool("highly_secure_tool", {}, auth_token="guest-token")

    mock_func.assert_not_called()


def test_execute_unregistered_tool(sandbox):
    """Test that attempting to execute an unregistered tool raises an error."""
    with pytest.raises(AuthSecurityError, match="not registered in the sandbox"):
        sandbox.execute_tool("unknown_tool", {})
