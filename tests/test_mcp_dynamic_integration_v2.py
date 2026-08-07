import pytest
from typing import Any, Dict
from unittest.mock import MagicMock
from magda_agent.skills.mcp_registry_v7 import MCPRegistryV7
from magda_agent.integration.mcp_dynamic_integration_v2 import MCPDynamicIntegrationV2

@pytest.fixture
def registry() -> MCPRegistryV7:
    """Provides a fresh instance of MCPRegistryV7."""
    return MCPRegistryV7()

@pytest.fixture
def mcp_integration(registry: MCPRegistryV7) -> MCPDynamicIntegrationV2:
    """Provides a fresh instance of MCPDynamicIntegrationV2."""
    return MCPDynamicIntegrationV2(registry=registry)

def test_register_and_wrap_success(mcp_integration: MCPDynamicIntegrationV2) -> None:
    """Test successfully wrapping and registering a valid MCP tool schema."""
    schema: Dict[str, Any] = {
        "name": "mock_tool",
        "description": "A mock tool for testing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "arg1": {"type": "string"}
            }
        }
    }
    mock_executor = MagicMock(return_value="success")

    result = mcp_integration.register_and_wrap(schema, mock_executor)
    assert result is True
    assert "mock_tool" in mcp_integration.registry.list_tools()
    assert mcp_integration._executors["mock_tool"] == mock_executor

def test_register_and_wrap_invalid_schema(mcp_integration: MCPDynamicIntegrationV2) -> None:
    """Test wrapping an invalid schema fails appropriately."""
    schema: Dict[str, Any] = {
        "description": "Missing name field."
    }
    mock_executor = MagicMock()

    result = mcp_integration.register_and_wrap(schema, mock_executor)
    assert result is False
    assert len(mcp_integration._executors) == 0

def test_execute_tool_success(mcp_integration: MCPDynamicIntegrationV2) -> None:
    """Test dynamically executing a registered tool."""
    schema: Dict[str, Any] = {
        "name": "mock_tool_exec",
        "description": "Mock executor."
    }

    def executor_func(arg1: str) -> str:
        return f"Executed with {arg1}"

    mcp_integration.register_and_wrap(schema, executor_func)

    result = mcp_integration.execute_tool("mock_tool_exec", arg1="test_value")
    assert result == "Executed with test_value"

def test_execute_tool_not_found(mcp_integration: MCPDynamicIntegrationV2) -> None:
    """Test executing a tool that doesn't exist returns None."""
    result = mcp_integration.execute_tool("nonexistent_tool")
    assert result is None

def test_execute_tool_exception_handling(mcp_integration: MCPDynamicIntegrationV2) -> None:
    """Test execution failure is handled gracefully."""
    schema: Dict[str, Any] = {
        "name": "mock_tool_fail",
        "description": "Mock failure."
    }

    mock_executor = MagicMock(side_effect=ValueError("Execution error"))
    mcp_integration.register_and_wrap(schema, mock_executor)

    result = mcp_integration.execute_tool("mock_tool_fail")
    assert result is None
    mock_executor.assert_called_once()

def test_unregister_tool(mcp_integration: MCPDynamicIntegrationV2) -> None:
    """Test unregistering an MCP tool."""
    schema: Dict[str, Any] = {
        "name": "mock_tool_unreg",
        "description": "Mock unregister."
    }
    mock_executor = MagicMock()

    mcp_integration.register_and_wrap(schema, mock_executor)
    assert "mock_tool_unreg" in mcp_integration._executors

    result = mcp_integration.unregister_tool("mock_tool_unreg")
    assert result is True
    assert "mock_tool_unreg" not in mcp_integration._executors
    assert "mock_tool_unreg" not in mcp_integration.registry.list_tools()
