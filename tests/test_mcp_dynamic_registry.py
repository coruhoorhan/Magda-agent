import pytest
from magda_agent.skills.mcp_registry import MCPRegistry
from magda_agent.skills.mcp_dynamic_registry import MCPDynamicRegistrarV4

def test_register_tool_at_runtime_success_sync():
    """Test successful dynamic registration of a synchronous MCP tool via V4."""
    registry = MCPRegistry()
    registrar = MCPDynamicRegistrarV4(registry)

    valid_schema = {
        "name": "test_sync_tool_v4",
        "description": "A tool for testing sync behavior in V4."
    }

    result = registrar.register_tool_at_runtime(valid_schema, is_async=False)

    assert result is True
    assert "test_sync_tool_v4" in registry.list_tools()
    tool = registry.get_tool("test_sync_tool_v4")
    assert tool["name"] == "test_sync_tool_v4"
    assert tool["__is_async__"] is False

def test_register_tool_at_runtime_success_async():
    """Test successful dynamic registration of an asynchronous MCP tool via V4."""
    registry = MCPRegistry()
    registrar = MCPDynamicRegistrarV4(registry)

    valid_schema = {
        "name": "test_async_tool_v4",
        "description": "A tool for testing async behavior in V4."
    }

    result = registrar.register_tool_at_runtime(valid_schema, is_async=True)

    assert result is True
    assert "test_async_tool_v4" in registry.list_tools()
    tool = registry.get_tool("test_async_tool_v4")
    assert tool["name"] == "test_async_tool_v4"
    assert tool["__is_async__"] is True

def test_register_tool_at_runtime_failure():
    """Test failing dynamic registration of an MCP tool due to invalid schema via V4."""
    registry = MCPRegistry()
    registrar = MCPDynamicRegistrarV4(registry)

    invalid_schema = {
        "name": "invalid_tool_v4"
        # Missing description
    }

    result = registrar.register_tool_at_runtime(invalid_schema)

    assert result is False
    assert "invalid_tool_v4" not in registry.list_tools()


def test_register_tool_at_runtime_enhanced_validation_failure():
    """Test failing dynamic registration due to enhanced validation (invalid name)."""
    registry = MCPRegistry()
    registrar = MCPDynamicRegistrarV4(registry)

    invalid_schema_name = {
        "name": "invalid name with spaces",
        "description": "A tool with an invalid name."
    }

    result = registrar.register_tool_at_runtime(invalid_schema_name)
    assert result is False

    invalid_schema_input = {
        "name": "valid_name",
        "description": "A tool with invalid inputSchema.",
        "inputSchema": "not_a_dict"
    }

    result = registrar.register_tool_at_runtime(invalid_schema_input)
    assert result is False

import asyncio
from unittest.mock import AsyncMock, MagicMock
from magda_agent.skills.mcp_dynamic_registry import MCPPrefixedToolRegistry
from magda_agent.skills.mcp_client import MCPClient

@pytest.fixture
def mcp_client_mock():
    client = MagicMock(spec=MCPClient)
    client.execute_tool = AsyncMock()
    return client

@pytest.fixture
def prefixed_registry(mcp_client_mock):
    return MCPPrefixedToolRegistry(mcp_client_mock)

@pytest.mark.asyncio
async def test_execute_tool_with_prefix(prefixed_registry, mcp_client_mock):
    mcp_client_mock.execute_tool.return_value = "prefixed result"

    result = await prefixed_registry.execute_tool("serverA.calculator", expr="2+2")

    mcp_client_mock.execute_tool.assert_called_once_with("serverA.calculator", expr="2+2")
    assert result == "prefixed result"

@pytest.mark.asyncio
async def test_execute_tool_without_prefix(prefixed_registry, mcp_client_mock):
    mcp_client_mock.execute_tool.return_value = "global result"

    result = await prefixed_registry.execute_tool("global_tool", p1="val1")

    mcp_client_mock.execute_tool.assert_called_once_with("global_tool", p1="val1")
    assert result == "global result"

@pytest.mark.asyncio
async def test_runtime_concurrency(prefixed_registry, mcp_client_mock):
    # Setup mock to return different things based on input
    async def mock_execute_tool(tool_name, **kwargs):
        if tool_name == "tool1":
            return "result 1"
        elif tool_name == "tool2":
            return "result 2"
        return "unknown"

    mcp_client_mock.execute_tool.side_effect = mock_execute_tool

    tasks = [
        {"tool": "tool1", "params": {"p1": "v1"}},
        {"tool": "tool2", "params": {"p2": "v2"}},
        {"tool": "tool3"} # missing params
    ]

    results = await prefixed_registry.execute_concurrently(tasks)

    assert len(results) == 3
    assert results == ["result 1", "result 2", "unknown"]
    assert mcp_client_mock.execute_tool.call_count == 3
