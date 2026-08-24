import pytest
from unittest.mock import AsyncMock
from magda_agent.skills.mcp_registry_v7 import MCPRegistryV7, MCPActionAdapter

@pytest.fixture
def registry():
    return MCPRegistryV7()

def test_load_action_tool_valid(registry):
    tool_schema = {
        "name": "create_file",
        "description": "Creates a file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"}
            }
        }
    }
    assert registry.load_action_tool(tool_schema) is True
    assert "create_file" in registry.list_action_tools()
    assert registry.get_action_tool("create_file") == tool_schema

def test_load_action_tool_invalid(registry):
    tool_schema = {
        "name": "create_file",
        # Missing description
    }
    assert registry.load_action_tool(tool_schema) is False
    assert "create_file" not in registry.list_action_tools()

    tool_schema_invalid_input = {
        "name": "create_file",
        "description": "Creates a file",
        "inputSchema": "invalid_string"
    }
    assert registry.load_action_tool(tool_schema_invalid_input) is False

def test_unload_action_tool(registry):
    tool_schema = {
        "name": "create_file",
        "description": "Creates a file"
    }
    registry.load_action_tool(tool_schema)
    assert registry.unload_action_tool("create_file") is True
    assert "create_file" not in registry.list_action_tools()
    assert registry.unload_action_tool("nonexistent") is False

def test_clear_registry(registry):
    tool_schema = {
        "name": "create_file",
        "description": "Creates a file"
    }
    registry.load_action_tool(tool_schema)
    mock_adapter = AsyncMock(spec=MCPActionAdapter)
    registry.register_adapter(mock_adapter)

    assert len(registry.list_action_tools()) == 1
    assert len(registry.adapters) == 1

    registry.clear()

    assert len(registry.list_action_tools()) == 0
    assert len(registry.adapters) == 0

@pytest.mark.asyncio
async def test_sync_from_adapters(registry):
    mock_adapter = AsyncMock(spec=MCPActionAdapter)
    mock_adapter.fetch_action_tools.return_value = [
        {"name": "tool1", "description": "Action Tool 1"},
        {"name": "tool2", "description": "Action Tool 2"}
    ]

    registry.register_adapter(mock_adapter)

    count = await registry.sync_from_adapters()
    assert count == 2
    assert "tool1" in registry.list_action_tools()
    assert "tool2" in registry.list_action_tools()

@pytest.mark.asyncio
async def test_sync_from_adapters_with_invalid_tool(registry):
    mock_adapter = AsyncMock(spec=MCPActionAdapter)
    mock_adapter.fetch_action_tools.return_value = [
        {"name": "valid_tool", "description": "Valid Tool"},
        {"name": "invalid_tool"} # Missing description
    ]

    registry.register_adapter(mock_adapter)

    count = await registry.sync_from_adapters()
    assert count == 1
    assert "valid_tool" in registry.list_action_tools()
    assert "invalid_tool" not in registry.list_action_tools()

@pytest.mark.asyncio
async def test_sync_from_adapters_exception(registry):
    mock_adapter = AsyncMock(spec=MCPActionAdapter)
    mock_adapter.fetch_action_tools.side_effect = Exception("Network error")

    registry.register_adapter(mock_adapter)

    count = await registry.sync_from_adapters()
    assert count == 0
    assert len(registry.list_action_tools()) == 0
