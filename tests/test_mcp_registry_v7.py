import pytest
from magda_agent.skills.mcp_registry_v7 import MCPRegistryV7

@pytest.fixture
def registry():
    return MCPRegistryV7()

def test_register_tool_valid(registry):
    tool_schema = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
    assert registry.register_tool(tool_schema) is True
    assert "test_tool" in registry.list_tools()
    assert registry.get_tool("test_tool") == tool_schema

def test_register_tool_invalid_missing_name(registry):
    tool_schema = {
        "description": "A test tool"
    }
    assert registry.register_tool(tool_schema) is False
    assert len(registry.list_tools()) == 0

def test_register_tool_invalid_missing_description(registry):
    tool_schema = {
        "name": "test_tool"
    }
    assert registry.register_tool(tool_schema) is False

def test_register_tool_invalid_wrong_input_schema(registry):
    tool_schema = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": "not a dictionary"
    }
    assert registry.register_tool(tool_schema) is False

def test_unregister_tool(registry):
    tool_schema = {
        "name": "test_tool",
        "description": "A test tool"
    }
    registry.register_tool(tool_schema)
    assert registry.unregister_tool("test_tool") is True
    assert "test_tool" not in registry.list_tools()

def test_unregister_tool_not_found(registry):
    assert registry.unregister_tool("nonexistent") is False

def test_list_tools(registry):
    assert registry.list_tools() == []
    registry.register_tool({"name": "tool1", "description": "desc1"})
    registry.register_tool({"name": "tool2", "description": "desc2"})
    tools = registry.list_tools()
    assert "tool1" in tools
    assert "tool2" in tools
    assert len(tools) == 2

def test_get_tool_not_found(registry):
    assert registry.get_tool("nonexistent") == {}
