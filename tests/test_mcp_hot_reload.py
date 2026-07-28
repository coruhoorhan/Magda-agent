import pytest
from magda_agent.skills.mcp_registry import MCPRegistry
from magda_agent.skills.mcp_engine import MCPEngine
from magda_agent.skills.mcp_client import MCPClient
from magda_agent.skills.registry import SkillRegistry


def test_mcp_registry_reload_tool_success():
    """Test successful hot reload of a tool in MCPRegistry."""
    registry = MCPRegistry()
    initial_schema = {
        "name": "test_tool",
        "description": "Initial description"
    }
    updated_schema = {
        "name": "test_tool",
        "description": "Updated description"
    }

    assert registry.load_tool(initial_schema) is True
    assert registry.get_tool("test_tool")["description"] == "Initial description"

    assert registry.reload_tool(updated_schema) is True
    assert registry.get_tool("test_tool")["description"] == "Updated description"


def test_mcp_registry_reload_tool_new():
    """Test hot reload handles new tools gracefully."""
    registry = MCPRegistry()
    schema = {
        "name": "new_tool",
        "description": "New description"
    }

    assert registry.reload_tool(schema) is True
    assert registry.get_tool("new_tool")["description"] == "New description"


def test_mcp_registry_reload_tool_invalid():
    """Test hot reload handles invalid schemas appropriately."""
    registry = MCPRegistry()
    invalid_schema = {
        "description": "Missing name"
    }

    assert registry.reload_tool(invalid_schema) is False


def test_mcp_engine_reload_tool():
    """Test successful hot reload of a tool in MCPEngine."""
    skill_registry = SkillRegistry()
    mcp_client = MCPClient()
    mcp_engine = MCPEngine(skill_registry, mcp_client)

    initial_def = {
        "name": "dynamic_tool",
        "description": "Dynamic initial"
    }
    updated_def = {
        "name": "dynamic_tool",
        "description": "Dynamic updated"
    }
    connection_info = {"url": "http://localhost:8000"}
    updated_connection_info = {"url": "http://localhost:8001"}

    # Import initially
    mcp_engine.import_mcp_tool(initial_def, connection_info)
    assert skill_registry.has_skill("dynamic_tool")
    assert skill_registry.descriptions["dynamic_tool"] == "Dynamic initial"
    assert mcp_client.registered_tools["dynamic_tool"] == {"url": "http://localhost:8000"}

    # Reload tool
    mcp_engine.reload_mcp_tool(updated_def, updated_connection_info)
    assert skill_registry.has_skill("dynamic_tool")
    assert skill_registry.descriptions["dynamic_tool"] == "Dynamic updated"
    assert mcp_client.registered_tools["dynamic_tool"] == {"url": "http://localhost:8001"}


def test_mcp_engine_reload_tool_with_server():
    """Test successful hot reload of a tool in MCPEngine with a server name."""
    skill_registry = SkillRegistry()
    mcp_client = MCPClient()
    mcp_engine = MCPEngine(skill_registry, mcp_client)

    initial_def = {
        "name": "server_tool",
        "description": "Server initial"
    }
    updated_def = {
        "name": "server_tool",
        "description": "Server updated"
    }
    connection_info = {"url": "http://server:8000"}
    updated_connection_info = {"url": "http://server:8001"}

    # Import initially
    mcp_engine.import_mcp_tool(initial_def, connection_info, server_name="test_server")
    effective_name = "test_server__server_tool"
    assert skill_registry.has_skill(effective_name)
    assert skill_registry.descriptions[effective_name] == "Server initial"
    assert mcp_client.registered_servers["test_server"] == {"url": "http://server:8000"}

    # Reload tool
    mcp_engine.reload_mcp_tool(updated_def, updated_connection_info, server_name="test_server")
    assert skill_registry.has_skill(effective_name)
    assert skill_registry.descriptions[effective_name] == "Server updated"
    assert mcp_client.registered_servers["test_server"] == {"url": "http://server:8001"}
