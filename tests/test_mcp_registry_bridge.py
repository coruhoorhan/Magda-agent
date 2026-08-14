import pytest
from unittest.mock import MagicMock
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.mcp_exporter import MCPSkillExporter
from magda_agent.skills.mcp_registry_v7 import MCPRegistryV7
from magda_agent.skills.mcp_registry_bridge import MCPRegistryBridge

def dummy_skill(arg1: str, arg2: int = 5) -> str:
    """A dummy skill for testing."""
    return f"{arg1} {arg2}"

def test_bridge_skills_success():
    """Test that skills are successfully bridged to MCPRegistryV7."""
    registry = SkillRegistry()
    registry.register_skill("dummy_skill", dummy_skill, "A dummy skill for testing.")

    exporter = MCPSkillExporter(registry)
    mcp_registry = MCPRegistryV7()
    bridge = MCPRegistryBridge(registry, exporter)

    # Initially, mcp_registry should be empty
    assert len(mcp_registry.list_tools()) == 0

    registered_count = bridge.bridge_skills(mcp_registry)

    # Should have registered 1 skill
    assert registered_count == 1
    assert "dummy_skill" in mcp_registry.list_tools()

    tool = mcp_registry.get_tool("dummy_skill")
    assert tool["name"] == "dummy_skill"
    assert tool["description"] == "A dummy skill for testing."
    assert "inputSchema" in tool
    assert tool["inputSchema"]["type"] == "object"
    assert "arg1" in tool["inputSchema"]["properties"]

def test_bridge_skills_failure(monkeypatch):
    """Test that failed registrations are handled correctly."""
    registry = SkillRegistry()
    registry.register_skill("dummy_skill", dummy_skill, "A dummy skill for testing.")

    exporter = MCPSkillExporter(registry)
    mcp_registry = MCPRegistryV7()

    # Mock the register_tool method to always fail
    def mock_register_tool(tool_schema):
        return False

    monkeypatch.setattr(mcp_registry, "register_tool", mock_register_tool)

    bridge = MCPRegistryBridge(registry, exporter)
    registered_count = bridge.bridge_skills(mcp_registry)

    # Should have registered 0 skills
    assert registered_count == 0
    assert len(mcp_registry.list_tools()) == 0
