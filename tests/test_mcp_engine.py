import pytest
from unittest.mock import MagicMock, AsyncMock
from magda_agent.skills.mcp_engine import MCPEngine
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.mcp_client import MCPClient

def test_mcp_engine_import() -> None:
    """Verify MCPEngine reads MCP standard tool definitions and wraps them."""
    registry = SkillRegistry()
    mcp_client = MCPClient()
    mcp_client.register_remote_tool = MagicMock()

    engine = MCPEngine(registry, mcp_client)

    tool_def = {
        "name": "external_weather",
        "description": "Get current weather.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
    connection_info = {"url": "http://localhost:8000/mcp"}

    engine.import_mcp_tool(tool_def, connection_info)

    # Verify tool routing is registered
    mcp_client.register_remote_tool.assert_called_once_with("external_weather", connection_info)

    # Verify skill is dynamically registered in Magda
    assert registry.has_skill("external_weather")
    assert registry.descriptions["external_weather"] == "Get current weather."

    # Verify schema preservation
    skill_func = registry.skills["external_weather"]
    assert hasattr(skill_func, "__mcp_schema__")
    assert getattr(skill_func, "__mcp_schema__") == tool_def["inputSchema"]

@pytest.mark.asyncio
async def test_mcp_engine_wrapper_execution() -> None:
    """Verify the dynamically wrapped skill executes correctly."""
    registry = SkillRegistry()
    mcp_client = MCPClient()
    mcp_client.execute_tool = AsyncMock(return_value="Sunny, 25C")

    engine = MCPEngine(registry, mcp_client)

    tool_def = {"name": "external_weather"}
    connection_info = {"url": "mock"}

    engine.import_mcp_tool(tool_def, connection_info)

    result = await registry.execute_skill("external_weather", location="Paris")

    assert result == "Sunny, 25C"
    mcp_client.execute_tool.assert_awaited_once_with("external_weather", location="Paris")

def test_mcp_engine_invalid_tool_def() -> None:
    """Verify MCPEngine raises ValueError on invalid tool definition without a name."""
    registry = SkillRegistry()
    mcp_client = MCPClient()
    engine = MCPEngine(registry, mcp_client)

    with pytest.raises(ValueError, match="must include a 'name'"):
        engine.import_mcp_tool({"description": "No name tool"}, {})

def test_mcp_engine_import_with_server_name() -> None:
    """Verify MCPEngine handles server_name prefixing correctly."""
    registry = SkillRegistry()
    mcp_client = MCPClient()
    mcp_client.register_mcp_server = MagicMock()

    engine = MCPEngine(registry, mcp_client)

    tool_def = {
        "name": "get_files",
        "description": "Get files from server.",
        "inputSchema": {}
    }
    connection_info = {"url": "http://localhost:8001/mcp"}

    engine.import_mcp_tool(tool_def, connection_info, server_name="filesystem")

    # Verify server routing is registered
    mcp_client.register_mcp_server.assert_called_once_with("filesystem", connection_info)

    # Verify skill is dynamically registered in Magda with the prefix
    assert registry.has_skill("filesystem.get_files")
    assert registry.descriptions["filesystem.get_files"] == "Get files from server."

@pytest.mark.asyncio
async def test_mcp_engine_wrapper_execution_with_server_name() -> None:
    """Verify the dynamically wrapped skill executes correctly with a server_name prefix."""
    registry = SkillRegistry()
    mcp_client = MCPClient()
    mcp_client.execute_tool = AsyncMock(return_value="file_content.txt")

    engine = MCPEngine(registry, mcp_client)

    tool_def = {"name": "get_files"}
    connection_info = {"url": "mock"}

    engine.import_mcp_tool(tool_def, connection_info, server_name="filesystem")

    result = await registry.execute_skill("filesystem.get_files", path="/home/user")

    assert result == "file_content.txt"
    mcp_client.execute_tool.assert_awaited_once_with("filesystem.get_files", path="/home/user")
