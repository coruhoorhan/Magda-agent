import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.skills.mcp_client import MCPClient
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.mcp_engine_v5 import MCPEngineV5

@pytest.fixture
def mock_mcp_client():
    client = MagicMock(spec=MCPClient)
    client.execute_tool = AsyncMock()
    return client

@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=SkillRegistry)
    # Provide a simple dictionary to hold skills
    registry.skills = {}

    def register_skill(name, func, description):
        registry.skills[name] = func

    registry.register_skill.side_effect = register_skill

    def has_skill(name):
        return name in registry.skills

    registry.has_skill.side_effect = has_skill
    return registry

@pytest.fixture
def mcp_engine(mock_registry, mock_mcp_client):
    return MCPEngineV5(registry=mock_registry, mcp_client=mock_mcp_client)

@pytest.mark.asyncio
async def test_mcp_engine_v5_normal_success(mcp_engine, mock_mcp_client, mock_registry):
    # Setup
    tool_def = {"name": "test_tool", "description": "Test tool desc"}
    connection_info = {"url": "http://localhost/test"}
    mock_mcp_client.execute_tool.return_value = "Normal Success"

    # Action
    mcp_engine.import_mcp_tool(tool_def, connection_info)

    assert "test_tool" in mock_registry.skills
    wrapper_func = mock_registry.skills["test_tool"]

    result = await wrapper_func(arg1="val1")

    # Verify
    mock_mcp_client.execute_tool.assert_called_once_with("test_tool", arg1="val1")
    assert result == "Normal Success"

@pytest.mark.asyncio
async def test_mcp_engine_v5_fallback_success(mcp_engine, mock_mcp_client, mock_registry):
    # Setup
    tool_def = {"name": "test_tool_with_fallback"}
    connection_info = {"url": "http://localhost/test"}

    # Mock remote execution failure
    mock_mcp_client.execute_tool.return_value = "Error executing remote MCP tool..."

    # Register fallback tool
    async def fallback_tool(**kwargs):
        return f"Fallback Success with {kwargs.get('arg1')}"

    mock_registry.register_skill("fallback_test", fallback_tool, "Fallback desc")

    # Action
    mcp_engine.import_mcp_tool(tool_def, connection_info, fallback_tool_name="fallback_test")

    wrapper_func = mock_registry.skills["test_tool_with_fallback"]
    result = await wrapper_func(arg1="val1")

    # Verify
    mock_mcp_client.execute_tool.assert_called_once_with("test_tool_with_fallback", arg1="val1")
    assert result == "Fallback Success with val1"

@pytest.mark.asyncio
async def test_mcp_engine_v5_fallback_exception(mcp_engine, mock_mcp_client, mock_registry):
    # Setup
    tool_def = {"name": "test_tool_exc"}
    connection_info = {"url": "http://localhost/test"}

    # Mock remote execution throwing exception
    mock_mcp_client.execute_tool.side_effect = Exception("Connection Refused")

    # Register fallback tool
    def fallback_tool_sync(**kwargs):
        return f"Sync Fallback Success with {kwargs.get('arg1')}"

    mock_registry.register_skill("fallback_test_sync", fallback_tool_sync, "Fallback desc")

    # Action
    mcp_engine.import_mcp_tool(tool_def, connection_info, fallback_tool_name="fallback_test_sync")

    wrapper_func = mock_registry.skills["test_tool_exc"]
    result = await wrapper_func(arg1="val1")

    # Verify
    mock_mcp_client.execute_tool.assert_called_once_with("test_tool_exc", arg1="val1")
    assert result == "Sync Fallback Success with val1"

@pytest.mark.asyncio
async def test_mcp_engine_v5_no_fallback(mcp_engine, mock_mcp_client, mock_registry):
    # Setup
    tool_def = {"name": "test_tool_no_fallback"}
    connection_info = {"url": "http://localhost/test"}

    # Mock remote execution failure
    mock_mcp_client.execute_tool.return_value = "Error executing remote MCP tool..."

    # Action
    mcp_engine.import_mcp_tool(tool_def, connection_info)

    wrapper_func = mock_registry.skills["test_tool_no_fallback"]
    result = await wrapper_func(arg1="val1")

    # Verify
    mock_mcp_client.execute_tool.assert_called_once_with("test_tool_no_fallback", arg1="val1")
    assert result == "Error executing remote MCP tool..."
