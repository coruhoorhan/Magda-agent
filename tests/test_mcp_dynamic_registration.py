import pytest
import respx
import httpx
from typing import Any

from magda_agent.skills.mcp_dynamic_registration import MCPDynamicRegistrationPipeline
from magda_agent.skills.registry import SkillRegistry

@pytest.fixture
def mock_registry() -> SkillRegistry:
    """Provides a mocked SkillRegistry instance."""
    return SkillRegistry()

@pytest.fixture
def pipeline(mock_registry: SkillRegistry) -> MCPDynamicRegistrationPipeline:
    """Provides an MCPDynamicRegistrationPipeline instance."""
    return MCPDynamicRegistrationPipeline(registry=mock_registry)

@pytest.mark.asyncio
@respx.mock
async def test_fetch_and_register_success(pipeline: MCPDynamicRegistrationPipeline, mock_registry: SkillRegistry) -> None:
    """Tests successful fetching and registration of remote MCP tools."""
    server_url = "http://mock-mcp-server.local"

    # Mock the /tools endpoint
    mock_tools_response = {
        "tools": [
            {"name": "remote_tool_1", "description": "First remote tool"},
            {"name": "remote_tool_2", "description": "Second remote tool"}
        ]
    }
    respx.get(f"{server_url}/tools").respond(json=mock_tools_response, status_code=200)

    success = await pipeline.fetch_and_register(server_url)

    assert success is True
    assert mock_registry.has_skill("remote_tool_1")
    assert mock_registry.has_skill("remote_tool_2")

@pytest.mark.asyncio
@respx.mock
async def test_fetch_and_register_failure(pipeline: MCPDynamicRegistrationPipeline) -> None:
    """Tests failure handling when fetching remote MCP tools."""
    server_url = "http://mock-mcp-server.local"

    # Mock a server error
    respx.get(f"{server_url}/tools").respond(status_code=500)

    success = await pipeline.fetch_and_register(server_url)

    assert success is False

@pytest.mark.asyncio
@respx.mock
async def test_native_proxy_execution(pipeline: MCPDynamicRegistrationPipeline, mock_registry: SkillRegistry) -> None:
    """Tests execution of the registered native proxy function."""
    server_url = "http://mock-mcp-server.local"

    # Mock the /tools endpoint to register one tool
    mock_tools_response = {
        "tools": [
            {"name": "remote_math_add"}
        ]
    }
    respx.get(f"{server_url}/tools").respond(json=mock_tools_response, status_code=200)

    await pipeline.fetch_and_register(server_url)

    assert mock_registry.has_skill("remote_math_add")

    # Mock the tool execution endpoint
    mock_execute_response = {"result": 42}
    respx.post(f"{server_url}/tools/remote_math_add/execute").respond(json=mock_execute_response, status_code=200)

    # Call the registered native skill
    # SkillRegistry executes tools through its execute_skill method
    result = await mock_registry.execute_skill("remote_math_add", a=20, b=22)

    assert result == 42
