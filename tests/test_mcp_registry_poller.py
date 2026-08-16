import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.skills.mcp_registry import MCPRegistry
from magda_agent.skills.mcp_registry_poller import MCPRegistryPoller

@pytest.fixture
def mock_registry() -> MagicMock:
    """Provides a mocked MCPRegistry."""
    registry = MagicMock(spec=MCPRegistry)
    registry.list_tools.return_value = []
    registry.load_tool.return_value = True
    registry.reload_tool.return_value = True
    registry.unload_tool.return_value = True
    return registry

@pytest.mark.asyncio
async def test_poller_starts_and_stops(mock_registry: MagicMock) -> None:
    """Test that the poller can start and stop gracefully."""
    poller = MCPRegistryPoller(registry=mock_registry, mcp_server_url="http://mock-server", poll_interval=1)
    poller.start()
    assert poller._running is True
    assert poller._task is not None
    await poller.stop()
    assert poller._running is False
    assert poller._task is None

@pytest.mark.asyncio
@patch("magda_agent.skills.mcp_registry_poller.httpx.AsyncClient")
async def test_poll_once_success(mock_client_cls: MagicMock, mock_registry: MagicMock) -> None:
    """Test that the poller successfully polls and registers new tools."""
    mock_client = AsyncMock()
    mock_client_cls.return_value.__aenter__.return_value = mock_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "tools": [
            {"name": "tool_one", "description": "desc one"},
            {"name": "tool_two", "description": "desc two"}
        ]
    }
    mock_client.get.return_value = mock_response

    mock_registry.list_tools.return_value = ["tool_one"]

    poller = MCPRegistryPoller(registry=mock_registry, mcp_server_url="http://mock-server", poll_interval=1)
    await poller.poll_once()

    mock_client.get.assert_called_once_with("http://mock-server/tools")

    # tool_two is new, so it should be loaded
    mock_registry.load_tool.assert_called_once_with({"name": "tool_two", "description": "desc two"})

    # tool_one exists, so it should be reloaded
    mock_registry.reload_tool.assert_called_once_with({"name": "tool_one", "description": "desc one"})

    # no tools to unload
    mock_registry.unload_tool.assert_not_called()

@pytest.mark.asyncio
@patch("magda_agent.skills.mcp_registry_poller.httpx.AsyncClient")
async def test_poll_once_unload_tools(mock_client_cls: MagicMock, mock_registry: MagicMock) -> None:
    """Test that the poller unloads tools that are no longer on the server."""
    mock_client = AsyncMock()
    mock_client_cls.return_value.__aenter__.return_value = mock_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "tools": [
            {"name": "tool_one", "description": "desc one"}
        ]
    }
    mock_client.get.return_value = mock_response

    mock_registry.list_tools.return_value = ["tool_one", "tool_two"]

    poller = MCPRegistryPoller(registry=mock_registry, mcp_server_url="http://mock-server", poll_interval=1)
    await poller.poll_once()

    mock_client.get.assert_called_once_with("http://mock-server/tools")

    # tool_one exists, so it should be reloaded
    mock_registry.reload_tool.assert_called_once_with({"name": "tool_one", "description": "desc one"})

    # tool_two is not on the server, so it should be unloaded
    mock_registry.unload_tool.assert_called_once_with("tool_two")

@pytest.mark.asyncio
@patch("magda_agent.skills.mcp_registry_poller.httpx.AsyncClient")
async def test_poll_once_invalid_response(mock_client_cls: MagicMock, mock_registry: MagicMock) -> None:
    """Test that the poller handles invalid responses gracefully."""
    mock_client = AsyncMock()
    mock_client_cls.return_value.__aenter__.return_value = mock_client
    mock_response = MagicMock()
    mock_response.json.return_value = {"not_tools": []} # Invalid schema
    mock_client.get.return_value = mock_response

    poller = MCPRegistryPoller(registry=mock_registry, mcp_server_url="http://mock-server", poll_interval=1)
    await poller.poll_once()

    # Nothing should change
    mock_registry.load_tool.assert_not_called()
    mock_registry.reload_tool.assert_not_called()
    mock_registry.unload_tool.assert_not_called()

@pytest.mark.asyncio
@patch("magda_agent.skills.mcp_registry_poller.httpx.AsyncClient")
async def test_poll_once_http_error(mock_client_cls: MagicMock, mock_registry: MagicMock) -> None:
    """Test that the poller handles HTTP errors gracefully."""
    mock_client = AsyncMock()
    mock_client_cls.return_value.__aenter__.return_value = mock_client
    mock_client.get.side_effect = Exception("HTTP Error")

    poller = MCPRegistryPoller(registry=mock_registry, mcp_server_url="http://mock-server", poll_interval=1)
    await poller.poll_once()

    # Nothing should change
    mock_registry.load_tool.assert_not_called()
    mock_registry.reload_tool.assert_not_called()
    mock_registry.unload_tool.assert_not_called()
