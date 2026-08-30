import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from magda_agent.skills.mcp_registry import MCPRegistry
from magda_agent.skills.mcp_registry_sync_v11 import MCPRegistrySyncV11
from unittest.mock import AsyncMock

@pytest.fixture
def registry() -> MCPRegistry:
    return MCPRegistry()

@pytest.mark.asyncio
async def test_sync_once_success_add_tool(registry: MCPRegistry) -> None:
    sync = MCPRegistrySyncV11(registry, "http://mock-mcp-server", sync_interval=60)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "tools": [
            {
                "name": "test_tool_1",
                "description": "A test tool"
            }
        ]
    }
    mock_response.raise_for_status.return_value = None

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        await sync.sync_once()

    assert "test_tool_1" in registry.list_tools()

@pytest.mark.asyncio
async def test_sync_once_success_remove_tool(registry: MCPRegistry) -> None:
    # Pre-load a tool
    registry.load_tool({
        "name": "test_tool_to_remove",
        "description": "Will be removed"
    })
    assert "test_tool_to_remove" in registry.list_tools()

    sync = MCPRegistrySyncV11(registry, "http://mock-mcp-server", sync_interval=60)

    # Remote server returns empty list of tools
    mock_response = MagicMock()
    mock_response.json.return_value = {"tools": []}
    mock_response.raise_for_status.return_value = None

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        await sync.sync_once()

    assert "test_tool_to_remove" not in registry.list_tools()

@pytest.mark.asyncio
async def test_sync_once_success_update_tool(registry: MCPRegistry) -> None:
    # Pre-load a tool
    registry.load_tool({
        "name": "test_tool_to_update",
        "description": "Old description"
    })

    sync = MCPRegistrySyncV11(registry, "http://mock-mcp-server", sync_interval=60)

    # Remote server returns updated tool
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "tools": [
            {
                "name": "test_tool_to_update",
                "description": "New description"
            }
        ]
    }
    mock_response.raise_for_status.return_value = None

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        await sync.sync_once()

    tool = registry.get_tool("test_tool_to_update")
    assert tool["description"] == "New description"

@pytest.mark.asyncio
async def test_sync_once_http_request_error(registry: MCPRegistry) -> None:
    registry.load_tool({
        "name": "test_tool",
        "description": "Should remain"
    })

    sync = MCPRegistrySyncV11(registry, "http://mock-mcp-server", sync_interval=60)

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.side_effect = httpx.RequestError("Network error", request=MagicMock())

    with patch("httpx.AsyncClient", return_value=mock_client):
        await sync.sync_once()

    # Tool should still be there because sync failed
    assert "test_tool" in registry.list_tools()

@pytest.mark.asyncio
async def test_sync_once_http_status_error(registry: MCPRegistry) -> None:
    registry.load_tool({
        "name": "test_tool",
        "description": "Should remain"
    })

    sync = MCPRegistrySyncV11(registry, "http://mock-mcp-server", sync_interval=60)

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("404 Not Found", request=MagicMock(), response=mock_response)
    mock_response.status_code = 404

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        await sync.sync_once()

    # Tool should still be there because sync failed
    assert "test_tool" in registry.list_tools()

@pytest.mark.asyncio
async def test_sync_once_invalid_json(registry: MCPRegistry) -> None:
    sync = MCPRegistrySyncV11(registry, "http://mock-mcp-server", sync_interval=60)

    mock_response = MagicMock()
    mock_response.json.return_value = ["not", "a", "dict"]
    mock_response.raise_for_status.return_value = None

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        await sync.sync_once()

    assert len(registry.list_tools()) == 0

@pytest.mark.asyncio
async def test_start_stop() -> None:
    registry = MCPRegistry()
    sync = MCPRegistrySyncV11(registry, "http://mock-mcp-server", sync_interval=1)

    with patch.object(sync, 'sync_once', new_callable=AsyncMock) as mock_sync_once:
        sync.start()
        assert sync._running is True
        assert sync._task is not None

        # let it run one cycle
        await asyncio.sleep(0.1)

        await sync.stop()
        assert sync._running is False
        assert sync._task is None

        mock_sync_once.assert_called()

@pytest.mark.asyncio
async def test_execute_tool(registry):

    sync_instance = MCPRegistrySyncV11(registry, "http://example.com")
    registry.execute_tool = AsyncMock(return_value="mock_result")

    # Mock the audit trail
    sync_instance.mcp_audit.log_mcp_invocation = AsyncMock()

    result = await sync_instance.execute_tool("my_tool", arg1="val1")

    assert result == "mock_result"
    registry.execute_tool.assert_called_once_with("my_tool", arg1="val1")
    sync_instance.mcp_audit.log_mcp_invocation.assert_called_once()
    args, kwargs = sync_instance.mcp_audit.log_mcp_invocation.call_args
    assert kwargs["server_name"] == "http://example.com"
    assert kwargs["tool_name"] == "my_tool"
    assert kwargs["arguments"] == {"arg1": "val1"}
    assert kwargs["result"] == "mock_result"
    assert kwargs["status"] == "success"

@pytest.mark.asyncio
async def test_execute_tool_error(registry):

    sync_instance = MCPRegistrySyncV11(registry, "http://example.com")
    registry.execute_tool = AsyncMock(side_effect=Exception("mock_error"))

    sync_instance.mcp_audit.log_mcp_invocation = AsyncMock()

    with pytest.raises(Exception):
        await sync_instance.execute_tool("my_tool", arg1="val1")

    sync_instance.mcp_audit.log_mcp_invocation.assert_called_once()
    args, kwargs = sync_instance.mcp_audit.log_mcp_invocation.call_args
    assert kwargs["status"] == "error"
    assert kwargs["result"] == {"error": "mock_error"}
