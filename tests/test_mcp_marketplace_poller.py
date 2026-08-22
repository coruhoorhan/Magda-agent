import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from magda_agent.integration.mcp_marketplace_poller import MCPMarketplacePoller
from magda_agent.skills.mcp_registry_v5 import MCPRegistryV5
from magda_agent.scheduler.cron_scheduler_v2 import CronSchedulerV2

@pytest.fixture
def registry():
    return MagicMock(spec=MCPRegistryV5)

@pytest.fixture
def scheduler():
    return MagicMock(spec=CronSchedulerV2)

@pytest.mark.asyncio
async def test_schedule_sync(registry, scheduler):
    """Test that the poller correctly schedules the task with the cron scheduler."""
    poller = MCPMarketplacePoller(registry, scheduler, cron_expr="*/5 * * * *")
    poller.schedule_sync()

    scheduler.add_task.assert_called_once_with(
        "mcp_marketplace_sync",
        "*/5 * * * *",
        poller.sync_marketplace
    )
    await poller.close()

@pytest.mark.asyncio
@patch("magda_agent.integration.mcp_marketplace_poller.httpx.AsyncClient.get")
async def test_sync_marketplace_success(mock_get, registry, scheduler):
    """Test successful fetching and registration of valid tools."""
    poller = MCPMarketplacePoller(registry, scheduler)

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()

    # In httpx, response.json() is NOT async. So we just need a MagicMock that returns our list.
    mock_response.json = MagicMock(return_value=[
        {
            "name": "valid_tool",
            "description": "Does something",
            "parameters": {"type": "object", "properties": {}}
        },
        {
            "name": "invalid_tool",
            "description": "Missing parameters"
        }
    ])

    mock_get.return_value = mock_response

    # Simulate registry success for the valid tool
    registry.load_tool.return_value = True

    await poller.sync_marketplace()

    # Verify the network call was made
    mock_get.assert_called_once_with(poller.marketplace_url, timeout=10.0)
    mock_response.raise_for_status.assert_called_once()

    # Verify the registry was called ONLY for the valid tool
    registry.load_tool.assert_called_once_with(mock_response.json()[0])

    await poller.close()


@pytest.mark.asyncio
@patch("magda_agent.integration.mcp_marketplace_poller.httpx.AsyncClient.get")
async def test_sync_marketplace_network_error(mock_get, registry, scheduler):
    """Test that network errors are caught gracefully."""
    poller = MCPMarketplacePoller(registry, scheduler)

    # Simulate a network error
    mock_get.side_effect = httpx.RequestError("Network unreachable", request=MagicMock())

    await poller.sync_marketplace()

    # Verify no tools were registered
    registry.load_tool.assert_not_called()

    await poller.close()


@pytest.mark.asyncio
@patch("magda_agent.integration.mcp_marketplace_poller.httpx.AsyncClient.get")
async def test_sync_marketplace_invalid_json_format(mock_get, registry, scheduler):
    """Test when API returns dict instead of list."""
    poller = MCPMarketplacePoller(registry, scheduler)

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    # Provide a dict instead of a list, fix json() to be a regular mock
    mock_response.json = MagicMock(return_value={"error": "Not found"})
    mock_get.return_value = mock_response

    await poller.sync_marketplace()

    # Verify no tools were registered
    registry.load_tool.assert_not_called()

    await poller.close()
