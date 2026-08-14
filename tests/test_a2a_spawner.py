"""
Tests for the A2A Spawner module.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from magda_agent.architecture.a2a_spawner import A2ASpawner
from magda_agent.integration.a2a_discovery import A2ADiscovery, AgentCard
import httpx

@pytest.fixture
def mock_discovery():
    discovery = MagicMock(spec=A2ADiscovery)

    # Setup local card
    local_card = MagicMock()
    local_card.agent_id = "local_agent"
    discovery.local_card = local_card

    # Setup security context
    security_context = MagicMock()
    security_context.generate_token.return_value = "mock_token"
    discovery.security_context = security_context

    return discovery


@pytest.fixture
def mock_agent_card():
    card = MagicMock(spec=AgentCard)
    card.agent_id = "remote_peer"
    card.name = "Remote Peer Agent"
    card.endpoints = {"mcp": "http://mock-peer/mcp"}
    return card


@pytest.mark.asyncio
async def test_spawn_a2a_subagent_success(mock_discovery, mock_agent_card):
    """Test successful delegation to a discovered peer."""
    mock_discovery.find_agents_by_capability.return_value = [mock_agent_card]

    spawner = A2ASpawner(discovery=mock_discovery)

    context = [{"role": "system", "content": "You are a test agent."}]
    task_desc = "Test task"
    capability = "code_execution"

    # Mock httpx.AsyncClient
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client_instance = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client_instance

        # Mock the post response
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"status": "Success"}}
        mock_response.raise_for_status = MagicMock()
        mock_client_instance.post.return_value = mock_response

        result = await spawner.spawn_a2a_subagent(task_desc, capability, context)

        assert result == "Delegated to Agent Remote Peer Agent: Success"
        mock_discovery.find_agents_by_capability.assert_called_once_with("code_execution")
        mock_client_instance.post.assert_called_once()

        # Verify the payload structure
        call_args = mock_client_instance.post.call_args
        assert call_args[0][0] == "http://mock-peer/mcp"

        json_payload = call_args[1]["json"]
        assert json_payload["method"] == "execute_subplan"
        assert "context" in json_payload["params"]
        assert "_a2a_handshake" in json_payload["params"]

        # Verify tracing injected headers and token
        headers = call_args[1]["headers"]
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer mock_token"


@pytest.mark.asyncio
async def test_spawn_a2a_subagent_no_agents(mock_discovery):
    """Test fallback when no agents with the required capability are found."""
    mock_discovery.find_agents_by_capability.return_value = []

    spawner = A2ASpawner(discovery=mock_discovery)

    context = [{"role": "system", "content": "system prompt"}]
    result = await spawner.spawn_a2a_subagent("Task", "missing_cap", context)

    assert result == "No agent found"


@pytest.mark.asyncio
async def test_spawn_a2a_subagent_no_endpoint(mock_discovery, mock_agent_card):
    """Test fallback when the discovered agent lacks an MCP endpoint."""
    mock_agent_card.endpoints = {}  # Remove endpoint
    mock_discovery.find_agents_by_capability.return_value = [mock_agent_card]

    spawner = A2ASpawner(discovery=mock_discovery)

    context = [{"role": "system", "content": "system prompt"}]
    result = await spawner.spawn_a2a_subagent("Task", "cap", context)

    assert result == "Agent Remote Peer Agent missing MCP endpoint"


@pytest.mark.asyncio
async def test_spawn_a2a_subagent_http_error(mock_discovery, mock_agent_card):
    """Test handling of HTTP errors during delegation."""
    mock_discovery.find_agents_by_capability.return_value = [mock_agent_card]

    spawner = A2ASpawner(discovery=mock_discovery)

    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client_instance = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client_instance

        # Simulate HTTP exception
        mock_client_instance.post.side_effect = httpx.RequestError("Connection failed")

        result = await spawner.spawn_a2a_subagent("Task", "cap", [])

        assert "failed: Connection failed" in result
