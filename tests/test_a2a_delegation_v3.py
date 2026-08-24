import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from magda_agent.integration.a2a_delegation_v3 import A2ADelegatorV3
from magda_agent.integration.a2a_cards import AgentCardV3

@pytest.mark.asyncio
async def test_delegate_task_success():
    """Test successful task delegation to a discovered peer."""
    delegator = A2ADelegatorV3()

    mock_agent = MagicMock(spec=AgentCardV3)
    mock_agent.name = "TestAgent"
    mock_agent.endpoints = {"rpc": "http://test-agent:8000/rpc"}

    with patch.object(delegator.discovery_service, 'find_agents_by_capability', return_value=[mock_agent]):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "success", "result": "task completed"}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await delegator.delegate_task("task-123", "data_analysis", {"data": [1, 2, 3]})

            assert result == {"status": "success", "result": "task completed"}
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert args[0] == "http://test-agent:8000/rpc"
            assert kwargs["json"]["id"] == "task-123"
            assert kwargs["json"]["params"]["capability"] == "data_analysis"

@pytest.mark.asyncio
async def test_delegate_task_no_agents():
    """Test task delegation when no agents are found."""
    delegator = A2ADelegatorV3()

    with patch.object(delegator.discovery_service, 'find_agents_by_capability', return_value=[]):
        result = await delegator.delegate_task("task-123", "data_analysis", {"data": [1, 2, 3]})

        assert result == {"error": "No agent found for capability: data_analysis", "status": "failed"}

@pytest.mark.asyncio
async def test_delegate_task_no_rpc_endpoint():
    """Test task delegation when the agent is missing an RPC endpoint."""
    delegator = A2ADelegatorV3()

    mock_agent = MagicMock(spec=AgentCardV3)
    mock_agent.name = "TestAgent"
    mock_agent.endpoints = {}  # Missing RPC endpoint

    with patch.object(delegator.discovery_service, 'find_agents_by_capability', return_value=[mock_agent]):
        result = await delegator.delegate_task("task-123", "data_analysis", {"data": [1, 2, 3]})

        assert result == {"error": "Agent TestAgent missing RPC endpoint", "status": "failed"}

@pytest.mark.asyncio
async def test_delegate_task_network_error():
    """Test task delegation handling network errors."""
    delegator = A2ADelegatorV3()

    mock_agent = MagicMock(spec=AgentCardV3)
    mock_agent.name = "TestAgent"
    mock_agent.endpoints = {"rpc": "http://test-agent:8000/rpc"}

    with patch.object(delegator.discovery_service, 'find_agents_by_capability', return_value=[mock_agent]):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_request = MagicMock(spec=httpx.Request)
            mock_post.side_effect = httpx.RequestError("Connection timeout", request=mock_request)

            result = await delegator.delegate_task("task-123", "data_analysis", {"data": [1, 2, 3]})

            assert result == {"error": "Network error: Connection timeout", "status": "failed"}

def test_format_task_payload():
    """Test that the payload is formatted correctly."""
    delegator = A2ADelegatorV3()
    payload = delegator.format_task_payload("task-123", "data_analysis", {"data": [1, 2, 3]})

    assert payload == {
        "jsonrpc": "2.0",
        "id": "task-123",
        "method": "execute_task",
        "params": {
            "capability": "data_analysis",
            "task_parameters": {"data": [1, 2, 3]}
        }
    }

def test_discover_peers():
    """Test discover peers registers cards via discovery service."""
    delegator = A2ADelegatorV3()
    raw_cards = ['{"name": "Agent1"}', '{"name": "Agent2"}']

    with patch.object(delegator.discovery_service, 'parse_and_register_cards') as mock_register:
        delegator.discover_peers(raw_cards)
        mock_register.assert_called_once_with(raw_cards)
