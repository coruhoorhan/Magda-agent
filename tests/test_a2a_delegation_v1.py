import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from magda_agent.integration.a2a_delegation_v1 import A2APeerDelegatorV1
from magda_agent.integration.a2a_discovery import AgentCard
from magda_agent.integration.a2a_discovery_v3_unique import A2ADiscoveryServiceV3Unique
import httpx

@pytest.fixture
def mock_discovery_service():
    service = MagicMock(spec=A2ADiscoveryServiceV3Unique)
    return service

@pytest.fixture
def delegator(mock_discovery_service):
    return A2APeerDelegatorV1(discovery_service=mock_discovery_service)

def test_discover_peers(delegator, mock_discovery_service):
    raw_cards = ['{"agent_id": "1"}']
    delegator.discover_peers(raw_cards)
    mock_discovery_service.parse_and_register_cards.assert_called_once_with(raw_cards)

def test_find_peer_for_capability(delegator, mock_discovery_service):
    card = AgentCard(agent_id="test-1", name="TestAgent", description="Test", capabilities=["math"], endpoints={"rpc": "http://test/rpc"})
    mock_discovery_service.find_agents_by_capability.return_value = [card]

    found_card = delegator.find_peer_for_capability("math")
    assert found_card == card
    mock_discovery_service.find_agents_by_capability.assert_called_once_with("math")

def test_find_peer_for_capability_not_found(delegator, mock_discovery_service):
    mock_discovery_service.find_agents_by_capability.return_value = []

    found_card = delegator.find_peer_for_capability("math")
    assert found_card is None

def test_format_task_payload(delegator):
    payload = delegator.format_task_payload("task-1", "math", {"a": 1, "b": 2})
    assert payload == {
        "jsonrpc": "2.0",
        "id": "task-1",
        "method": "execute_task",
        "params": {
            "capability": "math",
            "task_parameters": {"a": 1, "b": 2}
        }
    }

@pytest.mark.asyncio
async def test_delegate_task_success(delegator, mock_discovery_service):
    card = AgentCard(agent_id="test-1", name="TestAgent", description="Test", capabilities=["math"], endpoints={"rpc": "http://test/rpc"})
    mock_discovery_service.find_agents_by_capability.return_value = [card]

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": 3, "status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = await delegator.delegate_task("task-1", "math", {"a": 1, "b": 2})

        assert result == {"result": 3, "status": "success"}
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://test/rpc"
        assert kwargs["json"]["id"] == "task-1"

@pytest.mark.asyncio
async def test_delegate_task_no_peer(delegator, mock_discovery_service):
    mock_discovery_service.find_agents_by_capability.return_value = []

    result = await delegator.delegate_task("task-1", "math", {"a": 1, "b": 2})
    assert result == {"error": "No agent found for capability: math", "status": "failed"}

@pytest.mark.asyncio
async def test_delegate_task_no_rpc_endpoint(delegator, mock_discovery_service):
    card = AgentCard(agent_id="test-1", name="TestAgent", description="Test", capabilities=["math"], endpoints={})
    mock_discovery_service.find_agents_by_capability.return_value = [card]

    result = await delegator.delegate_task("task-1", "math", {"a": 1, "b": 2})
    assert result == {"error": "Agent TestAgent missing RPC endpoint", "status": "failed"}

@pytest.mark.asyncio
async def test_delegate_task_http_error(delegator, mock_discovery_service):
    card = AgentCard(agent_id="test-1", name="TestAgent", description="Test", capabilities=["math"], endpoints={"rpc": "http://test/rpc"})
    mock_discovery_service.find_agents_by_capability.return_value = [card]

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.HTTPError("Connection failed")

        result = await delegator.delegate_task("task-1", "math", {"a": 1, "b": 2})

        assert result == {"error": "Network error: Connection failed", "status": "failed"}

@pytest.mark.asyncio
async def test_delegate_task_general_error(delegator, mock_discovery_service):
    card = AgentCard(agent_id="test-1", name="TestAgent", description="Test", capabilities=["math"], endpoints={"rpc": "http://test/rpc"})
    mock_discovery_service.find_agents_by_capability.return_value = [card]

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = Exception("Unexpected error")

        result = await delegator.delegate_task("task-1", "math", {"a": 1, "b": 2})

        assert result == {"error": "Internal error: Unexpected error", "status": "failed"}
