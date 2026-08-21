import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from magda_agent.integration.a2a_discovery import A2ADiscovery, AgentCard
from magda_agent.integration.a2a_delegation import A2ADelegator
from magda_agent.integration.a2a_orchestrator import A2AOrchestrator
from magda_agent.integration.a2a_peer_discovery_v10 import A2APeerDiscoveryServiceV10

@pytest.fixture
def a2a_discovery() -> A2ADiscovery:
    """Provides a mocked A2ADiscovery instance."""
    local_card = AgentCard("local", "local", "local", [], {})
    discovery = A2ADiscovery(local_card)
    return discovery

@pytest.fixture
def a2a_orchestrator(a2a_discovery: A2ADiscovery) -> A2AOrchestrator:
    """Provides a mocked A2AOrchestrator instance."""
    delegator = A2ADelegator(a2a_discovery)
    return A2AOrchestrator(a2a_discovery, delegator)

@pytest.fixture
def a2a_peer_discovery_v10_service(a2a_discovery: A2ADiscovery, a2a_orchestrator: A2AOrchestrator) -> A2APeerDiscoveryServiceV10:
    """Provides an A2APeerDiscoveryServiceV10 instance."""
    return A2APeerDiscoveryServiceV10(a2a_discovery, a2a_orchestrator)

@pytest.mark.asyncio
async def test_register_and_rank_peers(a2a_peer_discovery_v10_service: A2APeerDiscoveryServiceV10, a2a_discovery: A2ADiscovery) -> None:
    """Tests that peers are correctly registered."""
    cards = [
        AgentCard("peer1", "Peer 1", "Desc 1", ["cap1"], {"mcp": "url1"}),
        AgentCard("peer2", "Peer 2", "Desc 2", ["cap2"], {"mcp": "url2"})
    ]
    await a2a_peer_discovery_v10_service.register_and_rank_peers(cards)

    assert len(a2a_discovery._discovered_agents) == 2
    assert "peer1" in a2a_discovery._discovered_agents
    assert "peer2" in a2a_discovery._discovered_agents

def test_map_task_to_peer(a2a_peer_discovery_v10_service: A2APeerDiscoveryServiceV10, a2a_discovery: A2ADiscovery) -> None:
    """Tests mapping tasks to peers."""
    card1 = AgentCard("peer1", "Peer 1", "Desc 1", ["cap1"], {"mcp": "url1"})
    a2a_discovery._register_agent(card1)

    matched_peer = a2a_peer_discovery_v10_service.map_task_to_peer({"required_capability": "cap1"})
    assert matched_peer == card1

    unmatched_peer = a2a_peer_discovery_v10_service.map_task_to_peer({"required_capability": "cap2"})
    assert unmatched_peer is None

@pytest.mark.asyncio
@patch('httpx.AsyncClient.get', new_callable=AsyncMock)
async def test_discover_peers_via_mdns(mock_get: AsyncMock, a2a_peer_discovery_v10_service: A2APeerDiscoveryServiceV10, a2a_discovery: A2ADiscovery) -> None:
    """Tests discovering peers via mDNS."""
    card_json = AgentCard("peer3", "Peer 3", "Desc 3", ["cap3"], {"mcp": "url3"}).to_json()

    mock_response = MagicMock()
    mock_response.json.return_value = {"cards": [card_json]}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    discovered_cards = await a2a_peer_discovery_v10_service.discover_peers_via_mdns("http://mdns-endpoint")

    assert len(discovered_cards) == 1
    assert discovered_cards[0].agent_id == "peer3"
    assert "peer3" in a2a_discovery._discovered_agents

@pytest.mark.asyncio
@patch('httpx.AsyncClient.get', new_callable=AsyncMock)
async def test_discover_peers_via_mdns_error(mock_get: AsyncMock, a2a_peer_discovery_v10_service: A2APeerDiscoveryServiceV10) -> None:
    """Tests handling errors when discovering peers via mDNS."""
    mock_get.side_effect = Exception("Simulated network error")

    discovered_cards = await a2a_peer_discovery_v10_service.discover_peers_via_mdns("http://mdns-endpoint")
    assert len(discovered_cards) == 0
