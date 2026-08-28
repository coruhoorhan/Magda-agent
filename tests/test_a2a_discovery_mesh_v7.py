import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from magda_agent.integration.a2a_discovery import A2ADiscovery, AgentCard
from magda_agent.integration.a2a_discovery_mesh_v7 import A2ADiscoveryMeshV7


@pytest.fixture
def local_card() -> AgentCard:
    """Fixture providing a mock local AgentCard."""
    return AgentCard(
        agent_id="local-agent-123",
        name="LocalAgent",
        description="A local test agent",
        capabilities=["test", "local"],
        endpoints={"gossip": "http://localhost:8000/gossip"}
    )


@pytest.fixture
def peer_card() -> AgentCard:
    """Fixture providing a mock peer AgentCard."""
    return AgentCard(
        agent_id="peer-agent-456",
        name="PeerAgent",
        description="A peer test agent",
        capabilities=["test", "peer"],
        endpoints={"gossip": "http://peer:8000/gossip"}
    )


@pytest.fixture
def discovery(local_card: AgentCard, peer_card: AgentCard) -> A2ADiscovery:
    """Fixture providing an A2ADiscovery instance with one registered peer."""
    d = A2ADiscovery(local_card)
    d._register_agent(peer_card)
    return d


@pytest.fixture
def mesh(discovery: A2ADiscovery) -> A2ADiscoveryMeshV7:
    """Fixture providing the A2ADiscoveryMeshV7 instance to test."""
    return A2ADiscoveryMeshV7(discovery)


def test_aggregate_cards(mesh: A2ADiscoveryMeshV7, local_card: AgentCard, peer_card: AgentCard) -> None:
    """Test that aggregate_cards returns both local and peer cards."""
    cards = mesh.aggregate_cards()
    assert len(cards) == 2
    agent_ids = [c.agent_id for c in cards]
    assert local_card.agent_id in agent_ids
    assert peer_card.agent_id in agent_ids


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_broadcast_gossip_network(mock_post: AsyncMock, mesh: A2ADiscoveryMeshV7) -> None:
    """Test that broadcast_gossip sends HTTP POST requests."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    endpoints = ["http://test1:8000/gossip", "http://test2:8000/gossip"]
    await mesh.broadcast_gossip(endpoints)

    assert mock_post.call_count == 2
    # Check that it tried to post to both endpoints
    call_args_list = mock_post.call_args_list
    called_urls = [call[0][0] for call in call_args_list]
    assert "http://test1:8000/gossip" in called_urls
    assert "http://test2:8000/gossip" in called_urls


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_broadcast_gossip_local_queue(mock_post: AsyncMock, mesh: A2ADiscoveryMeshV7) -> None:
    """Test that broadcast_gossip enqueues the cards for local reception."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    await mesh.broadcast_gossip(["http://dummy"])

    assert not mesh.local_broadcast_queue.empty()
    queued_data = await mesh.local_broadcast_queue.get()

    assert isinstance(queued_data, list)
    assert len(queued_data) == 2  # Local + 1 peer


def test_receive_gossip(mesh: A2ADiscoveryMeshV7) -> None:
    """Test that receive_gossip registers new agents properly."""
    new_card_data = {
        "agent_id": "new-agent-789",
        "name": "NewAgent",
        "description": "Brand new agent",
        "capabilities": ["new"],
        "endpoints": {"gossip": "http://new:8000/gossip"},
        "protocol_version": "2.0"
    }

    assert "new-agent-789" not in mesh.discovery._discovered_agents

    mesh.receive_gossip([new_card_data])

    assert "new-agent-789" in mesh.discovery._discovered_agents
    registered_card = mesh.discovery._discovered_agents["new-agent-789"]
    assert registered_card.name == "NewAgent"


def test_receive_gossip_ignores_self(mesh: A2ADiscoveryMeshV7, local_card: AgentCard) -> None:
    """Test that receive_gossip ignores the local agent's card."""
    # Temporarily remove local agent from discovered if it somehow got there
    # (it shouldn't be in discovered_agents, only in local_card, but let's be sure)
    mesh.discovery._discovered_agents.pop(local_card.agent_id, None)

    from dataclasses import asdict
    self_card_data = asdict(local_card)

    mesh.receive_gossip([self_card_data])

    # Still shouldn't be in discovered_agents
    assert local_card.agent_id not in mesh.discovery._discovered_agents
