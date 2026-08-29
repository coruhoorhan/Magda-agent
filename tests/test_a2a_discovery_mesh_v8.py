import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import asdict

from magda_agent.integration.a2a_discovery import A2ADiscovery, AgentCard
from magda_agent.integration.a2a_discovery_mesh_v8 import A2ADiscoveryMeshV8

@pytest.fixture
def local_card():
    return AgentCard(
        agent_id="local-1",
        name="Local Agent",
        description="Local agent for testing",
        capabilities=["test-cap-1"],
        endpoints={"gossip": "http://localhost:8000"}
    )

@pytest.fixture
def remote_card():
    return AgentCard(
        agent_id="remote-1",
        name="Remote Agent",
        description="Remote agent for testing",
        capabilities=["test-cap-2"],
        endpoints={"gossip": "http://remote:8000"}
    )

@pytest.fixture
def discovery(local_card, remote_card):
    disc = A2ADiscovery(local_card=local_card)
    disc._register_agent(remote_card)
    return disc

@pytest.fixture
def mesh(discovery):
    return A2ADiscoveryMeshV8(discovery)

def test_aggregate_cards(mesh, local_card, remote_card):
    cards = mesh.aggregate_cards()
    assert len(cards) == 2
    assert local_card in cards
    assert remote_card in cards

@pytest.mark.asyncio
async def test_broadcast_gossip_success(mesh, local_card, remote_card):
    endpoints = ["http://test:8000/gossip"]
    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = mock_client.return_value.__aenter__.return_value
        mock_post = mock_instance.post
        mock_post.return_value.raise_for_status = MagicMock()

        await mesh.broadcast_gossip(endpoints)

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == endpoints[0]
        assert "json" in kwargs
        assert len(kwargs["json"]) == 2

        # Test local broadcast queue
        assert mesh.local_broadcast_queue.qsize() == 1
        queue_items = await mesh.local_broadcast_queue.get()
        assert len(queue_items) == 2
        assert queue_items[0]["agent_id"] == "local-1"

@pytest.mark.asyncio
async def test_broadcast_gossip_failure(mesh, local_card, remote_card):
    endpoints = ["http://test:8000/gossip"]
    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = mock_client.return_value.__aenter__.return_value
        mock_post = mock_instance.post
        mock_post.side_effect = Exception("Network error")

        # Should not raise exception
        await mesh.broadcast_gossip(endpoints)
        mock_post.assert_called_once()

        # Should still enqueue locally
        assert mesh.local_broadcast_queue.qsize() == 1

def test_receive_gossip(mesh, local_card):
    # Create a new agent card data
    new_card_data = {
        "agent_id": "new-agent-1",
        "name": "New Agent",
        "description": "New agent",
        "capabilities": ["new-cap"],
        "endpoints": {},
        "protocol_version": "2.0"
    }

    # Receive our own card + new card
    cards_data = [asdict(local_card), new_card_data]

    mesh.receive_gossip(cards_data)

    # Should only register the new agent, not local one again
    assert "new-agent-1" in mesh.discovery._discovered_agents
    assert mesh.discovery.get_agent_by_id("new-agent-1").name == "New Agent"
