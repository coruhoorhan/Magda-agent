import pytest
import httpx
from unittest.mock import AsyncMock, patch
from dataclasses import asdict

from magda_agent.integration.a2a_discovery import A2ADiscovery, AgentCard
from magda_agent.integration.a2a_discovery_mesh import A2ADiscoveryMesh


@pytest.fixture
def local_card():
    return AgentCard(
        agent_id="agent-local-1",
        name="Local Agent",
        description="I am the local agent",
        capabilities=["chat", "planning"],
        endpoints={"gossip": "http://localhost:8000/gossip"}
    )


@pytest.fixture
def peer_card():
    return AgentCard(
        agent_id="agent-peer-2",
        name="Peer Agent",
        description="I am a remote peer",
        capabilities=["coding"],
        endpoints={"gossip": "http://192.168.1.10:8000/gossip"}
    )


@pytest.fixture
def discovery_service(local_card, peer_card):
    service = A2ADiscovery(local_card=local_card)
    service._register_agent(peer_card)
    return service


@pytest.fixture
def mesh(discovery_service):
    return A2ADiscoveryMesh(discovery=discovery_service)


def test_aggregate_cards(mesh, local_card, peer_card):
    cards = mesh.aggregate_cards()
    assert len(cards) == 2
    assert local_card in cards
    assert peer_card in cards


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_broadcast_gossip_success(mock_post, mesh, local_card, peer_card):
    # Setup mock response
    mock_post.return_value.raise_for_status = AsyncMock()

    endpoints = ["http://192.168.1.10:8000/gossip", "http://192.168.1.11:8000/gossip"]
    await mesh.broadcast_gossip(endpoints)

    assert mock_post.call_count == 2

    # Check what was sent
    expected_data = [asdict(local_card), asdict(peer_card)]

    # First call args
    first_call_args, first_call_kwargs = mock_post.call_args_list[0]
    assert first_call_args[0] == "http://192.168.1.10:8000/gossip"
    assert first_call_kwargs["json"] == expected_data

    # Second call args
    second_call_args, second_call_kwargs = mock_post.call_args_list[1]
    assert second_call_args[0] == "http://192.168.1.11:8000/gossip"
    assert second_call_kwargs["json"] == expected_data


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_broadcast_gossip_failure(mock_post, mesh):
    # Setup mock to simulate a network error
    mock_post.side_effect = httpx.RequestError("Network error", request=AsyncMock())

    endpoints = ["http://192.168.1.10:8000/gossip"]
    # Should not raise an exception, just log the error
    await mesh.broadcast_gossip(endpoints)

    mock_post.assert_called_once()


def test_receive_gossip_new_peer(mesh):
    new_peer_data = {
        "agent_id": "agent-peer-3",
        "name": "New Peer",
        "description": "Just joined the mesh",
        "capabilities": ["search"],
        "endpoints": {"gossip": "http://192.168.1.12:8000/gossip"},
        "protocol_version": "2.0"
    }

    mesh.receive_gossip([new_peer_data])

    # Verify it was registered in the discovery service
    discovered_agent = mesh.discovery.get_agent_by_id("agent-peer-3")
    assert discovered_agent is not None
    assert discovered_agent.name == "New Peer"
    assert "search" in discovered_agent.capabilities


def test_receive_gossip_ignores_local_card(mesh, local_card):
    # Receive gossip containing our own card
    gossip_data = [asdict(local_card)]

    # To check if it was ignored, we can monitor the call to _register_agent
    # However, since _register_agent is a simple method, we just verify the state doesn't break
    with patch.object(mesh.discovery, '_register_agent') as mock_register:
        mesh.receive_gossip(gossip_data)

        # Should not have called register for our own card
        mock_register.assert_not_called()


def test_receive_gossip_invalid_data(mesh):
    invalid_data = [
        {"not_an_agent_id": "missing_required_fields"}
    ]

    # Should handle the exception internally and not raise
    mesh.receive_gossip(invalid_data)

    # Should still just have the initial 1 peer registered during fixture setup
    assert len(mesh.discovery._discovered_agents) == 1
