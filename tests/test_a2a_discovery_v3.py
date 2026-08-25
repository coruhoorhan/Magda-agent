import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.integration.a2a_discovery_v3 import AgentCardV3, A2ADiscoveryServiceV3Unique

@pytest.mark.asyncio
async def test_broadcast_agent_card():
    mock_network = AsyncMock()
    service = A2ADiscoveryServiceV3Unique(network_interface=mock_network)

    card = AgentCardV3(
        agent_id="agent-123",
        capabilities=["search", "calculate"],
        endpoints={"rpc": "http://localhost:8080/rpc"}
    )

    await service.broadcast_agent_card(card)

    mock_network.broadcast.assert_called_once_with({
        "agent_id": "agent-123",
        "capabilities": ["search", "calculate"],
        "endpoints": {"rpc": "http://localhost:8080/rpc"}
    })

def test_receive_card():
    mock_network = AsyncMock()
    service = A2ADiscoveryServiceV3Unique(network_interface=mock_network)

    payload = {
        "agent_id": "agent-456",
        "capabilities": ["translate"],
        "endpoints": {"rpc": "http://peer:8080/rpc"}
    }

    service.receive_card(payload)

    assert "agent-456" in service.discovered_agents
    assert service.discovered_agents["agent-456"] == payload
