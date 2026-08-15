import pytest
import json
from magda_agent.integration.a2a_discovery_v6 import AgentCardV6, A2ADiscoveryV6


@pytest.fixture
def local_card():
    return AgentCardV6(
        agent_id="agent-123",
        name="LocalPlanner",
        description="A local planning agent",
        capabilities=["planning", "coordination"],
        endpoints={"mcp": "http://localhost:8080/mcp"}
    )


@pytest.fixture
def remote_card():
    return AgentCardV6(
        agent_id="agent-456",
        name="RemoteWorker",
        description="A remote execution agent",
        capabilities=["execution", "search"],
        endpoints={"mcp": "http://remote:8080/mcp"}
    )


@pytest.fixture
def discovery_service(local_card):
    return A2ADiscoveryV6(local_card=local_card)


@pytest.mark.asyncio
async def test_broadcast_card(discovery_service, local_card):
    """Test broadcasting the local agent card."""
    broadcast_json = await discovery_service.broadcast_card()
    card = AgentCardV6.from_json(broadcast_json)

    assert card.agent_id == local_card.agent_id
    assert card.name == local_card.name
    assert card.capabilities == local_card.capabilities
    assert card.protocol_version == "6.0"


@pytest.mark.asyncio
async def test_fetch_cards(discovery_service, remote_card):
    """Test fetching and parsing remote agent cards."""
    remote_json = remote_card.to_json()

    # Simulate fetching
    await discovery_service.fetch_cards(mock_network_cards=[remote_json])

    # Check if registered
    fetched_card = discovery_service.get_agent_by_id(remote_card.agent_id)
    assert fetched_card is not None
    assert fetched_card.name == remote_card.name
    assert fetched_card.endpoints == remote_card.endpoints


@pytest.mark.asyncio
async def test_fetch_cards_invalid_json(discovery_service):
    """Test fetching handles invalid JSON gracefully."""
    invalid_json = '{"agent_id": "bad'
    await discovery_service.fetch_cards(mock_network_cards=[invalid_json])

    # Should not throw exception and should not register anything
    assert len(discovery_service._discovered_agents) == 0


@pytest.mark.asyncio
async def test_find_agents_by_capability(discovery_service, remote_card):
    """Test finding registered agents by capability."""
    await discovery_service.fetch_cards(mock_network_cards=[remote_card.to_json()])

    # Find by existing capability
    exec_agents = discovery_service.find_agents_by_capability("execution")
    assert len(exec_agents) == 1
    assert exec_agents[0].agent_id == remote_card.agent_id

    # Find by non-existent capability
    unknown_agents = discovery_service.find_agents_by_capability("flying")
    assert len(unknown_agents) == 0
