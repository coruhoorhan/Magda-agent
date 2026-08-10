import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from magda_agent.integration.a2a_discovery import A2ADiscovery, AgentCard
from magda_agent.integration.a2a_delegation import A2ADelegator
from magda_agent.integration.a2a_orchestrator_v2 import A2AOrchestratorV2

@pytest.fixture
def mock_agent_card():
    return AgentCard(
        agent_id="test-agent-123",
        name="TestAgent",
        description="A test agent",
        capabilities=["coding", "analysis"],
        endpoints={"mcp": "http://localhost:9000"}
    )

@pytest.fixture
def a2a_discovery(mock_agent_card):
    local_card = AgentCard("local", "local", "local", [], {})
    discovery = A2ADiscovery(local_card)
    discovery._discovered_agents[mock_agent_card.agent_id] = mock_agent_card
    discovery._capability_index["coding"] = [mock_agent_card.agent_id]
    discovery._capability_index["analysis"] = [mock_agent_card.agent_id]
    return discovery

@pytest.fixture
def a2a_delegator(a2a_discovery):
    return A2ADelegator(a2a_discovery)

@pytest.fixture
def a2a_orchestrator_v2(a2a_discovery, a2a_delegator):
    return A2AOrchestratorV2(a2a_discovery, a2a_delegator)

@pytest.mark.asyncio
async def test_route_task_success(a2a_orchestrator_v2, mock_agent_card):
    task_context = {"task": "Write hello world"}
    required_capability = "coding"

    a2a_orchestrator_v2.delegator.delegate_to_peer = AsyncMock(return_value="Delegated to Peer Agent TestAgent: Success")

    result = await a2a_orchestrator_v2.route_task(task_context, required_capability)

    assert result == "Delegated to Peer Agent TestAgent: Success"
    a2a_orchestrator_v2.delegator.delegate_to_peer.assert_called_once_with(mock_agent_card, task_context)

@pytest.mark.asyncio
async def test_route_task_no_agents_found(a2a_orchestrator_v2):
    task_context = {"task": "Draw a cat"}
    required_capability = "drawing"

    a2a_orchestrator_v2.delegator.delegate_to_peer = AsyncMock()

    result = await a2a_orchestrator_v2.route_task(task_context, required_capability)

    assert result == "No agent found with capability: drawing"
    a2a_orchestrator_v2.delegator.delegate_to_peer.assert_not_called()
