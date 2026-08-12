import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any

from magda_agent.integration.a2a_mdns import A2AMDNSDiscovery
from magda_agent.integration.a2a_delegation import A2ADelegator
from magda_agent.integration.a2a_discovery import AgentCard
from magda_agent.integration.a2a_orchestrator_mdns import A2AOrchestratorMDNS


@pytest.fixture
def mock_discovery() -> MagicMock:
    """Provides a mocked A2AMDNSDiscovery instance."""
    discovery = MagicMock(spec=A2AMDNSDiscovery)
    return discovery


@pytest.fixture
def mock_delegator() -> MagicMock:
    """Provides a mocked A2ADelegator instance."""
    delegator = MagicMock(spec=A2ADelegator)
    return delegator


@pytest.fixture
def orchestrator(mock_discovery: MagicMock, mock_delegator: MagicMock) -> A2AOrchestratorMDNS:
    """Provides an instance of A2AOrchestratorMDNS with mocked dependencies."""
    return A2AOrchestratorMDNS(mock_discovery, mock_delegator)


@pytest.mark.asyncio
async def test_execute_plan_success(orchestrator: A2AOrchestratorMDNS, mock_discovery: MagicMock, mock_delegator: MagicMock) -> None:
    """Tests successful delegation of a plan to a discovered peer."""
    plan = [
        {"id": "step_1", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "coding"}}
    ]

    mock_delegator.split_plan.return_value = [
        {"capability": "coding", "steps": plan}
    ]

    mock_agent = AgentCard(
        agent_id="agent-123",
        name="TestAgent",
        description="A test agent",
        capabilities=["coding"],
        endpoints={"rpc": "http://localhost:8000"}
    )
    mock_discovery.find_agents_by_capability.return_value = [mock_agent]
    mock_delegator.delegate_to_peer = AsyncMock(return_value="Success")

    results = await orchestrator.execute_plan(plan)

    assert results == {"step_1": "Success"}
    mock_discovery.find_agents_by_capability.assert_called_once_with("coding")
    mock_delegator.delegate_to_peer.assert_called_once_with(mock_agent, plan[0])


@pytest.mark.asyncio
async def test_execute_plan_no_agents(orchestrator: A2AOrchestratorMDNS, mock_discovery: MagicMock, mock_delegator: MagicMock) -> None:
    """Tests plan execution when no agents are found for a capability."""
    plan = [
        {"id": "step_2", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "analysis"}}
    ]

    mock_delegator.split_plan.return_value = [
        {"capability": "analysis", "steps": plan}
    ]
    mock_discovery.find_agents_by_capability.return_value = []

    results = await orchestrator.execute_plan(plan)

    assert results == {"step_2": "No agent found with capability: analysis"}
    mock_discovery.find_agents_by_capability.assert_called_once_with("analysis")


@pytest.mark.asyncio
async def test_execute_plan_with_exception(orchestrator: A2AOrchestratorMDNS, mock_discovery: MagicMock, mock_delegator: MagicMock) -> None:
    """Tests plan execution when delegation raises an exception."""
    plan = [
        {"id": "step_3", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "search"}}
    ]

    mock_delegator.split_plan.return_value = [
        {"capability": "search", "steps": plan}
    ]
    mock_agent = AgentCard(
        agent_id="agent-456",
        name="TestAgent2",
        description="A test agent",
        capabilities=["search"],
        endpoints={"rpc": "http://localhost:8001"}
    )
    mock_discovery.find_agents_by_capability.return_value = [mock_agent]

    mock_delegator.delegate_to_peer = AsyncMock(side_effect=Exception("Network error"))

    results = await orchestrator.execute_plan(plan)

    assert "Delegation error: Network error" in results["step_3"]
    mock_discovery.find_agents_by_capability.assert_called_once_with("search")


@pytest.mark.asyncio
async def test_execute_plan_empty(orchestrator: A2AOrchestratorMDNS) -> None:
    """Tests execution with an empty plan."""
    results = await orchestrator.execute_plan([])
    assert results == {}
