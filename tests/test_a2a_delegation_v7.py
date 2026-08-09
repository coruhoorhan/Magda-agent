import pytest
import asyncio
import httpx
from unittest.mock import patch, AsyncMock, MagicMock
from magda_agent.integration.a2a_delegation_v7 import AgentCardV7, A2ADiscoveryRegistryV7, A2ADelegatorV7

def test_agent_card_v7_has_capability() -> None:
    """
    Test that AgentCardV7 correctly identifies if it has a capability or not.
    """
    card = AgentCardV7(
        agent_id="test-1",
        name="Test",
        description="Test",
        capabilities=["coding", "analysis_data"],
        endpoints={"rpc": "http://localhost/rpc"}
    )
    assert card.has_capability("coding") is True
    assert card.has_capability("analysis") is True
    assert card.has_capability("unknown") is False

def test_registry_find_agents_by_capability() -> None:
    """
    Test that A2ADiscoveryRegistryV7 can find an agent by capability.
    """
    registry = A2ADiscoveryRegistryV7()
    card1 = AgentCardV7(
        agent_id="test-1",
        name="Test 1",
        description="Test 1",
        capabilities=["coding"],
        endpoints={"rpc": "http://localhost/rpc"}
    )
    card2 = AgentCardV7(
        agent_id="test-2",
        name="Test 2",
        description="Test 2",
        capabilities=["analysis"],
        endpoints={"rpc": "http://localhost/rpc"}
    )
    registry.register_agent(card1)
    registry.register_agent(card2)

    agents = registry.find_agents_by_capability("coding")
    assert len(agents) == 1
    assert agents[0].agent_id == "test-1"

@pytest.mark.asyncio
async def test_delegator_delegate_task() -> None:
    """
    Test that A2ADelegatorV7 can successfully delegate a task.
    """
    card = AgentCardV7(
        agent_id="test-1",
        name="Test",
        description="Test",
        capabilities=["coding"],
        endpoints={"rpc": "http://localhost/rpc"}
    )
    delegator = A2ADelegatorV7()

    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success", "result": "done"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = await delegator.delegate_task(card, {"task": "do something"})

        assert result == {"status": "success", "result": "done"}
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost/rpc"
        assert kwargs["json"] == {"task": "do something"}

@pytest.mark.asyncio
async def test_delegator_delegate_by_capability() -> None:
    """
    Test that A2ADelegatorV7 can correctly discover and delegate a task by capability.
    """
    registry = A2ADiscoveryRegistryV7()
    card = AgentCardV7(
        agent_id="test-1",
        name="Test",
        description="Test",
        capabilities=["coding"],
        endpoints={"rpc": "http://localhost/rpc"}
    )
    registry.register_agent(card)
    delegator = A2ADelegatorV7(discovery_registry=registry)

    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success", "result": "done"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = await delegator.delegate_by_capability("coding", {"task": "do something"})

        assert result == {"status": "success", "result": "done"}

@pytest.mark.asyncio
async def test_delegator_delegate_by_capability_no_agents() -> None:
    """
    Test that A2ADelegatorV7 correctly raises a ValueError when no capable agent is found.
    """
    registry = A2ADiscoveryRegistryV7()
    delegator = A2ADelegatorV7(discovery_registry=registry)

    with pytest.raises(ValueError, match="No peer agents found supporting capability"):
        await delegator.delegate_by_capability("coding", {"task": "do something"})
