import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from magda_agent.integration.a2a_discovery import A2ADiscovery, AgentCard
from magda_agent.integration.a2a_delegation import A2ADelegator
from magda_agent.integration.a2a_orchestrator import A2AOrchestrator

@pytest.fixture
def a2a_discovery():
    local_card = AgentCard("local", "local", "local", [], {})
    discovery = A2ADiscovery(local_card)
    return discovery

@pytest.fixture
def a2a_delegator(a2a_discovery):
    delegator = A2ADelegator(a2a_discovery)
    return delegator

@pytest.fixture
def a2a_orchestrator(a2a_discovery, a2a_delegator):
    return A2AOrchestrator(a2a_discovery, a2a_delegator)


@pytest.mark.asyncio
async def test_dispatch_concurrently(a2a_orchestrator):
    # Mock the delegator's delegate_subplan method
    a2a_orchestrator.delegator.delegate_subplan = AsyncMock()

    async def mock_delegate(capability, step):
        # Simulate some delay to test concurrency properly
        await asyncio.sleep(0.01)
        return f"Delegated {step['id']} for {capability}"

    a2a_orchestrator.delegator.delegate_subplan.side_effect = mock_delegate

    sub_plans = [
        {"capability": "coding", "steps": [{"id": "step_1"}]},
        {"capability": "analysis", "steps": [{"id": "step_2"}, {"id": "step_3"}]}
    ]

    results = await a2a_orchestrator.dispatch_concurrently(sub_plans)

    assert len(results) == 3
    assert results["step_1"] == "Delegated step_1 for coding"
    assert results["step_2"] == "Delegated step_2 for analysis"
    assert results["step_3"] == "Delegated step_3 for analysis"
    assert a2a_orchestrator.delegator.delegate_subplan.call_count == 3


@pytest.mark.asyncio
async def test_dispatch_concurrently_with_exception(a2a_orchestrator, caplog):
    # Mock the delegator's delegate_subplan method
    a2a_orchestrator.delegator.delegate_subplan = AsyncMock()

    async def mock_delegate(capability, step):
        if step['id'] == "step_error":
            raise Exception("Simulated delegation failure")
        return f"Delegated {step['id']} for {capability}"

    a2a_orchestrator.delegator.delegate_subplan.side_effect = mock_delegate

    sub_plans = [
        {"capability": "coding", "steps": [{"id": "step_1"}, {"id": "step_error"}]},
    ]

    results = await a2a_orchestrator.dispatch_concurrently(sub_plans)

    # step_1 should succeed, step_error should fail and return exception
    assert len(results) == 1
    assert results["step_1"] == "Delegated step_1 for coding"
    assert "Simulated delegation failure" in caplog.text


@pytest.mark.asyncio
async def test_dispatch_concurrently_with_telemetry(a2a_discovery, a2a_delegator):
    mock_telemetry = MagicMock()
    orchestrator = A2AOrchestrator(a2a_discovery, a2a_delegator, telemetry=mock_telemetry)

    orchestrator.delegator.delegate_subplan = AsyncMock()

    async def mock_delegate(capability, step):
        if step['id'] == "step_error":
            raise Exception("Simulated delegation failure")
        return f"Delegated {step['id']} for {capability}"

    orchestrator.delegator.delegate_subplan.side_effect = mock_delegate

    sub_plans = [
        {"capability": "coding", "steps": [{"id": "step_1"}, {"id": "step_error"}]},
    ]

    results = await orchestrator.dispatch_concurrently(sub_plans)

    assert len(results) == 1
    assert mock_telemetry.track_event.call_count == 2
    mock_telemetry.track_event.assert_any_call(
        "orchestrator", "concurrent_delegation_start", {"num_sub_plans": 1}
    )
    mock_telemetry.track_event.assert_any_call(
        "orchestrator", "concurrent_delegation_end", {"success_count": 1, "failure_count": 1}
    )

@pytest.mark.asyncio
async def test_execute_orchestrated_plan_sequential(a2a_orchestrator):
    plan = [
        {"id": "step_1", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "coding"}, "description": "code it"}
    ]

    a2a_orchestrator.delegator.execute_plan = AsyncMock(return_value={"step_1": "Sequential Success"})

    results = await a2a_orchestrator.execute_orchestrated_plan(plan, concurrent=False)

    a2a_orchestrator.delegator.execute_plan.assert_called_once_with(plan)
    assert results["step_1"] == "Sequential Success"

@pytest.mark.asyncio
async def test_execute_orchestrated_plan_concurrent(a2a_orchestrator):
    plan = [
        {"id": "step_1", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "coding"}, "description": "code it"}
    ]

    # Mock the concurrent dispatch
    a2a_orchestrator.dispatch_concurrently = AsyncMock(return_value={"step_1": "Concurrent Success"})
    # Need to make sure split_plan is used
    a2a_orchestrator.delegator.split_plan = MagicMock(return_value=[{"capability": "coding", "steps": [plan[0]]}])

    results = await a2a_orchestrator.execute_orchestrated_plan(plan, concurrent=True)

    a2a_orchestrator.dispatch_concurrently.assert_called_once()
    assert results["step_1"] == "Concurrent Success"

@pytest.mark.asyncio
async def test_execute_orchestrated_plan_empty(a2a_orchestrator):
    results = await a2a_orchestrator.execute_orchestrated_plan([])
    assert results == {}

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post', new_callable=AsyncMock)
async def test_execute_direct_mcp_action(mock_post, a2a_orchestrator, a2a_discovery):
    # Register the test agent
    agent_card = AgentCard("target_agent", "Target", "Desc", ["capability_1"], {"mcp": "http://mcp-endpoint"})
    a2a_discovery._discovered_agents["target_agent"] = agent_card

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "content": [{"type": "text", "text": "Action Success"}]
        }
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = await a2a_orchestrator.execute_direct_mcp_action("target_agent", "some_tool", {"arg": "val"})

    assert result == "Action Success"
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs["json"]["method"] == "tools/call"
    assert kwargs["json"]["params"]["name"] == "some_tool"
    assert kwargs["json"]["params"]["arguments"] == {"arg": "val"}

@pytest.mark.asyncio
async def test_execute_orchestrated_plan_with_mcp_tool_sequential(a2a_orchestrator):
    plan = [
        {"id": "step_1", "skill": "execute_mcp_tool", "skill_kwargs": {
            "target_agent_id": "target_agent",
            "tool_name": "some_tool",
            "tool_kwargs": {"arg": "val"}
        }},
        {"id": "step_2", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "coding"}, "description": "code it"}
    ]

    a2a_orchestrator.execute_direct_mcp_action = AsyncMock(return_value="MCP Success")
    a2a_orchestrator.delegator.execute_plan = AsyncMock(return_value={"step_2": "Sequential Success"})

    results = await a2a_orchestrator.execute_orchestrated_plan(plan, concurrent=False)

    a2a_orchestrator.execute_direct_mcp_action.assert_called_once_with("target_agent", "some_tool", {"arg": "val"})
    a2a_orchestrator.delegator.execute_plan.assert_called_once()

    assert results["step_1"] == "MCP Success"
    assert results["step_2"] == "Sequential Success"

@pytest.mark.asyncio
async def test_execute_orchestrated_plan_with_mcp_tool_concurrent(a2a_orchestrator):
    plan = [
        {"id": "step_1", "skill": "execute_mcp_tool", "skill_kwargs": {
            "target_agent_id": "target_agent",
            "tool_name": "some_tool",
            "tool_kwargs": {"arg": "val"}
        }},
        {"id": "step_2", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "coding"}, "description": "code it"}
    ]

    a2a_orchestrator.execute_direct_mcp_action = AsyncMock(return_value="MCP Concurrent Success")
    a2a_orchestrator.dispatch_concurrently = AsyncMock(return_value={"step_2": "Delegation Concurrent Success"})
    a2a_orchestrator.delegator.split_plan = MagicMock(return_value=[{"capability": "coding", "steps": [plan[1]]}])

    results = await a2a_orchestrator.execute_orchestrated_plan(plan, concurrent=True)

    a2a_orchestrator.execute_direct_mcp_action.assert_called_once_with("target_agent", "some_tool", {"arg": "val"})
    a2a_orchestrator.dispatch_concurrently.assert_called_once()

    assert results["step_1"] == "MCP Concurrent Success"
    assert results["step_2"] == "Delegation Concurrent Success"
