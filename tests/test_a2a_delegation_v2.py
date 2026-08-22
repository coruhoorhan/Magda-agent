import pytest
import respx
import json
from httpx import Response
from magda_agent.integration.a2a_delegation_v2 import A2ADelegatorV2
from magda_agent.integration.a2a_discovery_v3_unique import A2ADiscoveryServiceV3Unique

@pytest.fixture
def discovery_service():
    return A2ADiscoveryServiceV3Unique()

@pytest.fixture
def delegator(discovery_service):
    return A2ADelegatorV2(discovery_service=discovery_service)

@pytest.fixture
def peer_card_json():
    return json.dumps({
        "agent_id": "peer-001",
        "name": "MathAgent",
        "description": "Solves math problems",
        "capabilities": ["math"],
        "endpoints": {"rpc": "http://math-agent/rpc"},
        "protocol_version": "v3"
    })

@pytest.fixture
def no_rpc_peer_card_json():
    return json.dumps({
        "agent_id": "peer-002",
        "name": "NoRpcAgent",
        "description": "Agent without RPC",
        "capabilities": ["math"],
        "endpoints": {},
        "protocol_version": "v3"
    })

def test_format_task_payload(delegator):
    task_id = "task-123"
    capability = "math"
    task_parameters = {"a": 1, "b": 2}

    payload = delegator.format_task_payload(task_id, capability, task_parameters)

    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == "task-123"
    assert payload["method"] == "execute_task"
    assert payload["params"]["capability"] == "math"
    assert payload["params"]["task_parameters"] == {"a": 1, "b": 2}

def test_discover_peers(delegator, peer_card_json):
    delegator.discover_peers([peer_card_json])

    agents = delegator.discovery_service.find_agents_by_capability("math")
    assert len(agents) == 1
    assert agents[0].name == "MathAgent"
    assert agents[0].agent_id == "peer-001"

@pytest.mark.asyncio
@respx.mock
async def test_delegate_task_success(delegator, peer_card_json):
    delegator.discover_peers([peer_card_json])

    respx.post("http://math-agent/rpc").mock(return_value=Response(200, json={
        "result": {"status": "Success", "data": "3"}
    }))

    result = await delegator.delegate_task("task-123", "math", {"a": 1, "b": 2})

    assert "result" in result
    assert result["result"]["status"] == "Success"
    assert result["result"]["data"] == "3"

@pytest.mark.asyncio
async def test_delegate_task_no_peer_found(delegator):
    result = await delegator.delegate_task("task-123", "unknown_cap", {})

    assert "error" in result
    assert result["status"] == "failed"
    assert "No agent found" in result["error"]

@pytest.mark.asyncio
async def test_delegate_task_missing_rpc_endpoint(delegator, no_rpc_peer_card_json):
    delegator.discover_peers([no_rpc_peer_card_json])

    result = await delegator.delegate_task("task-124", "math", {})

    assert "error" in result
    assert result["status"] == "failed"
    assert "missing RPC endpoint" in result["error"]

@pytest.mark.asyncio
@respx.mock
async def test_delegate_task_network_error(delegator, peer_card_json):
    delegator.discover_peers([peer_card_json])

    respx.post("http://math-agent/rpc").mock(return_value=Response(500, json={"error": "Internal Server Error"}))

    result = await delegator.delegate_task("task-123", "math", {"a": 1, "b": 2})

    assert "error" in result
    assert result["status"] == "failed"
    assert "Network error" in result["error"]
