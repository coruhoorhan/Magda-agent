import pytest
import json
import respx
import httpx
from dataclasses import asdict
from magda_agent.integration.a2a_discovery_v2 import AgentCardV2, A2ADiscoveryV2

@pytest.fixture
def local_card_v2() -> AgentCardV2:
    return AgentCardV2(
        agent_id="agent-v2-001",
        name="MagdaLocalV2",
        description="Local agent for testing v2",
        capabilities=["chat", "code_execution"],
        endpoints={"rpc": "http://localhost:8080/rpc"}
    )

@pytest.fixture
def remote_card_1_v2() -> AgentCardV2:
    return AgentCardV2(
        agent_id="agent-remote-v2-001",
        name="RemoteWorker1V2",
        description="Worker node v2",
        capabilities=["image_generation", "chat"],
        endpoints={"rpc": "http://192.168.1.10:8080/rpc"}
    )

@pytest.fixture
def remote_card_2_v2() -> AgentCardV2:
    return AgentCardV2(
        agent_id="agent-remote-v2-002",
        name="RemoteWorker2V2",
        description="Code worker node v2",
        capabilities=["code_execution", "linting"],
        endpoints={"rpc": "http://192.168.1.11:8080/rpc"}
    )

@pytest.mark.asyncio
async def test_broadcast_card_v2(local_card_v2: AgentCardV2) -> None:
    discovery = A2ADiscoveryV2(local_card=local_card_v2)
    broadcasted_json = await discovery.broadcast_card()

    envelope = json.loads(broadcasted_json)
    assert envelope["type"] == "a2a_discovery_broadcast"
    assert envelope["version"] == "2.0"

    payload = json.loads(envelope["payload"])
    assert payload["agent_id"] == "agent-v2-001"
    assert payload["name"] == "MagdaLocalV2"
    assert "chat" in payload["capabilities"]
    assert payload["protocol_version"] == "v2"

@pytest.mark.asyncio
async def test_fetch_and_index_cards_v2(local_card_v2: AgentCardV2, remote_card_1_v2: AgentCardV2, remote_card_2_v2: AgentCardV2) -> None:
    discovery = A2ADiscoveryV2(local_card=local_card_v2)

    network_envelopes = [
        json.dumps({"type": "a2a_discovery_broadcast", "version": "2.0", "payload": remote_card_1_v2.to_json()}),
        json.dumps({"type": "a2a_discovery_broadcast", "version": "2.0", "payload": remote_card_2_v2.to_json()})
    ]

    await discovery.fetch_cards(network_envelopes=network_envelopes)

    # Test getting by id
    fetched_agent = discovery.get_agent_by_id("agent-remote-v2-001")
    assert fetched_agent is not None
    assert fetched_agent.name == "RemoteWorker1V2"

    # Test indexing by capability (chat)
    chat_agents = discovery.find_agents_by_capability("chat")
    assert len(chat_agents) == 1
    assert chat_agents[0].agent_id == "agent-remote-v2-001"

    # Test indexing by capability (code_execution)
    code_agents = discovery.find_agents_by_capability("code_execution")
    assert len(code_agents) == 1
    assert code_agents[0].agent_id == "agent-remote-v2-002"

    # Test missing capability
    missing_agents = discovery.find_agents_by_capability("unknown_cap")
    assert len(missing_agents) == 0

@pytest.mark.asyncio
async def test_fetch_invalid_card_json_v2(local_card_v2: AgentCardV2) -> None:
    discovery = A2ADiscoveryV2(local_card=local_card_v2)

    # Passing invalid JSON should be caught and not crash
    network_envelopes = [
        '{"invalid": "json"',
        json.dumps({"type": "wrong_type", "version": "2.0", "payload": "{}"}),
        json.dumps({"type": "a2a_discovery_broadcast", "version": "1.0", "payload": "{}"}),
        json.dumps({"type": "a2a_discovery_broadcast", "version": "2.0", "payload": '{"agent_id": "missing_fields"}'})
    ]

    await discovery.fetch_cards(network_envelopes=network_envelopes)

    # No agents should be discovered
    assert len(discovery._discovered_agents) == 0

@pytest.mark.asyncio
async def test_find_agents_by_capabilities_filtering(local_card_v2: AgentCardV2, remote_card_1_v2: AgentCardV2, remote_card_2_v2: AgentCardV2) -> None:
    discovery = A2ADiscoveryV2(local_card=local_card_v2)

    network_envelopes = [
        json.dumps({"type": "a2a_discovery_broadcast", "version": "2.0", "payload": remote_card_1_v2.to_json()}),
        json.dumps({"type": "a2a_discovery_broadcast", "version": "2.0", "payload": remote_card_2_v2.to_json()})
    ]

    await discovery.fetch_cards(network_envelopes=network_envelopes)

    # Remote 1: "image_generation", "chat"
    # Remote 2: "code_execution", "linting"

    # Match all (intersection check)
    match_both = discovery.find_agents_by_capabilities(["chat", "image_generation"], match_all=True)
    assert len(match_both) == 1
    assert match_both[0].agent_id == remote_card_1_v2.agent_id

    match_non_existent = discovery.find_agents_by_capabilities(["chat", "code_execution"], match_all=True)
    assert len(match_non_existent) == 0

    # Match any (union check)
    match_any = discovery.find_agents_by_capabilities(["chat", "code_execution"], match_all=False)
    assert len(match_any) == 2
    agent_ids = {a.agent_id for a in match_any}
    assert remote_card_1_v2.agent_id in agent_ids
    assert remote_card_2_v2.agent_id in agent_ids

    # Empty capabilities input should return all discovered agents
    match_empty = discovery.find_agents_by_capabilities([])
    assert len(match_empty) == 2

@pytest.mark.asyncio
@respx.mock
async def test_register_with_registry_v2(local_card_v2: AgentCardV2) -> None:
    discovery = A2ADiscoveryV2(local_card=local_card_v2)
    registry_url = "http://discovery-registry-v2.local"

    # Mock the registration endpoint
    route = respx.post(f"{registry_url}/register").mock(return_value=httpx.Response(201))

    success = await discovery.register_with_registry(registry_url, auth_token="test-token-v2")

    assert success is True
    assert route.called
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-token-v2"

    # Verify the payload
    sent_data = json.loads(route.calls.last.request.content)
    assert sent_data["agent_id"] == local_card_v2.agent_id
    assert sent_data["protocol_version"] == "v2"

@pytest.mark.asyncio
@respx.mock
async def test_discover_from_registry_v2(local_card_v2: AgentCardV2, remote_card_1_v2: AgentCardV2, remote_card_2_v2: AgentCardV2) -> None:
    discovery = A2ADiscoveryV2(local_card=local_card_v2)
    registry_url = "http://discovery-registry-v2.local"

    # Mock the discovery endpoint
    cards_data = [asdict(remote_card_1_v2), asdict(remote_card_2_v2)]
    route = respx.get(f"{registry_url}/cards").mock(return_value=httpx.Response(200, json=cards_data))

    discovered = await discovery.discover_from_registry(registry_url)

    assert len(discovered) == 2
    assert route.called
    assert discovery.get_agent_by_id(remote_card_1_v2.agent_id).name == remote_card_1_v2.name
    assert discovery.get_agent_by_id(remote_card_2_v2.agent_id).name == remote_card_2_v2.name
    assert remote_card_1_v2.name in [c.name for c in discovery.find_agents_by_capabilities(["chat"])]

@pytest.mark.asyncio
@respx.mock
async def test_register_with_registry_failure_v2(local_card_v2: AgentCardV2) -> None:
    discovery = A2ADiscoveryV2(local_card=local_card_v2)
    registry_url = "http://discovery-registry-v2.local"

    # Mock a failure
    respx.post(f"{registry_url}/register").mock(return_value=httpx.Response(500))

    success = await discovery.register_with_registry(registry_url)
    assert success is False


@pytest.mark.asyncio
async def test_prefix_matching_and_workflow_subagent_discovery_v2(
    local_card_v2: AgentCardV2, remote_card_1_v2: AgentCardV2, remote_card_2_v2: AgentCardV2
) -> None:
    discovery = A2ADiscoveryV2(local_card=local_card_v2)

    # Remote 1 capabilities: ["image_generation", "chat"]
    # Remote 2 capabilities: ["code_execution", "linting"]
    network_envelopes = [
        json.dumps({"type": "a2a_discovery_broadcast", "version": "2.0", "payload": remote_card_1_v2.to_json()}),
        json.dumps({"type": "a2a_discovery_broadcast", "version": "2.0", "payload": remote_card_2_v2.to_json()})
    ]

    await discovery.fetch_cards(network_envelopes=network_envelopes)

    # Prefix match test
    code_subagents = discovery.find_agents_by_capability("code_", prefix_match=True)
    assert len(code_subagents) == 1
    assert code_subagents[0].agent_id == remote_card_2_v2.agent_id

    # Test workflow subagent discovery ranking
    discovered = discovery.discover_workflow_subagents(
        required_capabilities=["code_"],
        optional_capabilities=["linting"],
        prefix_match=True,
    )
    assert len(discovered) == 1
    assert discovered[0]["agent_id"] == remote_card_2_v2.agent_id
    assert discovered[0]["score"] == 150.0

    # Select subagent for workflow task
    selected = discovery.select_subagent_for_workflow_task(
        required_capabilities=["code_execution"], prefix_match=False
    )
    assert selected is not None
    assert selected.agent_id == remote_card_2_v2.agent_id

    # Non-matching subagent selection
    no_match = discovery.select_subagent_for_workflow_task(
        required_capabilities=["unsupported_task"], prefix_match=True
    )
    assert no_match is None
