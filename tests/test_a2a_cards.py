import pytest
import json
from unittest.mock import MagicMock
from magda_agent.integration.a2a_cards import AgentCardV3, A2ADiscoveryV3, AgentCardV4, A2ADiscoveryV4


def test_agent_card_v3_serialization():
    card = AgentCardV3(
        agent_id="test-agent",
        name="Test Agent",
        description="A test agent",
        capabilities=["code_execution", "web_search"],
        endpoints={"mcp": "http://localhost:8080"}
    )
    json_str = card.to_json()
    new_card = AgentCardV3.from_json(json_str)
    assert new_card.agent_id == card.agent_id
    assert new_card.capabilities == card.capabilities
    assert new_card.protocol_version == "v3"


def test_agent_card_v3_capability_matching():
    card = AgentCardV3(
        agent_id="test-agent",
        name="Test Agent",
        description="A test agent",
        capabilities=["code_execution", "web_search", "image_gen"],
        endpoints={}
    )

    # Exact match
    assert card.has_capability("code_execution") is True
    assert card.has_capability("web_search") is True

    # Prefix match
    assert card.has_capability("code") is True
    assert card.has_capability("image") is True

    # No match
    assert card.has_capability("translation") is False
    assert card.has_capability("image_generation") is False # image_gen vs image_generation

    # Matches any
    assert card.matches_any_capability(["translation", "web_search"]) is True
    assert card.matches_any_capability(["translation", "data_science"]) is False


@pytest.mark.asyncio
async def test_a2a_discovery_v3():
    local_card = AgentCardV3(
        agent_id="local-agent",
        name="Local Agent",
        description="Local agent",
        capabilities=["orchestration"],
        endpoints={}
    )
    discovery = A2ADiscoveryV3(local_card=local_card)

    # Test broadcast
    broadcast_json = await discovery.broadcast_card()
    broadcast_data = json.loads(broadcast_json)
    assert broadcast_data["type"] == "a2a_discovery_broadcast"
    assert broadcast_data["version"] == "3.0"
    assert broadcast_data["payload"]["agent_id"] == "local-agent"

    # Test fetch and register
    peer_card = AgentCardV3(
        agent_id="peer-agent",
        name="Peer Agent",
        description="Peer agent",
        capabilities=["code_execution"],
        endpoints={}
    )
    peer_envelope = {
        "type": "a2a_discovery_broadcast",
        "version": "3.0",
        "payload": json.loads(peer_card.to_json())
    }

    await discovery.fetch_cards(network_envelopes=[json.dumps(peer_envelope)])

    assert discovery.get_agent_by_id("peer-agent") is not None

    # Test search by capability
    matched = discovery.find_agents_by_capability("code_execution")
    assert len(matched) == 1
    assert matched[0].agent_id == "peer-agent"

    # Test search by prefix capability
    matched_prefix = discovery.find_agents_by_capability("code")
    assert len(matched_prefix) == 1
    assert matched_prefix[0].agent_id == "peer-agent"

    # Test no match
    matched_none = discovery.find_agents_by_capability("translation")
    assert len(matched_none) == 0


def test_agent_card_v4_serialization():
    card = AgentCardV4(
        agent_id="v4-agent-1",
        name="V4 Agent One",
        description="A version 4 agent",
        capabilities=["data_analysis", "ml_inference"],
        endpoints={"a2a": "http://localhost:9090"},
        metadata={"region": "us-east-1"},
        status="active"
    )
    json_str = card.to_json()
    deserialized = AgentCardV4.from_json(json_str)
    assert deserialized.agent_id == "v4-agent-1"
    assert deserialized.name == "V4 Agent One"
    assert deserialized.capabilities == ["data_analysis", "ml_inference"]
    assert deserialized.protocol_version == "v4"
    assert deserialized.metadata == {"region": "us-east-1"}
    assert deserialized.status == "active"


def test_agent_card_v4_capability_matching():
    card = AgentCardV4(
        agent_id="v4-agent-2",
        name="V4 Agent Two",
        description="Capabilities tester",
        capabilities=["code_refactoring", "data_processing"],
        endpoints={}
    )
    assert card.has_capability("code_refactoring") is True
    assert card.has_capability("code") is True
    assert card.has_capability("data") is True
    assert card.has_capability("nlp") is False

    assert card.matches_any_capability(["nlp", "data"]) is True
    assert card.matches_any_capability(["nlp", "vision"]) is False


@pytest.mark.asyncio
async def test_a2a_discovery_v4_broadcast_and_parse():
    local_card = AgentCardV4(
        agent_id="local-v4",
        name="Local V4",
        description="Local V4 discovery agent",
        capabilities=["orchestration_v4"],
        endpoints={"http": "http://127.0.0.1:8000"}
    )
    discovery = A2ADiscoveryV4(local_card=local_card)

    broadcast_json = await discovery.broadcast_card()
    broadcast_data = json.loads(broadcast_json)
    assert broadcast_data["type"] == "a2a_discovery_broadcast"
    assert broadcast_data["version"] == "4.0"
    assert broadcast_data["payload"]["agent_id"] == "local-v4"

    # Test parse envelope with string payload
    peer_card = AgentCardV4(
        agent_id="peer-v4-1",
        name="Peer V4 One",
        description="Peer 1",
        capabilities=["vision_processing"],
        endpoints={}
    )
    envelope_dict = {
        "type": "a2a_discovery_broadcast",
        "version": "4.0",
        "payload": peer_card.to_json()
    }
    parsed = discovery.parse_envelope(json.dumps(envelope_dict))
    assert parsed is not None
    assert parsed.agent_id == "peer-v4-1"

    # Test parse envelope with dict payload
    envelope_dict2 = {
        "type": "a2a_discovery_broadcast",
        "version": "4.0",
        "payload": json.loads(peer_card.to_json())
    }
    parsed2 = discovery.parse_envelope(json.dumps(envelope_dict2))
    assert parsed2 is not None
    assert parsed2.agent_id == "peer-v4-1"

    # Test parse direct json card format
    parsed_direct = discovery.parse_envelope(peer_card.to_json())
    assert parsed_direct is not None
    assert parsed_direct.agent_id == "peer-v4-1"

    # Test invalid envelope format returns None
    assert discovery.parse_envelope("invalid json") is None
    assert discovery.parse_envelope(json.dumps({"unrecognized": True})) is None


@pytest.mark.asyncio
async def test_a2a_discovery_v4_fetch_and_lookup():
    local_card = AgentCardV4(
        agent_id="local-v4",
        name="Local V4",
        description="Local V4 agent",
        capabilities=["manager"],
        endpoints={}
    )
    discovery = A2ADiscoveryV4(local_card=local_card)

    peer1 = AgentCardV4(
        agent_id="peer-v4-a",
        name="Peer A",
        description="Peer A desc",
        capabilities=["sql_query"],
        endpoints={}
    )
    peer2 = AgentCardV4(
        agent_id="peer-v4-b",
        name="Peer B",
        description="Peer B desc",
        capabilities=["sql_optimization"],
        endpoints={}
    )

    env1 = json.dumps({"type": "a2a_discovery_broadcast", "version": "4.0", "payload": json.loads(peer1.to_json())})
    env2 = json.dumps({"type": "a2a_discovery_broadcast", "version": "4.0", "payload": json.loads(peer2.to_json())})

    fetched = await discovery.fetch_cards(network_envelopes=[env1, env2])
    assert len(fetched) == 2

    assert discovery.get_agent_by_id("peer-v4-a") is not None
    assert discovery.get_agent_by_id("peer-v4-b") is not None
    assert len(discovery.get_all_agents()) == 2

    sql_agents = discovery.find_agents_by_capability("sql")
    assert len(sql_agents) == 2

    exact_agents = discovery.find_agents_by_capability("sql_query")
    assert len(exact_agents) == 1
    assert exact_agents[0].agent_id == "peer-v4-a"


@pytest.mark.asyncio
async def test_a2a_discovery_v4_auth():
    local_card = AgentCardV4(
        agent_id="local-v4",
        name="Local V4",
        description="Local V4 agent",
        capabilities=["manager"],
        endpoints={}
    )
    mock_sec_ctx = MagicMock()
    mock_sec_ctx.validate_token.side_effect = lambda t: t == "valid-token"

    discovery = A2ADiscoveryV4(local_card=local_card, security_context=mock_sec_ctx)

    # Invalid token raises ValueError
    with pytest.raises(ValueError, match="Invalid authentication token"):
        await discovery.fetch_cards(network_envelopes=[], auth_token="invalid-token")

    # Valid token succeeds
    res = await discovery.fetch_cards(network_envelopes=[], auth_token="valid-token")
    assert res == []
