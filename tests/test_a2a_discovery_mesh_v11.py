import pytest
from typing import List
from magda_agent.integration.a2a_discovery import A2ADiscovery, AgentCard
from magda_agent.integration.a2a_discovery_mesh_v11 import A2ADiscoveryMeshV11

@pytest.fixture
def a2a_discovery() -> A2ADiscovery:
    """Provides a mocked A2ADiscovery instance."""
    local_card = AgentCard("local", "local", "local", [], {})
    discovery = A2ADiscovery(local_card)
    return discovery

@pytest.fixture
def a2a_discovery_mesh(a2a_discovery: A2ADiscovery) -> A2ADiscoveryMeshV11:
    """Provides an A2ADiscoveryMeshV11 instance."""
    return A2ADiscoveryMeshV11(a2a_discovery)

def test_register_peers(a2a_discovery_mesh: A2ADiscoveryMeshV11, a2a_discovery: A2ADiscovery) -> None:
    """Tests registering peers into the mesh."""
    cards = [
        AgentCard("peer1", "Peer 1", "Desc 1", ["cap1"], {"mcp": "url1"}),
        AgentCard("peer2", "Peer 2", "Desc 2", ["cap2"], {"mcp": "url2"})
    ]
    a2a_discovery_mesh.register_peers(cards)

    assert "peer1" in a2a_discovery._discovered_agents
    assert "peer2" in a2a_discovery._discovered_agents
    assert a2a_discovery_mesh.get_peer_state("peer1") == "active"
    assert a2a_discovery_mesh.get_peer_state("peer2") == "active"

def test_update_peer_state(a2a_discovery_mesh: A2ADiscoveryMeshV11) -> None:
    """Tests updating the state of a registered peer."""
    card = AgentCard("peer1", "Peer 1", "Desc 1", ["cap1"], {"mcp": "url1"})
    a2a_discovery_mesh.register_peers([card])

    assert a2a_discovery_mesh.get_peer_state("peer1") == "active"

    a2a_discovery_mesh.update_peer_state("peer1", "offline")
    assert a2a_discovery_mesh.get_peer_state("peer1") == "offline"

    # Should not raise exception for unknown peer
    a2a_discovery_mesh.update_peer_state("unknown_peer", "active")
    assert a2a_discovery_mesh.get_peer_state("unknown_peer") is None

def test_find_best_peer_for_capability(a2a_discovery_mesh: A2ADiscoveryMeshV11) -> None:
    """Tests finding the best active peer for a capability."""
    cards = [
        AgentCard("peer1", "Peer 1", "Desc 1", ["cap1"], {"mcp": "url1"}),
        AgentCard("peer2", "Peer 2", "Desc 2", ["cap1"], {"mcp": "url2"}),
        AgentCard("peer3", "Peer 3", "Desc 3", ["cap2"], {"mcp": "url3"})
    ]
    a2a_discovery_mesh.register_peers(cards)

    # All active initially
    best_peer = a2a_discovery_mesh.find_best_peer_for_capability("cap1")
    assert best_peer is not None
    assert best_peer.agent_id in ["peer1", "peer2"]

    # Make peer1 offline, so it should return peer2
    a2a_discovery_mesh.update_peer_state("peer1", "offline")
    best_peer_after_update = a2a_discovery_mesh.find_best_peer_for_capability("cap1")
    assert best_peer_after_update is not None
    assert best_peer_after_update.agent_id == "peer2"

    # Capability with only offline peers
    a2a_discovery_mesh.update_peer_state("peer2", "offline")
    assert a2a_discovery_mesh.find_best_peer_for_capability("cap1") is None

    # Capability that doesn't exist
    assert a2a_discovery_mesh.find_best_peer_for_capability("unknown_cap") is None
