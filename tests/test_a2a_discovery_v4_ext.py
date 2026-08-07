import pytest
from typing import List, Optional
from magda_agent.integration.a2a_discovery_v4 import AgentCardV4
from magda_agent.integration.a2a_discovery_v4_ext import A2ADiscoveryRegistryV4Ext

@pytest.fixture
def registry() -> A2ADiscoveryRegistryV4Ext:
    """Provides a fresh instance of A2ADiscoveryRegistryV4Ext."""
    return A2ADiscoveryRegistryV4Ext()

@pytest.fixture
def mock_cards() -> List[str]:
    """Provides mock JSON string representations of Agent Cards."""
    return [
        AgentCardV4(
            agent_id="agent-1",
            name="Planner Agent",
            description="Specializes in generating plans",
            capabilities=["planning", "research", "summarization"],
            endpoints={"rpc": "http://agent-1/rpc"}
        ).to_json(),
        AgentCardV4(
            agent_id="agent-2",
            name="Coder Agent",
            description="Writes and reviews code",
            capabilities=["coding", "review", "git"],
            endpoints={"rpc": "http://agent-2/rpc"}
        ).to_json(),
        AgentCardV4(
            agent_id="agent-3",
            name="DevOps Agent",
            description="Handles deployments",
            capabilities=["deployment", "docker", "git"],
            endpoints={"rpc": "http://agent-3/rpc"}
        ).to_json()
    ]

def test_filter_by_capabilities_matches_exact(registry: A2ADiscoveryRegistryV4Ext, mock_cards: List[str]) -> None:
    """Verifies that filter_by_capabilities returns the exact match for required capabilities."""
    registry.parse_and_register_cards(mock_cards)

    results = registry.filter_by_capabilities(["coding", "review"])
    assert len(results) == 1
    assert results[0].agent_id == "agent-2"

def test_filter_by_capabilities_matches_multiple(registry: A2ADiscoveryRegistryV4Ext, mock_cards: List[str]) -> None:
    """Verifies that filter_by_capabilities returns multiple agents if they possess the capability."""
    registry.parse_and_register_cards(mock_cards)

    results = registry.filter_by_capabilities(["git"])
    assert len(results) == 2
    agent_ids = {agent.agent_id for agent in results}
    assert agent_ids == {"agent-2", "agent-3"}

def test_filter_by_capabilities_no_match(registry: A2ADiscoveryRegistryV4Ext, mock_cards: List[str]) -> None:
    """Verifies that filter_by_capabilities returns empty list when no agent has the capability."""
    registry.parse_and_register_cards(mock_cards)

    results = registry.filter_by_capabilities(["machine-learning"])
    assert len(results) == 0

def test_filter_by_capabilities_empty_requirements(registry: A2ADiscoveryRegistryV4Ext, mock_cards: List[str]) -> None:
    """Verifies that filter_by_capabilities returns all agents when empty capabilities are provided."""
    registry.parse_and_register_cards(mock_cards)

    results = registry.filter_by_capabilities([])
    assert len(results) == 3

def test_find_agent_for_delegation(registry: A2ADiscoveryRegistryV4Ext, mock_cards: List[str]) -> None:
    """Verifies find_agent_for_delegation successfully returns a matching agent."""
    registry.parse_and_register_cards(mock_cards)

    agent = registry.find_agent_for_delegation(["planning", "summarization"])
    assert agent is not None
    assert agent.agent_id == "agent-1"

def test_find_agent_for_delegation_with_exclude(registry: A2ADiscoveryRegistryV4Ext, mock_cards: List[str]) -> None:
    """Verifies find_agent_for_delegation respects the exclude_agent_id parameter."""
    registry.parse_and_register_cards(mock_cards)

    agent = registry.find_agent_for_delegation(["git"], exclude_agent_id="agent-3")
    assert agent is not None
    assert agent.agent_id == "agent-2"

def test_find_agent_for_delegation_no_match(registry: A2ADiscoveryRegistryV4Ext, mock_cards: List[str]) -> None:
    """Verifies find_agent_for_delegation returns None if no single agent satisfies all capabilities."""
    registry.parse_and_register_cards(mock_cards)

    agent = registry.find_agent_for_delegation(["planning", "coding"])
    assert agent is None
