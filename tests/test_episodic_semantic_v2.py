import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.episodic_semantic_v2 import EpisodicSemanticMemoryManagerV2
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.memory.semantic import SemanticMemory
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_episodic():
    episodic = MagicMock(spec=EpisodicMemory)
    episodic.store_event = MagicMock()
    episodic.recall_events = MagicMock(return_value=["event 1", "event 2"])
    return episodic

@pytest.fixture
def mock_semantic():
    semantic = MagicMock(spec=SemanticMemory)
    semantic.store_fact = MagicMock()
    semantic.recall_facts = MagicMock(return_value=["fact 1"])
    return semantic

@pytest.fixture
def mock_llm_client():
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock()
    return llm_client

@pytest.mark.asyncio
async def test_route_and_store_semantic_llm(mock_episodic, mock_semantic, mock_llm_client):
    manager = EpisodicSemanticMemoryManagerV2(mock_episodic, mock_semantic, mock_llm_client)

    mock_llm_client.generate.return_value = " semantic \n"

    store_used = await manager.route_and_store("The sky is blue.", metadata={"key": "value"}, user_id=1)

    assert store_used == "semantic"
    mock_semantic.store_fact.assert_called_once_with(text="The sky is blue.", metadata={"key": "value"}, user_id=1)
    mock_episodic.store_event.assert_not_called()
    mock_llm_client.generate.assert_called_once()

@pytest.mark.asyncio
async def test_route_and_store_episodic_llm(mock_episodic, mock_semantic, mock_llm_client):
    manager = EpisodicSemanticMemoryManagerV2(mock_episodic, mock_semantic, mock_llm_client)

    mock_llm_client.generate.return_value = "episodic"

    store_used = await manager.route_and_store("Yesterday I went to the store.", metadata={"key": "value"}, user_id=2)

    assert store_used == "episodic"
    mock_episodic.store_event.assert_called_once_with(text="Yesterday I went to the store.", metadata={"key": "value"}, user_id=2)
    mock_semantic.store_fact.assert_not_called()
    mock_llm_client.generate.assert_called_once()

@pytest.mark.asyncio
async def test_route_and_store_semantic_heuristic(mock_episodic, mock_semantic):
    # No LLM client provided, relies on heuristics
    manager = EpisodicSemanticMemoryManagerV2(mock_episodic, mock_semantic)

    store_used = await manager.route_and_store("Water is a liquid.", user_id=3)

    assert store_used == "semantic"
    mock_semantic.store_fact.assert_called_once_with(text="Water is a liquid.", metadata=None, user_id=3)
    mock_episodic.store_event.assert_not_called()

@pytest.mark.asyncio
async def test_route_and_store_episodic_heuristic(mock_episodic, mock_semantic):
    # No LLM client provided, relies on heuristics
    manager = EpisodicSemanticMemoryManagerV2(mock_episodic, mock_semantic)

    store_used = await manager.route_and_store("I did my homework today.", user_id=4)

    assert store_used == "episodic"
    mock_episodic.store_event.assert_called_once_with(text="I did my homework today.", metadata=None, user_id=4)
    mock_semantic.store_fact.assert_not_called()

@pytest.mark.asyncio
async def test_route_and_store_llm_failure_fallback(mock_episodic, mock_semantic, mock_llm_client):
    manager = EpisodicSemanticMemoryManagerV2(mock_episodic, mock_semantic, mock_llm_client)

    # Simulate LLM failure
    mock_llm_client.generate.side_effect = Exception("API error")

    # "is a" triggers semantic heuristic
    store_used = await manager.route_and_store("A dog is a mammal.", user_id=5)

    assert store_used == "semantic"
    mock_semantic.store_fact.assert_called_once_with(text="A dog is a mammal.", metadata=None, user_id=5)
    mock_episodic.store_event.assert_not_called()

def test_retrieve_episodic(mock_episodic, mock_semantic):
    manager = EpisodicSemanticMemoryManagerV2(mock_episodic, mock_semantic)

    results = manager.retrieve_episodic("went to store", top_k=2, user_id=1)

    assert results == ["event 1", "event 2"]
    mock_episodic.recall_events.assert_called_once_with(query="went to store", top_k=2, user_id=1)

def test_retrieve_semantic(mock_episodic, mock_semantic):
    manager = EpisodicSemanticMemoryManagerV2(mock_episodic, mock_semantic)

    results = manager.retrieve_semantic("what is water", top_k=1, user_id=2)

    assert results == ["fact 1"]
    mock_semantic.recall_facts.assert_called_once_with(query="what is water", top_k=1, user_id=2)
