import pytest
from magda_agent.memory.context_reranker import ContextLiveRerankerPlugin
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState

@pytest.fixture
def mock_context_engine():
    plugin = ContextLiveRerankerPlugin()
    engine = ContextEngine(plugins=[plugin])
    return engine

def test_context_live_reranker_after_retrieval(mock_context_engine):
    # Setup test entries
    # importance, arousal
    state1 = PADState(pleasure=0.1, arousal=0.8, dominance=0.5)
    entry1 = MemoryEntry(content="Entry 1", importance=0.2, emotional_state=state1) # Score: 0.2 + 0.4 = 0.6

    state2 = PADState(pleasure=0.5, arousal=0.2, dominance=0.5)
    entry2 = MemoryEntry(content="Entry 2", importance=0.8, emotional_state=state2) # Score: 0.8 + 0.1 = 0.9

    state3 = PADState(pleasure=0.5, arousal=0.5, dominance=0.5)
    entry3 = MemoryEntry(content="Entry 3", importance=0.1, emotional_state=state3) # Score: 0.1 + 0.25 = 0.35

    entries = [entry1, entry2, entry3]

    # Mock retrieval function
    def base_retrieval(query: str, user_id: int):
        return entries

    result = mock_context_engine.retrieve_context("query", 1, base_retrieval)

    # Expected order: entry2 (0.9), entry1 (0.6), entry3 (0.35)
    assert len(result) == 3
    assert result[0] == entry2
    assert result[1] == entry1
    assert result[2] == entry3

def test_context_live_reranker_empty_context():
    plugin = ContextLiveRerankerPlugin()
    result = plugin.after_retrieval([], "query", 1)
    assert result == []

def test_context_live_reranker_missing_attributes():
    plugin = ContextLiveRerankerPlugin()

    class DummyEntry:
        def __init__(self, content):
            self.content = content

    entry1 = DummyEntry("missing attr")
    entry2 = DummyEntry("missing attr too")

    result = plugin.after_retrieval([entry1, entry2], "query", 1)
    # They both have score 0.0, so order is preserved or depends on sorting stability
    assert len(result) == 2
    assert result[0] in [entry1, entry2]
