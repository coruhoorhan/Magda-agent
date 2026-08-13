import pytest
from magda_agent.memory.semantic_reranker import SemanticRerankerPlugin, compute_cosine_similarity
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState

def test_cosine_similarity_basic():
    # Identical strings should be perfectly similar (1.0)
    assert compute_cosine_similarity("apple banana cherry", "banana cherry apple") == pytest.approx(1.0)

    # Completely disjoint strings should have 0.0 similarity
    assert compute_cosine_similarity("apple banana", "cherry orange") == 0.0

    # Partial overlaps
    sim1 = compute_cosine_similarity("apple banana", "apple banana cherry")
    sim2 = compute_cosine_similarity("apple banana", "apple cherry")
    assert sim1 > sim2

def test_cosine_similarity_edge_cases():
    assert compute_cosine_similarity("", "apple") == 0.0
    assert compute_cosine_similarity("apple", "") == 0.0
    assert compute_cosine_similarity("   ", "   ") == 0.0

def test_semantic_reranker_sorting():
    plugin = SemanticRerankerPlugin()

    # Query: find "apple"
    query = "apple pie"

    # Context entries: different structures
    entry1 = "This is a cherry tart dessert" # similarity to query should be 0.0
    entry2 = {"text": "I love eating delicious apple pie!"} # high similarity

    state = PADState(pleasure=0.5, arousal=0.5, dominance=0.5)
    entry3 = MemoryEntry(content="An apple is a red or green fruit", importance=0.5, emotional_state=state) # medium similarity

    context = [entry1, entry2, entry3]

    reranked = plugin.after_retrieval(context, query, 1)

    # Expected order: entry2 (apple pie overlap), entry3 (apple overlap), entry1 (no overlap)
    assert len(reranked) == 3
    assert reranked[0] == entry2
    assert reranked[1] == entry3
    assert reranked[2] == entry1

def test_semantic_reranker_mocked_similarity():
    # Use a custom mock similarity function that returns hardcoded scores
    mock_scores = {
        "A": 0.1,
        "B": 0.9,
        "C": 0.5
    }
    def mock_similarity_fn(q, text):
        return mock_scores.get(text, 0.0)

    plugin = SemanticRerankerPlugin(similarity_fn=mock_similarity_fn)

    context = ["A", "B", "C"]
    reranked = plugin.after_retrieval(context, "query", 1)

    # Expected order: B (0.9), C (0.5), A (0.1)
    assert reranked == ["B", "C", "A"]

def test_semantic_reranker_bootstrap_config():
    plugin = SemanticRerankerPlugin()

    # Bootstrap with a mock similarity function
    config = {
        "similarity_fn": lambda q, t: 1.0 if t == "target" else 0.0
    }
    # Create an event loop or run synchronously since bootstrap is async
    import asyncio
    asyncio.run(plugin.bootstrap(config))

    context = ["other", "target", "other_too"]
    reranked = plugin.after_retrieval(context, "query", 1)

    # "target" must be first
    assert reranked[0] == "target"

def test_context_engine_integration():
    plugin = SemanticRerankerPlugin()
    engine = ContextEngine(plugins=[plugin])

    entries = ["strawberry ice cream", "apple crumble dessert", "chocolate cake"]

    def base_retrieval(query: str, user_id: int):
        return entries

    # Query specifically targetting apple
    result = engine.retrieve_context("apple crumble", 1, base_retrieval)

    # "apple crumble dessert" must be re-ranked to the front
    assert len(result) == 3
    assert result[0] == "apple crumble dessert"

def test_semantic_reranker_empty_or_none():
    plugin = SemanticRerankerPlugin()

    assert plugin.after_retrieval([], "query", 1) == []
    assert plugin.after_retrieval(["item"], "", 1) == ["item"]
