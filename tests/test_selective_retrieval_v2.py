import pytest
from unittest.mock import MagicMock
from magda_agent.memory.selective_retrieval_v2 import SelectiveRetrievalV2

def test_selective_retrieval_filters_by_threshold():
    """Test that context retrieval identifies relevant episodic memory and filters out noise."""
    mock_memory = MagicMock()
    mock_collection = MagicMock()
    mock_memory.collection = mock_collection

    # Mocking ChromaDB query results: distances are L2 (lower is better)
    mock_collection.query.return_value = {
        "documents": [["Relevant Mem", "Noise Mem 1", "Noise Mem 2"]],
        "distances": [[0.5, 2.0, 1.8]],
        "metadatas": [[{"decayed": False}, {"decayed": False}, {"decayed": False}]]
    }

    retriever = SelectiveRetrievalV2(episodic_memory=mock_memory, similarity_threshold=1.5, max_results=5)
    results = retriever.retrieve_relevant_context("query string")

    # Only "Relevant Mem" has distance 0.5 <= 1.5
    assert len(results) == 1
    assert results[0] == "Relevant Mem"

def test_selective_retrieval_applies_priority_scaling():
    """Test that meta priority modifies the ranking."""
    mock_memory = MagicMock()
    mock_collection = MagicMock()
    mock_memory.collection = mock_collection

    mock_collection.query.return_value = {
        "documents": [["Mem A", "Mem B"]],
        "distances": [[1.2, 1.4]],
        "metadatas": [[{"priority": 1.0}, {"priority": 2.0}]]
    }

    retriever = SelectiveRetrievalV2(episodic_memory=mock_memory, similarity_threshold=1.5, max_results=5)
    results = retriever.retrieve_relevant_context("query string")

    # Mem A adjusted score = 1.2 / 1.0 = 1.2
    # Mem B adjusted score = 1.4 / 2.0 = 0.7 (Better)
    assert len(results) == 2
    assert results[0] == "Mem B"
    assert results[1] == "Mem A"

def test_selective_retrieval_empty_results():
    """Test behavior when no results are found."""
    mock_memory = MagicMock()
    mock_collection = MagicMock()
    mock_memory.collection = mock_collection

    mock_collection.query.return_value = {"documents": []}

    retriever = SelectiveRetrievalV2(episodic_memory=mock_memory)
    results = retriever.retrieve_relevant_context("query string")

    assert len(results) == 0
