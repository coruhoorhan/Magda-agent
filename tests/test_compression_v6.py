import pytest
import asyncio
from typing import List, Dict, Any, Optional

from magda_agent.memory.compression_v6 import ClaudeContextCompressorV6
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState
from unittest.mock import MagicMock


class MockLLMClientV6:
    """Mock LLM client for testing V6 compression without real API calls."""
    async def chat_completion(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        """Mocks the chat completion endpoint."""
        for msg in messages:
            if "Summarize the following memory context" in msg.get("content", ""):
                if "Retrieved Semantic Context:" in msg["content"]:
                    return "V6 Compressed summary with semantic context."
                return "V6 Compressed summary without semantic context."
        return "V6 fallback summary."


@pytest.mark.asyncio
async def test_v6_compress_entries_with_retrieval() -> None:
    """Tests that entries exceeding the limit trigger the LLM for summarization and integrate semantic context."""
    llm = MockLLMClientV6()

    mock_retriever = MagicMock()
    # Mock to return some context when queried
    mock_retriever.retrieve_relevant_context.return_value = ["Previous discussion about architecture.", "Related code snippet."]

    compressor = ClaudeContextCompressorV6(llm_client=llm, retriever=mock_retriever)
    state = PADState(0, 0, 0)

    # Make the combined content larger than 10 tokens (40 characters)
    e1 = MemoryEntry("A very very long entry about things.", 0.5, state, tags=["long1"], user_id=1)
    e2 = MemoryEntry("Another long entry about things to make it long.", 0.6, state, tags=["long2"], user_id=1)

    # 10 tokens * 4 = 40 chars limit. Combined length + context is > 40 chars.
    result = await compressor.compress_entries([e1, e2], token_limit=10, semantic_query="architecture")

    # Assert LLM was used and injected context
    assert result.content == "V6 Compressed summary with semantic context."
    assert result.importance == 0.55
    assert result.user_id == 1
    assert "long1" in result.tags
    assert "long2" in result.tags
    assert "semantic-enriched" in result.tags

    # Assert retriever was called
    mock_retriever.retrieve_relevant_context.assert_called_once_with("architecture", user_id=1)


@pytest.mark.asyncio
async def test_v6_compress_entries_without_retrieval() -> None:
    """Tests that entries are compressed properly when no query is provided (behaves like V5)."""
    llm = MockLLMClientV6()
    mock_retriever = MagicMock()

    compressor = ClaudeContextCompressorV6(llm_client=llm, retriever=mock_retriever)
    state = PADState(0, 0, 0)

    e1 = MemoryEntry("A very very long entry about things.", 0.5, state, tags=["long1"], user_id=1)
    e2 = MemoryEntry("Another long entry about things to make it long.", 0.6, state, tags=["long2"], user_id=1)

    # Provide no semantic_query
    result = await compressor.compress_entries([e1, e2], token_limit=10)

    # Assert LLM was used but no context injected
    assert result.content == "V6 Compressed summary without semantic context."
    assert "semantic-enriched" not in result.tags

    # Assert retriever was NOT called
    mock_retriever.retrieve_relevant_context.assert_not_called()


@pytest.mark.asyncio
async def test_v6_compress_entries_within_limit() -> None:
    """Tests that entries within the limit are not unnecessarily summarized via LLM if they fit, even with semantic query."""
    llm = MockLLMClientV6()
    mock_retriever = MagicMock()
    mock_retriever.retrieve_relevant_context.return_value = ["Small context."]

    compressor = ClaudeContextCompressorV6(llm_client=llm, retriever=mock_retriever)
    state = PADState(0, 0, 0)

    e1 = MemoryEntry("Hello.", 0.5, state, tags=["greeting"], user_id=1)
    e2 = MemoryEntry("Goodbye.", 0.6, state, tags=["farewell"], user_id=1)

    # 2000 tokens * 4 = 8000 chars limit. So no LLM should be called.
    result = await compressor.compress_entries([e1, e2], token_limit=2000, semantic_query="greeting")

    # Assert the returned text combines the contexts and entries without LLM summarization
    expected_combined = "Retrieved Semantic Context:\nSmall context.\n\nRecent Memory:\nHello.\nGoodbye."
    assert result.content == expected_combined
    assert result.importance == 0.55
    assert result.user_id == 1
    assert "greeting" in result.tags
    assert "farewell" in result.tags
    assert "semantic-enriched" in result.tags


@pytest.mark.asyncio
async def test_v6_empty_entries() -> None:
    """Tests that compressing an empty list of entries raises a ValueError."""
    compressor = ClaudeContextCompressorV6()
    with pytest.raises(ValueError, match="No entries to compress"):
        await compressor.compress_entries([])
