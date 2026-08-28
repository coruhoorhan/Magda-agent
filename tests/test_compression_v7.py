import pytest
from typing import List, Dict, Any, Optional

from magda_agent.memory.compression_v7 import ClaudeContextCompressorV7
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState
from unittest.mock import MagicMock, AsyncMock

class MockLLMClientV7:
    """Mock LLM client for testing V7 compression without real API calls."""
    async def chat_completion(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        """Mocks the chat completion endpoint."""
        for msg in messages:
            if "Summarize the following memory context" in msg.get("content", ""):
                return "V7 Compressed summary."
        return "V7 fallback summary."


@pytest.mark.asyncio
async def test_v7_compress_entries_with_retrieval() -> None:
    """Tests that entries exceeding the limit are first pruned by the retriever and then compressed via LLM."""
    llm = MockLLMClientV7()

    mock_retriever = AsyncMock()
    # Mock retriever to prune the entries down to a smaller set
    # but their content still exceeds the token limit in character count
    # Let's say it returns just one entry that is still very long
    state = PADState(0, 0, 0)
    pruned_entry = MemoryEntry("A very very long entry about things that is still too long for the token limit.", 0.9, state, tags=["pruned1"], user_id=1)
    mock_retriever.prune_context.return_value = [pruned_entry]

    compressor = ClaudeContextCompressorV7(llm_client=llm, retriever=mock_retriever)

    e1 = MemoryEntry("A very very long entry about things.", 0.5, state, tags=["long1"], user_id=1)
    e2 = MemoryEntry("Another long entry about things to make it long.", 0.6, state, tags=["long2"], user_id=1)

    # 5 tokens * 4 = 20 chars limit. Combined length is > 20 chars.
    result = await compressor.compress_entries([e1, e2], token_limit=5, semantic_query="architecture")

    # Assert LLM was used for summarization
    assert result.content == "V7 Compressed summary."
    assert result.importance == 0.9
    assert result.user_id == 1
    assert "pruned1" in result.tags
    assert "semantic-enriched" in result.tags

    # Assert retriever was called
    mock_retriever.prune_context.assert_called_once_with([e1, e2], max_tokens=5, query="architecture")

@pytest.mark.asyncio
async def test_v7_compress_entries_without_retrieval() -> None:
    """Tests that entries are compressed properly when no retriever is provided."""
    llm = MockLLMClientV7()

    compressor = ClaudeContextCompressorV7(llm_client=llm)
    state = PADState(0, 0, 0)

    e1 = MemoryEntry("A very very long entry about things.", 0.5, state, tags=["long1"], user_id=1)
    e2 = MemoryEntry("Another long entry about things to make it long.", 0.6, state, tags=["long2"], user_id=1)

    # Provide no semantic_query
    result = await compressor.compress_entries([e1, e2], token_limit=10)

    # Assert LLM was used
    assert result.content == "V7 Compressed summary."
    assert "semantic-enriched" not in result.tags


@pytest.mark.asyncio
async def test_v7_compress_entries_within_limit() -> None:
    """Tests that entries within the limit are not unnecessarily summarized via LLM if they fit."""
    llm = MockLLMClientV7()

    mock_retriever = AsyncMock()
    state = PADState(0, 0, 0)

    e1 = MemoryEntry("Hello.", 0.5, state, tags=["greeting"], user_id=1)
    e2 = MemoryEntry("Goodbye.", 0.6, state, tags=["farewell"], user_id=1)

    mock_retriever.prune_context.return_value = [e1, e2]

    compressor = ClaudeContextCompressorV7(llm_client=llm, retriever=mock_retriever)

    # 2000 tokens * 4 = 8000 chars limit. So no LLM should be called.
    result = await compressor.compress_entries([e1, e2], token_limit=2000, semantic_query="greeting")

    # Assert the returned text combines the contexts and entries without LLM summarization
    expected_combined = "Hello.\nGoodbye."
    assert result.content == expected_combined
    assert result.importance == 0.55
    assert result.user_id == 1
    assert "greeting" in result.tags
    assert "farewell" in result.tags
    assert "semantic-enriched" in result.tags


@pytest.mark.asyncio
async def test_v7_empty_entries() -> None:
    """Tests that compressing an empty list of entries raises a ValueError."""
    compressor = ClaudeContextCompressorV7()
    with pytest.raises(ValueError, match="No entries to compress"):
        await compressor.compress_entries([])
