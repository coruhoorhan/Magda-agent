import pytest
import asyncio
from magda_agent.memory.compression_v5 import ClaudeContextCompressorV5
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState
from typing import List, Dict, Any

class MockLLMClientV5:
    """Mock LLM client for testing V5 compression without real API calls."""
    async def chat_completion(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        """Mocks the chat completion endpoint."""
        return "V5 Compressed summary."

@pytest.mark.asyncio
async def test_v5_compress_entries_within_limit() -> None:
    """Tests that entries within the limit are not unnecessarily summarized via LLM if they fit."""
    llm = MockLLMClientV5()
    compressor = ClaudeContextCompressorV5(llm_client=llm)
    state = PADState(0, 0, 0)

    e1 = MemoryEntry("User says hello.", 0.5, state, tags=["greeting"], user_id=1)
    e2 = MemoryEntry("User says goodbye.", 0.6, state, tags=["farewell"], user_id=1)

    # 2000 tokens * 4 = 8000 chars limit. So no LLM should be called.
    result = await compressor.compress_entries([e1, e2], token_limit=2000)
    assert result.content == "User says hello.\nUser says goodbye."
    assert result.importance == 0.55
    assert result.user_id == 1
    assert "greeting" in result.tags
    assert "farewell" in result.tags

@pytest.mark.asyncio
async def test_v5_compress_entries_exceeds_limit() -> None:
    """Tests that entries exceeding the limit trigger the LLM for summarization."""
    llm = MockLLMClientV5()
    compressor = ClaudeContextCompressorV5(llm_client=llm)
    state = PADState(0, 0, 0)

    # Make the combined content larger than 10 tokens (40 characters)
    e1 = MemoryEntry("A very very long entry about things.", 0.5, state, tags=["long1"], user_id=1)
    e2 = MemoryEntry("Another long entry about things to make it long.", 0.6, state, tags=["long2"], user_id=1)

    # 10 tokens * 4 = 40 chars limit. Combined length is > 40 chars.
    result = await compressor.compress_entries([e1, e2], token_limit=10)
    assert result.content == "V5 Compressed summary."
    assert result.importance == 0.55
    assert result.user_id == 1
    assert "long1" in result.tags
    assert "long2" in result.tags

@pytest.mark.asyncio
async def test_v5_compress_workflow() -> None:
    """Tests that v5 compress_workflow respects token limits and handles fallbacks."""
    llm = MockLLMClientV5()
    compressor = ClaudeContextCompressorV5(llm_client=llm)

    # Test within limit
    short_context = "This is a short workflow context."
    result = await compressor.compress_workflow(short_context, 100)
    assert result == short_context

    # Test exceeding limit with LLM
    long_context = "A" * 500  # 500 characters
    result_llm = await compressor.compress_workflow(long_context, 10) # 10 tokens * 4 = 40 chars limit
    assert result_llm == "V5 Compressed summary."

    # Test exceeding limit without LLM (fallback to truncation)
    compressor_no_llm = ClaudeContextCompressorV5(llm_client=None)
    result_no_llm = await compressor_no_llm.compress_workflow(long_context, 10)
    assert result_no_llm.endswith("... [TRUNCATED]")
    assert len(result_no_llm) == 40 + len("... [TRUNCATED]")

@pytest.mark.asyncio
async def test_v5_empty_entries() -> None:
    """Tests that compressing an empty list of entries raises a ValueError."""
    compressor = ClaudeContextCompressorV5()
    with pytest.raises(ValueError, match="No entries to compress"):
        await compressor.compress_entries([])
