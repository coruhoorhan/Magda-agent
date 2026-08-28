import pytest
from typing import List, Dict, Any, Optional

from magda_agent.memory.compression_v8 import ClaudeContextCompressorV8
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState
from unittest.mock import MagicMock, AsyncMock, patch

def llm_generate_side_effect(prompt, temperature=0.0):
    if "Summarize the following memory context" in prompt:
        return "V8 Compressed summary."
    return "V8 fallback summary."

@pytest.mark.asyncio
@patch('magda_agent.llm_client.LLMClient.generate', new_callable=AsyncMock)
async def test_v8_compress_entries_exceeding_limit(mock_generate) -> None:
    """Tests that entries exceeding the limit are recursively shrunk."""
    mock_generate.side_effect = llm_generate_side_effect
    from magda_agent.llm_client import LLMClient
    llm = LLMClient()

    compressor = ClaudeContextCompressorV8(llm_client=llm)
    state = PADState(0, 0, 0)

    e1 = MemoryEntry("A very very long entry about things.", 0.5, state, tags=["long1"], user_id=1)
    e2 = MemoryEntry("Another long entry about things to make it long.", 0.6, state, tags=["long2"], user_id=1)
    e3 = MemoryEntry("Third long entry that will cause recursive shrinkage.", 0.7, state, tags=["long3"], user_id=1)

    # 10 tokens * 4 = 40 chars limit. Combined length is > 40 chars.
    result = await compressor.compress_entries([e1, e2, e3], token_limit=10)

    # Assert LLM was used for summarization
    # Since it's recursive, e1+e2 compressed to "V8 Compressed summary." -> len = 22
    # Then combined with e3: "V8 Compressed summary.\nThird long entry that will cause recursive shrinkage." -> len = 77
    # 77 > 40, so it will summarize again.
    # Result should be "V8 Compressed summary."
    assert result.content == "V8 Compressed summary."
    assert result.importance == 0.6
    assert result.user_id == 1
    assert "long1" in result.tags
    assert "long2" in result.tags
    assert "long3" in result.tags

@pytest.mark.asyncio
@patch('magda_agent.llm_client.LLMClient.generate', new_callable=AsyncMock)
async def test_v8_compress_entries_within_limit(mock_generate) -> None:
    """Tests that entries within the limit are not unnecessarily summarized via LLM if they fit."""
    mock_generate.side_effect = llm_generate_side_effect
    from magda_agent.llm_client import LLMClient
    llm = LLMClient()

    compressor = ClaudeContextCompressorV8(llm_client=llm)
    state = PADState(0, 0, 0)

    e1 = MemoryEntry("Hello.", 0.5, state, tags=["greeting"], user_id=1)
    e2 = MemoryEntry("Goodbye.", 0.6, state, tags=["farewell"], user_id=1)

    # 2000 tokens * 4 = 8000 chars limit. So no LLM should be called.
    result = await compressor.compress_entries([e1, e2], token_limit=2000)

    # Assert the returned text combines the entries without LLM summarization
    expected_combined = "Hello.\nGoodbye."
    assert result.content == expected_combined
    assert result.importance == 0.55
    assert result.user_id == 1
    assert "greeting" in result.tags
    assert "farewell" in result.tags


@pytest.mark.asyncio
async def test_v8_empty_entries() -> None:
    """Tests that compressing an empty list of entries raises a ValueError."""
    compressor = ClaudeContextCompressorV8()
    with pytest.raises(ValueError, match="No entries to compress"):
        await compressor.compress_entries([])
