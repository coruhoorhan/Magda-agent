import pytest
from unittest.mock import AsyncMock

from magda_agent.emotions.engine import PADState
from magda_agent.memory.working import MemoryEntry
from magda_agent.memory.virtual_context_compression_v1 import OpenClawVirtualContextCompressionHook
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_compress_under_threshold() -> None:
    llm_mock = AsyncMock(spec=LLMClient)
    hook = OpenClawVirtualContextCompressionHook(llm=llm_mock, token_threshold=2000)

    state = PADState(0.1, 0.2, 0.3)
    e1 = MemoryEntry("short context 1", importance=0.5, emotional_state=state, user_id=1)
    e2 = MemoryEntry("short context 2", importance=0.5, emotional_state=state, user_id=1)

    entries = [e1, e2]

    # Check length logic ~6 words = ~7 tokens < 2000
    assert hook._get_token_length(entries) < 2000

    new_entries = await hook.compress(entries)

    # Should not be compressed
    assert new_entries == entries
    llm_mock.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_compress_over_threshold() -> None:
    llm_mock = AsyncMock(spec=LLMClient)
    llm_mock.chat_completion.return_value = "This is a summarized context by the LLM."

    # Very small threshold to trigger compression easily
    hook = OpenClawVirtualContextCompressionHook(llm=llm_mock, token_threshold=10)

    state = PADState(0.1, 0.2, 0.3)

    # 5 words per entry = ~6 tokens each
    e1 = MemoryEntry("word1 word2 word3 word4 word5", importance=0.5, emotional_state=state, user_id=1)
    e2 = MemoryEntry("word6 word7 word8 word9 word10", importance=0.6, emotional_state=state, user_id=1)
    e3 = MemoryEntry("word11 word12 word13 word14 word15", importance=0.7, emotional_state=state, user_id=1)
    e4 = MemoryEntry("word16 word17 word18 word19 word20", importance=0.8, emotional_state=state, user_id=1)

    entries = [e1, e2, e3, e4]

    # Total tokens: 20 words * 1.3 = 26 tokens, > 10 threshold
    original_tokens = hook._get_token_length(entries)
    assert original_tokens > 10

    new_entries = await hook.compress(entries)

    # Compression: half of 4 is 2. So e1 and e2 are compressed into 1 entry. e3 and e4 are kept.
    # Total new entries should be 3.
    assert len(new_entries) == 3

    assert new_entries[0].content == "This is a summarized context by the LLM."
    assert new_entries[1] == e3
    assert new_entries[2] == e4

    # Verify LLM was called
    llm_mock.chat_completion.assert_called_once()

    # Verify importance and emotion averaging
    assert new_entries[0].importance == (0.5 + 0.6) / 2
    assert new_entries[0].emotional_state.pleasure == (0.1 + 0.1) / 2

    # Verify token reduction
    new_tokens = hook._get_token_length(new_entries)
    assert new_tokens < original_tokens

@pytest.mark.asyncio
async def test_compress_empty_list() -> None:
    llm_mock = AsyncMock(spec=LLMClient)
    hook = OpenClawVirtualContextCompressionHook(llm=llm_mock, token_threshold=10)

    new_entries = await hook.compress([])
    assert new_entries == []
    llm_mock.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_compress_llm_failure_fallback() -> None:
    llm_mock = AsyncMock(spec=LLMClient)
    llm_mock.chat_completion.side_effect = Exception("LLM API Error")

    hook = OpenClawVirtualContextCompressionHook(llm=llm_mock, token_threshold=5)

    state = PADState(0.1, 0.2, 0.3)

    e1 = MemoryEntry("long word1 word2 word3 word4 word5", importance=0.5, emotional_state=state, user_id=1)
    e2 = MemoryEntry("long word6 word7 word8 word9 word10", importance=0.6, emotional_state=state, user_id=1)

    entries = [e1, e2]

    new_entries = await hook.compress(entries)

    # Should fallback to basic string slicing
    assert len(new_entries) == 2  # half of 2 is 1. 1 compressed, 1 kept.
    assert "Summary of 1 items:" in new_entries[0].content
    assert new_entries[1] == e2
