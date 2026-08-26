from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from magda_agent.memory.virtual_context_v4 import VirtualContextManagerV4
from magda_agent.memory.working import WorkingMemory, MemoryEntry
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.emotions.engine import PADState

@pytest.mark.asyncio
async def test_virtual_context_v4_compress_without_llm() -> None:
    vcm = VirtualContextManagerV4()
    state = PADState(0.2, 0.4, 0.6)
    e1 = MemoryEntry("First item text", 0.4, state, tags=["tag1"], user_id=1)
    e2 = MemoryEntry("Second item text", 0.8, state, tags=["tag2"], user_id=1)

    summary = await vcm.compress_context([e1, e2])
    assert "Summary of 2 items: First item text\nSecond item text" in summary.content
    assert pytest.approx(summary.importance) == 0.6
    assert summary.user_id == 1
    assert set(summary.tags) == {"tag1", "tag2"}

@pytest.mark.asyncio
async def test_virtual_context_v4_compress_with_llm() -> None:
    mock_llm = AsyncMock()
    mock_llm.chat_completion.return_value = "LLM Compressed Summary V4"
    vcm = VirtualContextManagerV4(llm_client=mock_llm)

    state = PADState(0.1, 0.3, 0.5)
    e1 = MemoryEntry("First item text", 0.3, state, user_id=2)
    e2 = MemoryEntry("Second item text", 0.7, state, user_id=2)

    summary = await vcm.compress_context([e1, e2])
    assert summary.content == "LLM Compressed Summary V4"
    assert summary.user_id == 2
    mock_llm.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_virtual_context_v4_selective_page_out_importance() -> None:
    wm = WorkingMemory(limit=10)
    em = EpisodicMemory(persist_directory=":memory:")
    em.collection_name = "test_episodic_memory_v4_selective"
    em.collection = em.client.get_or_create_collection(name=em.collection_name)
    vcm = VirtualContextManagerV4(default_importance_threshold=0.5)

    state = PADState(0, 0, 0)
    e_low = MemoryEntry("Low importance item", 0.2, state, user_id=1)
    e_high = MemoryEntry("High importance item", 0.9, state, user_id=1)

    await wm.add(e_low)
    await wm.add(e_high)

    assert len(wm.get_entries(user_id=1)) == 2

    # Perform selective page out. Should prioritize e_low
    await vcm.selective_page_out(wm, em, user_id=1, count=1)

    remaining = wm.get_entries(user_id=1)
    assert len(remaining) == 1
    assert remaining[0].content == "High importance item"

    events = em.get_all_events(user_id=1)
    assert len(events) == 1
    assert events[0]["text"] == "Low importance item"
    assert events[0]["metadata"]["paged_out_selectively"] is True

@pytest.mark.asyncio
async def test_virtual_context_v4_selective_page_in() -> None:
    wm = WorkingMemory(limit=5)
    em = EpisodicMemory(persist_directory=":memory:")
    em.collection_name = "test_episodic_memory_v4_page_in"
    em.collection = em.client.get_or_create_collection(name=em.collection_name)
    vcm = VirtualContextManagerV4()

    em.store_event("Stored memory fact for retrieval", metadata={"paged_out_selectively": True}, user_id=3)

    await vcm.selective_page_in(wm, em, user_id=3, query="Stored memory", top_k=1)

    entries = wm.get_entries(user_id=3)
    assert len(entries) == 1
    assert entries[0].content == "Stored memory fact for retrieval"
    assert "paged_in" in entries[0].tags

@pytest.mark.asyncio
async def test_virtual_context_v4_maintain_working_memory_limits() -> None:
    wm = WorkingMemory(limit=10)
    em = EpisodicMemory(persist_directory=":memory:")
    em.collection_name = "test_episodic_memory_v4_limits"
    em.collection = em.client.get_or_create_collection(name=em.collection_name)
    vcm = VirtualContextManagerV4(default_max_tokens=10)

    state = PADState(0, 0, 0)
    e1 = MemoryEntry("word1 word2 word3 word4", 0.3, state, user_id=1)
    e2 = MemoryEntry("word5 word6 word7 word8", 0.4, state, user_id=1)
    e3 = MemoryEntry("word9 word10 word11 word12", 0.9, state, user_id=1)

    await wm.add(e1)
    await wm.add(e2)
    await wm.add(e3)

    assert vcm.get_token_length(wm.get_entries(user_id=1)) > 10

    await vcm.maintain_working_memory_limits(wm, em, user_id=1, max_tokens=10)

    remaining = wm.get_entries(user_id=1)
    assert len(remaining) < 3
    assert vcm.get_token_length(remaining) <= 10
