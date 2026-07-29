import pytest
from unittest.mock import MagicMock
from magda_agent.memory.virtual_context_pagination import VirtualContextPaginationPlugin
from magda_agent.memory.working import WorkingMemory, MemoryEntry
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.emotions.engine import PADState

@pytest.mark.asyncio
async def test_virtual_context_pagination_plugin_bootstrap() -> None:
    plugin = VirtualContextPaginationPlugin(max_tokens=20)
    wm = WorkingMemory()
    em = EpisodicMemory(persist_directory=":memory:")
    em.collection_name = "test_episodic_memory_pagination_bootstrap"
    em.collection = em.client.get_or_create_collection(name=em.collection_name)

    config = {
        "working_memory": wm,
        "episodic_memory": em,
        "max_tokens": 50
    }

    await plugin.bootstrap(config)

    assert plugin.working_memory == wm
    assert plugin.episodic_memory == em
    assert plugin.max_tokens == 50

@pytest.mark.asyncio
async def test_virtual_context_pagination_plugin_before_retrieval_no_pagination() -> None:
    plugin = VirtualContextPaginationPlugin(max_tokens=50)
    wm = WorkingMemory()
    em = EpisodicMemory(persist_directory=":memory:")
    em.collection_name = "test_episodic_memory_pagination_no"
    em.collection = em.client.get_or_create_collection(name=em.collection_name)

    config = {
        "working_memory": wm,
        "episodic_memory": em,
    }
    await plugin.bootstrap(config)

    state = PADState(0, 0, 0)

    e1 = MemoryEntry("short context", 0.5, state, user_id=1)
    await wm.add(e1)

    assert len(wm.get_entries(user_id=1)) == 1

    query = "test query"
    result = plugin.before_retrieval(query, user_id=1)

    assert result == query
    assert len(wm.get_entries(user_id=1)) == 1
    assert len(em.get_all_events(user_id=1)) == 0

@pytest.mark.asyncio
async def test_virtual_context_pagination_plugin_before_retrieval_with_pagination() -> None:
    plugin = VirtualContextPaginationPlugin(max_tokens=15)
    wm = WorkingMemory(limit=10)
    em = EpisodicMemory(persist_directory=":memory:")
    em.collection_name = "test_episodic_memory_pagination_yes"
    em.collection = em.client.get_or_create_collection(name=em.collection_name)

    config = {
        "working_memory": wm,
        "episodic_memory": em,
    }
    await plugin.bootstrap(config)

    state = PADState(0, 0, 0)

    # Each entry is 4 words -> ~5 tokens. Total 4 entries = ~20 tokens.
    e1 = MemoryEntry("word1 word2 word3 word4", 0.5, state, user_id=1)
    e2 = MemoryEntry("word5 word6 word7 word8", 0.6, state, user_id=1)
    e3 = MemoryEntry("word9 word10 word11 word12", 0.7, state, user_id=1)
    e4 = MemoryEntry("word13 word14 word15 word16", 0.8, state, user_id=1)

    await wm.add(e1)
    await wm.add(e2)
    await wm.add(e3)
    await wm.add(e4)

    assert len(wm.get_entries(user_id=1)) == 4
    # Check that initial token size is greater than max_tokens
    assert plugin.get_token_length(wm.get_entries(user_id=1)) > 15

    query = "trigger pagination"
    result = plugin.before_retrieval(query, user_id=1)

    assert result == query

    # After pagination, the size should be within limit (15 tokens -> ~3 entries max)
    entries = wm.get_entries(user_id=1)
    # The plugin removes half of entries per loop. It had 4, half is 2.
    # It removes e1, e2. Remaining e3, e4 (which is ~10 tokens). It stops.
    assert len(entries) == 2
    assert entries[0].content == "word9 word10 word11 word12"
    assert entries[1].content == "word13 word14 word15 word16"

    # Check episodic memory
    events = em.get_all_events(user_id=1)
    assert len(events) == 2
    paged_contents = [event["text"] for event in events]
    assert "word1 word2 word3 word4" in paged_contents
    assert "word5 word6 word7 word8" in paged_contents
    assert events[0]["metadata"]["paged_out_explicitly"] == True
