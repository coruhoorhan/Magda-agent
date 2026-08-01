import pytest
from unittest.mock import AsyncMock
from magda_agent.memory.eviction_manager import LettaVirtualContextEvictionManager
from magda_agent.memory.working import WorkingMemory, MemoryEntry
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.emotions.engine import PADState

@pytest.mark.asyncio
async def test_eviction_manager_bootstrap() -> None:
    llm_mock = AsyncMock()
    plugin = LettaVirtualContextEvictionManager(max_tokens=20, llm=llm_mock)
    wm = WorkingMemory()
    em = EpisodicMemory(persist_directory=":memory:")

    config = {
        "working_memory": wm,
        "episodic_memory": em,
        "max_tokens": 50,
        "llm": llm_mock
    }

    await plugin.bootstrap(config)

    assert plugin.working_memory == wm
    assert plugin.episodic_memory == em
    assert plugin.max_tokens == 50
    assert plugin.llm == llm_mock

@pytest.mark.asyncio
async def test_eviction_manager_summarize_and_page_out() -> None:
    llm_mock = AsyncMock()
    llm_mock.chat_completion.return_value = "Mocked summary block"

    plugin = LettaVirtualContextEvictionManager(max_tokens=15, llm=llm_mock)
    wm = WorkingMemory(limit=10)
    em = EpisodicMemory(persist_directory=":memory:")

    await plugin.bootstrap({
        "working_memory": wm,
        "episodic_memory": em,
    })

    state = PADState(0, 0, 0)

    e1 = MemoryEntry("word1 word2 word3 word4", 0.5, state, user_id=1)
    e2 = MemoryEntry("word5 word6 word7 word8", 0.6, state, user_id=1)
    e3 = MemoryEntry("word9 word10 word11 word12", 0.7, state, user_id=1)
    e4 = MemoryEntry("word13 word14 word15 word16", 0.8, state, user_id=1)

    await wm.add(e1)
    await wm.add(e2)
    await wm.add(e3)
    await wm.add(e4)

    await plugin.summarize_and_page_out(user_id=1)

    entries = wm.get_entries(user_id=1)
    assert len(entries) == 2
    assert entries[0].content == "word9 word10 word11 word12"
    assert entries[1].content == "word13 word14 word15 word16"

    events = em.get_all_events(user_id=1)
    assert len(events) == 1
    assert events[0]["text"] == "Mocked summary block"
    assert events[0]["metadata"]["paged_out_explicitly"] == True
    assert events[0]["metadata"]["summarized"] == True
    assert events[0]["metadata"]["original_items"] == 2
    assert events[0]["metadata"]["importance"] == 0.55

    llm_mock.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_eviction_manager_compact() -> None:
    llm_mock = AsyncMock()
    llm_mock.chat_completion.return_value = "Mocked summary block"

    plugin = LettaVirtualContextEvictionManager(max_tokens=15, llm=llm_mock)
    wm = WorkingMemory(limit=10)
    em = EpisodicMemory(persist_directory=":memory:")

    await plugin.bootstrap({
        "working_memory": wm,
        "episodic_memory": em,
    })

    state = PADState(0, 0, 0)

    e1 = MemoryEntry("word1 word2 word3 word4", 0.5, state, user_id=1)
    e2 = MemoryEntry("word5 word6 word7 word8", 0.6, state, user_id=1)
    e3 = MemoryEntry("word9 word10 word11 word12", 0.7, state, user_id=1)
    e4 = MemoryEntry("word13 word14 word15 word16", 0.8, state, user_id=1)

    await wm.add(e1)
    await wm.add(e2)
    await wm.add(e3)
    await wm.add(e4)

    assert plugin.get_token_length(wm.get_entries(user_id=1)) > 15

    # We call compact hook
    metadata = {"user_id": 1}
    new_items = await plugin.compact(wm.get_entries(user_id=1), metadata)

    assert len(new_items) == 2

    entries = wm.get_entries(user_id=1)
    assert len(entries) == 2
