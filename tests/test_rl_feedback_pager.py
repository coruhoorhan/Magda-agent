import pytest
from unittest.mock import MagicMock
from magda_agent.memory.rl_feedback_pager import RLFeedbackPagerPlugin
from magda_agent.memory.working import WorkingMemory, MemoryEntry
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.emotions.engine import PADState
import time

@pytest.mark.asyncio
async def test_rl_feedback_pager_bootstrap() -> None:
    plugin = RLFeedbackPagerPlugin(max_tokens=20)
    wm = WorkingMemory()
    em = EpisodicMemory(persist_directory=":memory:")
    em.collection_name = "test_rl_episodic_memory_bootstrap"
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
async def test_rl_feedback_pager_no_pagination() -> None:
    plugin = RLFeedbackPagerPlugin(max_tokens=50)
    wm = WorkingMemory()
    em = EpisodicMemory(persist_directory=":memory:")
    em.collection_name = "test_rl_episodic_memory_no_pagination"
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
async def test_rl_feedback_pager_with_feedback_prioritization() -> None:
    plugin = RLFeedbackPagerPlugin(max_tokens=15)
    wm = WorkingMemory(limit=10)
    em = EpisodicMemory(persist_directory=":memory:")
    em.collection_name = "test_rl_episodic_memory_prioritization"
    em.collection = em.client.get_or_create_collection(name=em.collection_name)

    config = {
        "working_memory": wm,
        "episodic_memory": em,
    }
    await plugin.bootstrap(config)

    state = PADState(0, 0, 0)

    # 4 entries, each around 5 tokens. Total ~20 tokens.
    e1 = MemoryEntry("word1 word2 word3 word4", 0.5, state, user_id=1)
    e1.user_feedback_score = 1.0  # High score, older
    time.sleep(0.01)

    e2 = MemoryEntry("word5 word6 word7 word8", 0.6, state, user_id=1)
    e2.user_feedback_score = -1.0 # Very negative score
    time.sleep(0.01)

    e3 = MemoryEntry("word9 word10 word11 word12", 0.7, state, user_id=1)
    e3.user_feedback_score = 2.0  # Very high score
    time.sleep(0.01)

    e4 = MemoryEntry("word13 word14 word15 word16", 0.8, state, user_id=1)
    # e4 has no explicit user_feedback_score, defaults to 0.0
    time.sleep(0.01)

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

    # After pagination, the size should be within limit (15 tokens -> ~2 entries max out of 4 to stay <= 15)
    # The plugin removes half of entries per loop. It had 4, half is 2.
    # It sorts by score: e2 (-1.0), e4 (0.0), e1 (1.0), e3 (2.0)
    # So it should remove e2 and e4.
    # Remaining should be e1 and e3.
    entries = wm.get_entries(user_id=1)
    assert len(entries) == 2

    remaining_contents = [e.content for e in entries]
    assert "word1 word2 word3 word4" in remaining_contents # e1 kept
    assert "word9 word10 word11 word12" in remaining_contents # e3 kept

    # Check episodic memory
    events = em.get_all_events(user_id=1)
    assert len(events) == 2
    paged_contents = [event["text"] for event in events]
    assert "word5 word6 word7 word8" in paged_contents # e2 removed
    assert "word13 word14 word15 word16" in paged_contents # e4 removed
    assert events[0]["metadata"]["paged_out_explicitly"] == True
