"""Tests for MCP Kernel Taint Tracking Context Integrator V5."""

import pytest

from magda_agent.emotions.engine import PADState
from magda_agent.memory.working import MemoryEntry
from magda_agent.safety.taint_context_v5 import TaintedWorkingMemory
from magda_agent.safety.taint_tracking_v3 import TaintTrackerV3


@pytest.mark.asyncio
async def test_taint_context_propagation():
    """Test that taint tags propagate correctly through working memory."""
    tracker = TaintTrackerV3()
    memory = TaintedWorkingMemory(limit=10, tracker=tracker)

    # Create clean content
    clean_content = "This is clean memory."
    clean_entry = MemoryEntry(
        content=clean_content,
        importance=0.5,
        emotional_state=PADState(0, 0, 0),
        user_id=1
    )

    # Create tainted content
    tainted_content = tracker.taint("This is tainted memory from malicious actor.", "malicious_actor")
    tainted_entry = MemoryEntry(
        content=tainted_content,
        importance=0.9,
        emotional_state=PADState(0, 0, 0),
        user_id=1
    )

    # Add to memory
    await memory.add(clean_entry)
    await memory.add(tainted_entry)

    # Retrieve from memory
    entries = memory.get_entries(user_id=1)

    assert len(entries) == 2

    # Verify clean entry
    retrieved_clean = entries[0]
    assert retrieved_clean.content == "This is clean memory."
    assert not tracker.is_tainted(retrieved_clean.content)

    # Verify tainted entry
    retrieved_tainted = entries[1]
    assert retrieved_tainted.content == "This is tainted memory from malicious actor."
    assert tracker.is_tainted(retrieved_tainted.content)
    assert tracker.get_origins(retrieved_tainted.content) == {"malicious_actor"}


@pytest.mark.asyncio
async def test_taint_context_limit_enforcement():
    """Test that limits are still enforced with tainted memory."""
    tracker = TaintTrackerV3()
    memory = TaintedWorkingMemory(limit=2, tracker=tracker)

    for i in range(3):
        content = f"Entry {i}"
        if i == 2:
            content = tracker.taint(content, "origin_2")

        entry = MemoryEntry(
            content=content,
            importance=0.5,
            emotional_state=PADState(0, 0, 0),
            user_id=1
        )
        await memory.add(entry)

    entries = memory.get_entries(user_id=1)
    assert len(entries) == 2

    # Entry 0 should be dropped, so we have Entry 1 and Entry 2
    assert entries[0].content == "Entry 1"
    assert not tracker.is_tainted(entries[0].content)

    assert entries[1].content == "Entry 2"
    assert tracker.is_tainted(entries[1].content)
    assert tracker.get_origins(entries[1].content) == {"origin_2"}
