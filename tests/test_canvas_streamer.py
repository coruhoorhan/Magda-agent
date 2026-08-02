import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock
from magda_agent.visualization.canvas_streamer import CanvasMemoryStreamer
from magda_agent.memory.working import WorkingMemory, MemoryEntry

def test_canvas_streamer_connect_disconnect():
    async def run_test():
        working_memory = WorkingMemory()
        streamer = CanvasMemoryStreamer(working_memory)

        websocket = AsyncMock()
        await streamer.connect(websocket)
        websocket.accept.assert_called_once()
        assert websocket in streamer.active_connections

        streamer.disconnect(websocket)
        assert websocket not in streamer.active_connections

    asyncio.run(run_test())

def test_canvas_streamer_diff_calculation():
    working_memory = WorkingMemory()
    streamer = CanvasMemoryStreamer(working_memory)

    # State 1: Add memory entry
    entry1 = MemoryEntry("Hello", 0.8, None, user_id=1)
    working_memory._entries_by_user[1] = [entry1]

    state1 = streamer._get_current_state()
    diffs1 = streamer._calculate_diff(state1)
    assert 1 in diffs1["added"]
    assert len(diffs1["added"][1]) == 1
    assert diffs1["added"][1][0]["content"] == "Hello"

    streamer._last_state = state1

    # State 2: Update entry and add a new one
    entry1.content = "Hello World"
    entry2 = MemoryEntry("Hi", 0.5, None, user_id=1)
    working_memory._entries_by_user[1] = [entry1, entry2]

    state2 = streamer._get_current_state()
    diffs2 = streamer._calculate_diff(state2)
    assert 1 in diffs2["added"]
    assert len(diffs2["added"][1]) == 1
    assert diffs2["added"][1][0]["content"] == "Hi"

    assert 1 in diffs2["updated"]
    assert len(diffs2["updated"][1]) == 1
    assert diffs2["updated"][1][0]["content"] == "Hello World"

    streamer._last_state = state2

    # State 3: Remove entry 1
    working_memory._entries_by_user[1] = [entry2]

    state3 = streamer._get_current_state()
    diffs3 = streamer._calculate_diff(state3)
    assert 1 in diffs3["removed"]
    assert len(diffs3["removed"][1]) == 1
    assert diffs3["removed"][1][0]["id"] == entry1.id

def test_canvas_streamer_broadcast():
    async def run_test():
        working_memory = WorkingMemory()
        streamer = CanvasMemoryStreamer(working_memory)

        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock() # This one will fail

        ws3.send_text.side_effect = Exception("Connection closed")

        await streamer.connect(ws1)
        await streamer.connect(ws2)
        await streamer.connect(ws3)

        await streamer.broadcast("test message")

        ws1.send_text.assert_called_once_with("test message")
        ws2.send_text.assert_called_once_with("test message")
        ws3.send_text.assert_called_once_with("test message")

        assert ws1 in streamer.active_connections
        assert ws2 in streamer.active_connections
        assert ws3 not in streamer.active_connections # Should be disconnected on error

    asyncio.run(run_test())
