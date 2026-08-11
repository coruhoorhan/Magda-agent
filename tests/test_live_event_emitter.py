import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from magda_agent.telemetry.live_event_emitter import LiveEventEmitter


class MockWebsocket:
    def __init__(self):
        self.send_text = AsyncMock()


class MockWebsocketFallback:
    def __init__(self):
        self.send = AsyncMock()


def test_initialization():
    """Test streamer initialization with and without websocket."""
    emitter_no_ws = LiveEventEmitter()
    assert emitter_no_ws.websocket is None

    mock_ws = MockWebsocket()
    emitter_with_ws = LiveEventEmitter(mock_ws)
    assert emitter_with_ws.websocket == mock_ws


def test_set_websocket():
    """Test updating the websocket connection dynamically."""
    emitter = LiveEventEmitter()
    assert emitter.websocket is None

    mock_ws = MockWebsocket()
    emitter.set_websocket(mock_ws)
    assert emitter.websocket == mock_ws


def test_emit_planner_change():
    """Test emitting planner change."""
    mock_ws = MockWebsocket()
    emitter = LiveEventEmitter(mock_ws)

    planner_state = {"current_step": "research_topic", "status": "active"}

    # Run the emitter
    asyncio.run(emitter.emit_planner_change(planner_state))

    assert mock_ws.send_text.called
    sent_payload_str = mock_ws.send_text.call_args[0][0]
    payload = json.loads(sent_payload_str)

    assert payload["type"] == "planner_change"
    assert payload["category"] == "planner"
    assert "timestamp" in payload
    assert isinstance(payload["timestamp"], (int, float))
    assert payload["data"] == planner_state


def test_emit_memory_change():
    """Test emitting memory change."""
    mock_ws = MockWebsocket()
    emitter = LiveEventEmitter(mock_ws)

    memory_state = {"episodic_count": 42}

    # Run the emitter
    asyncio.run(emitter.emit_memory_change(memory_state))

    assert mock_ws.send_text.called
    sent_payload_str = mock_ws.send_text.call_args[0][0]
    payload = json.loads(sent_payload_str)

    assert payload["type"] == "memory_change"
    assert payload["category"] == "memory"
    assert "timestamp" in payload
    assert payload["data"] == memory_state


def test_emit_execution_change():
    """Test emitting execution change."""
    mock_ws = MockWebsocket()
    emitter = LiveEventEmitter(mock_ws)

    execution_state = {"tool_name": "search", "args": {"query": "pytest"}}

    # Run the emitter
    asyncio.run(emitter.emit_execution_change(execution_state))

    assert mock_ws.send_text.called
    sent_payload_str = mock_ws.send_text.call_args[0][0]
    payload = json.loads(sent_payload_str)

    assert payload["type"] == "execution_change"
    assert payload["category"] == "execution"
    assert "timestamp" in payload
    assert payload["data"] == execution_state


def test_emit_fallback_send():
    """Test that standard send is used if send_text is absent on websocket."""
    mock_ws = MockWebsocketFallback()
    emitter = LiveEventEmitter(mock_ws)

    data = {"msg": "hello"}
    asyncio.run(emitter.emit_state_change("custom", data))

    assert mock_ws.send.called
    sent_payload_str = mock_ws.send.call_args[0][0]
    payload = json.loads(sent_payload_str)

    assert payload["type"] == "custom_change"
    assert payload["category"] == "custom"
    assert payload["data"] == data


def test_emit_no_websocket():
    """Test that emitting works gracefully without websocket connected."""
    emitter = LiveEventEmitter()
    # Should run fine without exceptions
    asyncio.run(emitter.emit_planner_change({"key": "val"}))


def test_emit_exception_handling():
    """Test that exceptions raised by websocket are handled gracefully."""
    mock_ws = MockWebsocket()
    mock_ws.send_text.side_effect = Exception("Websocket connection closed unexpectedly")

    emitter = LiveEventEmitter(mock_ws)

    # Should run fine without rising exceptions
    asyncio.run(emitter.emit_planner_change({"key": "val"}))
    assert mock_ws.send_text.called
