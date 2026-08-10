import asyncio
import json
from typing import Any
import pytest
from unittest.mock import AsyncMock, MagicMock

from magda_agent.telemetry.canvas_stream import CanvasTelemetryStreamer


class MockWebsocket:
    def __init__(self):
        self.send_text = AsyncMock()


def test_initialization():
    """Test streamer initialization with and without websocket."""
    streamer_no_ws = CanvasTelemetryStreamer()
    assert streamer_no_ws.websocket is None

    mock_ws = MockWebsocket()
    streamer_with_ws = CanvasTelemetryStreamer(mock_ws)
    assert streamer_with_ws.websocket == mock_ws


def test_broadcast_memory_state():
    """Test broadcasting memory state."""
    mock_ws = MockWebsocket()
    streamer = CanvasTelemetryStreamer(mock_ws)

    memory_state = {"working_memory": ["item1", "item2"]}
    expected_payload = {
        "type": "memory_state_update",
        "data": memory_state
    }

    asyncio.run(streamer.broadcast_memory_state(memory_state))

    mock_ws.send_text.assert_called_once_with(json.dumps(expected_payload))


def test_broadcast_planner_step():
    """Test broadcasting planner step."""
    mock_ws = MockWebsocket()
    streamer = CanvasTelemetryStreamer(mock_ws)

    planner_step = {"step": 1, "action": "think"}
    expected_payload = {
        "type": "planner_step_update",
        "data": planner_step
    }

    asyncio.run(streamer.broadcast_planner_step(planner_step))

    mock_ws.send_text.assert_called_once_with(json.dumps(expected_payload))


def test_broadcast_no_websocket():
    """Test broadcast behavior when no websocket is connected."""
    streamer = CanvasTelemetryStreamer()
    # This should not raise an exception
    asyncio.run(streamer.broadcast_memory_state({"data": "test"}))


def test_broadcast_exception_handling():
    """Test handling of exceptions during broadcast."""
    mock_ws = MockWebsocket()
    mock_ws.send_text.side_effect = Exception("Connection closed")

    streamer = CanvasTelemetryStreamer(mock_ws)

    # This should catch the exception and not raise it
    asyncio.run(streamer.broadcast_planner_step({"data": "test"}))
    assert mock_ws.send_text.called
