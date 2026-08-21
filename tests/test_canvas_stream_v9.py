import asyncio
import json
import logging
from unittest.mock import AsyncMock, patch

import pytest

from magda_agent.telemetry.canvas_stream_v9 import CanvasTelemetryStreamerV9


@pytest.mark.asyncio
async def test_broadcast_memory_state() -> None:
    """Tests broadcasting memory state via websocket."""
    mock_websocket = AsyncMock()
    streamer = CanvasTelemetryStreamerV9(websocket=mock_websocket)

    memory_state = {"working_memory": ["item1"], "episodic_memory": ["episode1"]}
    await streamer.broadcast_memory_state(memory_state)

    expected_payload = {
        "type": "memory_state_update",
        "data": memory_state
    }

    # Verify send_text was called with the correct JSON string
    mock_websocket.send_text.assert_called_once_with(json.dumps(expected_payload))


@pytest.mark.asyncio
async def test_broadcast_planner_step() -> None:
    """Tests broadcasting a planner step via websocket."""
    mock_websocket = AsyncMock()
    streamer = CanvasTelemetryStreamerV9(websocket=mock_websocket)

    planner_step = {"step_id": 1, "description": "Do something"}
    await streamer.broadcast_planner_step(planner_step)

    expected_payload = {
        "type": "planner_step_update",
        "data": planner_step
    }

    # Verify send_text was called with the correct JSON string
    mock_websocket.send_text.assert_called_once_with(json.dumps(expected_payload))


@pytest.mark.asyncio
async def test_no_websocket_connected() -> None:
    """Tests that broadcast gracefully handles missing websocket."""
    streamer = CanvasTelemetryStreamerV9(websocket=None)

    with patch.object(streamer.logger, "debug") as mock_debug:
        await streamer.broadcast_memory_state({"data": "test"})
        mock_debug.assert_called_with("Skipping broadcast for memory_state_update - no websocket connected.")


@pytest.mark.asyncio
async def test_websocket_send_failure() -> None:
    """Tests that broadcast gracefully handles a send exception."""
    mock_websocket = AsyncMock()
    mock_websocket.send_text.side_effect = Exception("Connection closed")
    streamer = CanvasTelemetryStreamerV9(websocket=mock_websocket)

    with patch.object(streamer.logger, "error") as mock_error:
        await streamer.broadcast_memory_state({"data": "test"})
        mock_error.assert_called_with("Failed to broadcast memory_state_update event: Connection closed")
