import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from magda_agent.telemetry.canvas_memory_v5 import CanvasMemoryTelemetryV5

@pytest.fixture
def mock_websocket_send_text():
    mock_ws = AsyncMock()
    mock_ws.send_text = AsyncMock()
    return mock_ws

@pytest.fixture
def mock_websocket_send():
    mock_ws = AsyncMock()
    # explicitly remove send_text to trigger the fallback
    del mock_ws.send_text
    mock_ws.send = AsyncMock()
    return mock_ws

@pytest.mark.asyncio
@patch('time.time', return_value=1234567890.0)
async def test_broadcast_consolidation_event_with_send_text(mock_time, mock_websocket_send_text):
    telemetry = CanvasMemoryTelemetryV5(websocket=mock_websocket_send_text)
    user_id = 42
    memories = [
        {"id": "mem_1", "text": "learned python"},
        {"id": "mem_2", "text": "fixed a bug"}
    ]

    await telemetry.broadcast_consolidation_event(user_id, memories)

    expected_payload = {
        "type": "episodic_memory_consolidation",
        "timestamp": 1234567890.0,
        "data": {
            "user_id": 42,
            "consolidated_count": 2,
            "memories": memories
        }
    }

    expected_json = json.dumps(expected_payload)
    mock_websocket_send_text.send_text.assert_awaited_once_with(expected_json)

@pytest.mark.asyncio
@patch('time.time', return_value=1234567890.0)
async def test_broadcast_consolidation_event_with_send(mock_time, mock_websocket_send):
    telemetry = CanvasMemoryTelemetryV5(websocket=mock_websocket_send)
    user_id = 42
    memories = [
        {"id": "mem_1", "text": "learned python"}
    ]

    await telemetry.broadcast_consolidation_event(user_id, memories)

    expected_payload = {
        "type": "episodic_memory_consolidation",
        "timestamp": 1234567890.0,
        "data": {
            "user_id": 42,
            "consolidated_count": 1,
            "memories": memories
        }
    }

    expected_json = json.dumps(expected_payload)
    mock_websocket_send.send.assert_awaited_once_with(expected_json)

@pytest.mark.asyncio
async def test_broadcast_no_websocket():
    telemetry = CanvasMemoryTelemetryV5(websocket=None)
    telemetry.logger = MagicMock()

    user_id = 42
    memories = [{"id": "mem_1", "text": "test"}]

    # This should not raise any exceptions
    await telemetry.broadcast_consolidation_event(user_id, memories)

    telemetry.logger.debug.assert_called_once_with("Skipping broadcast for episodic_memory_consolidation - no websocket connected.")

@pytest.mark.asyncio
async def test_broadcast_exception_handling(mock_websocket_send_text):
    mock_websocket_send_text.send_text.side_effect = Exception("WebSocket closed")
    telemetry = CanvasMemoryTelemetryV5(websocket=mock_websocket_send_text)
    telemetry.logger = MagicMock()

    user_id = 42
    memories = []

    # This should log the error and not raise it
    await telemetry.broadcast_consolidation_event(user_id, memories)

    telemetry.logger.error.assert_called_once()
    assert "Failed to broadcast episodic_memory_consolidation event: WebSocket closed" in telemetry.logger.error.call_args[0][0]
