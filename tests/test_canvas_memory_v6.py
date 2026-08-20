import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from magda_agent.telemetry.canvas_memory_v6 import CanvasMemoryTelemetryV6

@pytest.mark.asyncio
async def test_broadcast_consolidation_event_send_text() -> None:
    """Test broadcasting an event when websocket has send_text."""
    mock_websocket = AsyncMock()
    # Ensure send_text is a coroutine mock
    mock_websocket.send_text = AsyncMock()

    telemetry = CanvasMemoryTelemetryV6(websocket=mock_websocket)

    user_id = 123
    memories = [{"id": "mem_1", "content": "test memory 1"}]

    with patch("time.time", return_value=1234567890.0):
        await telemetry.broadcast_consolidation_event(user_id, memories)

    expected_payload = {
        "type": "episodic_memory_consolidation",
        "timestamp": 1234567890.0,
        "data": {
            "user_id": user_id,
            "consolidated_count": 1,
            "memories": memories
        }
    }
    expected_message = json.dumps(expected_payload)

    mock_websocket.send_text.assert_awaited_once_with(expected_message)

@pytest.mark.asyncio
async def test_broadcast_consolidation_event_send() -> None:
    """Test broadcasting an event when websocket only has send."""
    mock_websocket = AsyncMock()
    # Remove send_text attribute to trigger fallback to send
    del mock_websocket.send_text
    mock_websocket.send = AsyncMock()

    telemetry = CanvasMemoryTelemetryV6(websocket=mock_websocket)

    user_id = 456
    memories = [{"id": "mem_2", "content": "test memory 2"}]

    with patch("time.time", return_value=1234567891.0):
        await telemetry.broadcast_consolidation_event(user_id, memories)

    expected_payload = {
        "type": "episodic_memory_consolidation",
        "timestamp": 1234567891.0,
        "data": {
            "user_id": user_id,
            "consolidated_count": 1,
            "memories": memories
        }
    }
    expected_message = json.dumps(expected_payload)

    mock_websocket.send.assert_awaited_once_with(expected_message)

@pytest.mark.asyncio
async def test_broadcast_consolidation_event_no_websocket() -> None:
    """Test broadcasting an event when no websocket is provided."""
    telemetry = CanvasMemoryTelemetryV6(websocket=None)
    telemetry.logger = MagicMock()

    user_id = 789
    memories = []

    # Should not raise an exception
    await telemetry.broadcast_consolidation_event(user_id, memories)

    # Verify logger was called
    telemetry.logger.debug.assert_called_once_with("Skipping broadcast for episodic_memory_consolidation - no websocket connected.")

@pytest.mark.asyncio
async def test_broadcast_consolidation_event_exception() -> None:
    """Test handling of exceptions during broadcast."""
    mock_websocket = AsyncMock()
    mock_websocket.send_text = AsyncMock(side_effect=Exception("Connection lost"))

    telemetry = CanvasMemoryTelemetryV6(websocket=mock_websocket)
    telemetry.logger = MagicMock()

    user_id = 101
    memories = [{"id": "mem_3", "content": "error test"}]

    # Should catch the exception and not raise
    await telemetry.broadcast_consolidation_event(user_id, memories)

    telemetry.logger.error.assert_called_once()
    assert "Failed to broadcast episodic_memory_consolidation event: Connection lost" in str(telemetry.logger.error.call_args)
