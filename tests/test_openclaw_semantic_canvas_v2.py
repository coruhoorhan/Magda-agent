import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from magda_agent.visualization.openclaw_semantic_canvas_v2 import OpenClawSemanticCanvasV2

@pytest.fixture
def canvas() -> OpenClawSemanticCanvasV2:
    return OpenClawSemanticCanvasV2(host="127.0.0.1", port=8766)

@pytest.mark.asyncio
async def test_canvas_handler_adds_removes_clients(canvas: OpenClawSemanticCanvasV2) -> None:
    mock_websocket = AsyncMock()
    mock_websocket.remote_address = ("127.0.0.1", 12345)

    # Run the handler but control wait_closed execution
    future = asyncio.Future()

    async def wait_closed_impl():
        assert mock_websocket in canvas.clients
        future.set_result(True)
        return None

    mock_websocket.wait_closed.side_effect = wait_closed_impl

    await canvas._handler(mock_websocket)

    assert future.done()
    assert mock_websocket not in canvas.clients

import json

@pytest.mark.asyncio
async def test_canvas_start_stop(canvas: OpenClawSemanticCanvasV2) -> None:
    with patch("magda_agent.visualization.openclaw_semantic_canvas_v2.serve", new_callable=AsyncMock) as mock_serve:
        mock_server = MagicMock()
        mock_server.wait_closed = AsyncMock()
        mock_serve.return_value = mock_server

        await canvas.start()
        mock_serve.assert_called_once_with(canvas._handler, "127.0.0.1", 8766)

        await canvas.stop()
        mock_server.close.assert_called_once()
        mock_server.wait_closed.assert_awaited_once()

@pytest.mark.asyncio
async def test_broadcast_semantic_event_no_clients(canvas: OpenClawSemanticCanvasV2) -> None:
    await canvas.broadcast_semantic_event("test_event", {"key": "value"})

@pytest.mark.asyncio
async def test_broadcast_semantic_event_with_clients(canvas: OpenClawSemanticCanvasV2) -> None:
    mock_client1 = AsyncMock()
    mock_client1.closed = False
    mock_client2 = AsyncMock()
    mock_client2.closed = False

    canvas.clients.add(mock_client1)
    canvas.clients.add(mock_client2)

    event_type = "concept_added"
    event_data = {"concept": "agent", "confidence": 0.9}

    await canvas.broadcast_semantic_event(event_type, event_data)

    expected_message = json.dumps({"type": event_type, "data": event_data})

    mock_client1.send.assert_awaited_once_with(expected_message)
    mock_client2.send.assert_awaited_once_with(expected_message)

@pytest.mark.asyncio
async def test_broadcast_semantic_event_serialization_error(canvas: OpenClawSemanticCanvasV2) -> None:
    mock_client = AsyncMock()
    mock_client.closed = False
    canvas.clients.add(mock_client)

    class UnserializableObject:
        pass

    await canvas.broadcast_semantic_event("test", {"obj": UnserializableObject()})

    mock_client.send.assert_not_awaited()

@pytest.mark.asyncio
async def test_stop_closes_active_clients(canvas: OpenClawSemanticCanvasV2) -> None:
    mock_client = AsyncMock()
    canvas.clients.add(mock_client)

    with patch("magda_agent.visualization.openclaw_semantic_canvas_v2.serve", new_callable=AsyncMock):
        await canvas.stop()

    mock_client.close.assert_awaited_once()
    assert len(canvas.clients) == 0
