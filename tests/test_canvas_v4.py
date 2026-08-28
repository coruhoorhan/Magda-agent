import asyncio
import json
from unittest.mock import AsyncMock, patch
import pytest
from magda_agent.visualization.canvas_v4 import OpenClawRLCanvasV6


@pytest.fixture
def canvas() -> OpenClawRLCanvasV6:
    """Fixture to provide a clean OpenClawRLCanvasV6 instance for each test."""
    return OpenClawRLCanvasV6(host="127.0.0.1", port=8765)


@pytest.mark.asyncio
async def test_start_and_stop(canvas: OpenClawRLCanvasV6) -> None:
    """Tests starting and stopping the WebSocket server."""
    with patch('magda_agent.visualization.canvas_v4.serve', new_callable=AsyncMock) as mock_serve:
        mock_server_instance = AsyncMock()
        mock_serve.return_value = mock_server_instance

        await canvas.start()
        mock_serve.assert_called_once_with(canvas._handler, "127.0.0.1", 8765)
        assert canvas._server is not None

        await canvas.stop()
        mock_server_instance.close.assert_called_once()
        mock_server_instance.wait_closed.assert_called_once()
        assert canvas._server is None


@pytest.mark.asyncio
async def test_handler(canvas: OpenClawRLCanvasV6) -> None:
    """Tests that the handler registers and unregisters clients correctly."""
    mock_websocket = AsyncMock()
    mock_websocket.remote_address = ("127.0.0.1", 12345)

    mock_websocket.wait_closed.return_value = None

    await canvas._handler(mock_websocket)

    assert mock_websocket not in canvas.clients


@pytest.mark.asyncio
async def test_broadcast_rl_event(canvas: OpenClawRLCanvasV6) -> None:
    """Tests broadcasting an RL event to all connected clients."""
    mock_client1 = AsyncMock()
    mock_client1.closed = False
    mock_client2 = AsyncMock()
    mock_client2.closed = False

    canvas.clients.add(mock_client1)
    canvas.clients.add(mock_client2)

    event_type = "reward_update"
    event_data = {"score": 10.5, "reason": "success"}

    await canvas.broadcast_rl_event(event_type, event_data)

    expected_payload = json.dumps({
        "type": event_type,
        "data": event_data
    })

    mock_client1.send.assert_called_once_with(expected_payload)
    mock_client2.send.assert_called_once_with(expected_payload)


@pytest.mark.asyncio
async def test_broadcast_rl_event_with_closed_client(canvas: OpenClawRLCanvasV6) -> None:
    """Tests that closed clients are ignored during broadcasting."""
    mock_client1 = AsyncMock()
    mock_client1.closed = False
    mock_client2 = AsyncMock()
    mock_client2.closed = True

    canvas.clients.add(mock_client1)
    canvas.clients.add(mock_client2)

    await canvas.broadcast_rl_event("test", {})

    mock_client1.send.assert_called_once()
    mock_client2.send.assert_not_called()


@pytest.mark.asyncio
async def test_broadcast_rl_event_no_clients(canvas: OpenClawRLCanvasV6) -> None:
    """Tests that broadcasting with no clients doesn't cause errors."""
    await canvas.broadcast_rl_event("test", {})


@pytest.mark.asyncio
async def test_broadcast_rl_event_serialization_error(canvas: OpenClawRLCanvasV6, caplog: pytest.LogCaptureFixture) -> None:
    """Tests that serialization errors are caught and logged."""
    mock_client = AsyncMock()
    mock_client.closed = False
    canvas.clients.add(mock_client)

    class Unserializable:
        pass

    await canvas.broadcast_rl_event("test", {"obj": Unserializable()})

    mock_client.send.assert_not_called()
    assert "Failed to serialize broadcast payload" in caplog.text


@pytest.mark.asyncio
async def test_stop_with_clients(canvas: OpenClawRLCanvasV6) -> None:
    """Tests that stopping the server correctly disconnects active clients."""
    mock_client = AsyncMock()
    canvas.clients.add(mock_client)

    await canvas.stop()

    mock_client.close.assert_called_once()
    assert len(canvas.clients) == 0
