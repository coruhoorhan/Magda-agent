import asyncio
import json
from unittest.mock import AsyncMock, patch
import pytest
from magda_agent.visualization.openclaw_canvas_v5 import OpenClawRLCanvasV5

@pytest.fixture
def canvas():
    return OpenClawRLCanvasV5(host="127.0.0.1", port=9999)

@pytest.mark.asyncio
async def test_start_and_stop(canvas):
    """
    Test that starting and stopping the canvas works correctly and cleans up resources.
    """
    with patch("magda_agent.visualization.openclaw_canvas_v5.serve", new_callable=AsyncMock) as mock_serve:
        mock_server = AsyncMock()
        mock_serve.return_value = mock_server

        await canvas.start()

        mock_serve.assert_called_once_with(canvas._handler, "127.0.0.1", 9999)
        assert canvas._server == mock_server

        await canvas.stop()

        mock_server.close.assert_called_once()
        mock_server.wait_closed.assert_awaited_once()
        assert canvas._server is None

@pytest.mark.asyncio
async def test_handler_registers_client(canvas):
    """
    Test that the handler registers a new client and removes it when closed.
    """
    mock_websocket = AsyncMock()
    mock_websocket.remote_address = ("127.0.0.1", 54321)

    async def mock_wait_closed():
        fut = asyncio.Future()
        mock_websocket.close_fut = fut
        await fut
        return

    mock_websocket.wait_closed = mock_wait_closed

    # Run the handler as a task so we can check the state before it closes
    handler_task = asyncio.create_task(canvas._handler(mock_websocket))

    # Yield to event loop to let handler run
    await asyncio.sleep(0.01)

    assert mock_websocket in canvas.clients

    # Close the websocket to complete the handler by setting the future result
    mock_websocket.close_fut.set_result(None)

    # Wait for the handler task to finish
    await handler_task

    assert mock_websocket not in canvas.clients

@pytest.mark.asyncio
async def test_broadcast_rl_event_sends_to_clients(canvas):
    """
    Test that broadcasting an RL event sends the payload to connected clients.
    """
    mock_websocket1 = AsyncMock()
    mock_websocket1.closed = False
    mock_websocket2 = AsyncMock()
    mock_websocket2.closed = False

    canvas.clients.add(mock_websocket1)
    canvas.clients.add(mock_websocket2)

    event_data = {"reward": 1.0, "state": "active"}

    await canvas.broadcast_rl_event("reward_update", event_data)

    expected_payload = json.dumps({
        "type": "reward_update",
        "data": event_data
    })

    mock_websocket1.send.assert_awaited_once_with(expected_payload)
    mock_websocket2.send.assert_awaited_once_with(expected_payload)

@pytest.mark.asyncio
async def test_broadcast_rl_event_ignores_closed_clients(canvas):
    """
    Test that broadcasting ignores closed client connections.
    """
    mock_websocket1 = AsyncMock()
    mock_websocket1.closed = False
    mock_websocket2 = AsyncMock()
    mock_websocket2.closed = True

    canvas.clients.add(mock_websocket1)
    canvas.clients.add(mock_websocket2)

    event_data = {"reward": -1.0}

    await canvas.broadcast_rl_event("penalty", event_data)

    mock_websocket1.send.assert_awaited_once()
    mock_websocket2.send.assert_not_awaited()

@pytest.mark.asyncio
async def test_broadcast_rl_event_no_clients(canvas):
    """
    Test that broadcasting with no clients returns early without error.
    """
    # Should return early without errors
    event_data = {"reward": 0.0}
    await canvas.broadcast_rl_event("neutral", event_data)
