import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from magda_agent.visualization.canvas_live_v1 import CanvasLiveVisualizer

@pytest.fixture
def visualizer():
    return CanvasLiveVisualizer(host="127.0.0.1", port=9999)

@pytest.mark.asyncio
async def test_start_and_stop(visualizer):
    with patch("magda_agent.visualization.canvas_live_v1.serve", new_callable=AsyncMock) as mock_serve:
        mock_server = AsyncMock()
        mock_serve.return_value = mock_server

        await visualizer.start()

        mock_serve.assert_called_once_with(visualizer._handler, "127.0.0.1", 9999)
        assert visualizer._server == mock_server

        await visualizer.stop()

        mock_server.close.assert_called_once()
        mock_server.wait_closed.assert_awaited_once()
        assert visualizer._server is None

@pytest.mark.asyncio
async def test_handler_registers_client(visualizer):
    mock_websocket = AsyncMock()
    mock_websocket.remote_address = ("127.0.0.1", 54321)

    async def mock_wait_closed():
        fut = asyncio.Future()
        mock_websocket.close_fut = fut
        await fut
        return

    mock_websocket.wait_closed = mock_wait_closed

    # Run the handler as a task so we can check the state before it closes
    handler_task = asyncio.create_task(visualizer._handler(mock_websocket))

    # Yield to event loop to let handler run
    await asyncio.sleep(0.01)

    assert mock_websocket in visualizer.clients

    # Close the websocket to complete the handler by setting the future result
    mock_websocket.close_fut.set_result(None)

    # Wait for the handler task to finish
    await handler_task

    assert mock_websocket not in visualizer.clients

@pytest.mark.asyncio
async def test_broadcast_state_sends_to_clients(visualizer):
    mock_websocket1 = AsyncMock()
    mock_websocket1.closed = False
    mock_websocket2 = AsyncMock()
    mock_websocket2.closed = False

    visualizer.clients.add(mock_websocket1)
    visualizer.clients.add(mock_websocket2)

    state = {"current_task": "testing"}
    pad_shifts = {"pleasure": 0.5, "arousal": 0.1, "dominance": -0.2}

    await visualizer.broadcast_state(state, pad_shifts)

    expected_payload = json.dumps({
        "type": "state_update",
        "data": {
            "state": state,
            "pad_shifts": pad_shifts
        }
    })

    mock_websocket1.send.assert_awaited_once_with(expected_payload)
    mock_websocket2.send.assert_awaited_once_with(expected_payload)

@pytest.mark.asyncio
async def test_broadcast_state_ignores_closed_clients(visualizer):
    mock_websocket1 = AsyncMock()
    mock_websocket1.closed = False
    mock_websocket2 = AsyncMock()
    mock_websocket2.closed = True

    visualizer.clients.add(mock_websocket1)
    visualizer.clients.add(mock_websocket2)

    state = {"current_task": "testing"}
    pad_shifts = {"pleasure": 0.5}

    await visualizer.broadcast_state(state, pad_shifts)

    mock_websocket1.send.assert_awaited_once()
    mock_websocket2.send.assert_not_awaited()

@pytest.mark.asyncio
async def test_broadcast_state_no_clients(visualizer):
    # Should return early without errors
    state = {"current_task": "testing"}
    pad_shifts = {"pleasure": 0.5}
    await visualizer.broadcast_state(state, pad_shifts)
