import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from magda_agent.memory.openclaw_canvas_v3 import OpenClawCanvasMemoryVizV3

@pytest.mark.asyncio
async def test_server_start_stop():
    """Test starting and stopping the canvas server."""
    viz = OpenClawCanvasMemoryVizV3(host="127.0.0.1", port=8767)

    with patch('magda_agent.memory.openclaw_canvas_v3.serve', new_callable=AsyncMock) as mock_serve:
        mock_server = AsyncMock()
        mock_serve.return_value = mock_server

        await viz.start()

        mock_serve.assert_called_once_with(viz._handler, "127.0.0.1", 8767)
        assert viz._server == mock_server

        await viz.stop()

        mock_server.close.assert_called_once()
        mock_server.wait_closed.assert_called_once()
        assert viz._server is None

@pytest.mark.asyncio
async def test_client_connection_handler():
    """Test handling of client connections."""
    viz = OpenClawCanvasMemoryVizV3(host="127.0.0.1", port=8767)
    mock_websocket = AsyncMock()
    mock_websocket.remote_address = ("127.0.0.1", 12345)

    # Run the handler as a task so we can cancel it or let it finish if we mock wait_closed appropriately
    # We'll just let it finish by having wait_closed return immediately

    await viz._handler(mock_websocket)

    # Because wait_closed returns immediately, the client should be added then removed
    assert mock_websocket not in viz.clients
    mock_websocket.wait_closed.assert_called_once()


@pytest.mark.asyncio
async def test_broadcast_memory_nodes():
    """Test broadcasting memory nodes to connected clients."""
    viz = OpenClawCanvasMemoryVizV3(host="127.0.0.1", port=8767)

    mock_ws1 = AsyncMock()
    mock_ws1.closed = False

    mock_ws2 = AsyncMock()
    mock_ws2.closed = False

    mock_ws3 = AsyncMock()
    mock_ws3.closed = True

    viz.clients.add(mock_ws1)
    viz.clients.add(mock_ws2)
    viz.clients.add(mock_ws3)

    nodes = [
        {"id": "node1", "type": "episodic", "label": "Test Memory"}
    ]

    await viz.broadcast_memory_nodes(nodes)

    expected_payload = json.dumps({
        "type": "memory_nodes_update",
        "data": {
            "nodes": nodes
        }
    })

    mock_ws1.send.assert_called_once_with(expected_payload)
    mock_ws2.send.assert_called_once_with(expected_payload)
    mock_ws3.send.assert_not_called()

@pytest.mark.asyncio
async def test_broadcast_no_clients():
    """Test broadcasting when no clients are connected."""
    viz = OpenClawCanvasMemoryVizV3(host="127.0.0.1", port=8767)
    nodes = [{"id": "node1"}]

    # Should just return without error
    await viz.broadcast_memory_nodes(nodes)
