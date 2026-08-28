import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import WebSocket, WebSocketDisconnect

from magda_agent.visualization.canvas_api_v3 import SemanticCanvasServerV3, get_canvas_v3_router

@pytest.mark.asyncio
async def test_semantic_canvas_stream_success() -> None:
    """Tests successful semantic canvas websocket stream connection and message parsing."""
    mock_canvas_server = MagicMock(spec=SemanticCanvasServerV3)
    mock_canvas_server.connect = AsyncMock()
    mock_canvas_server.disconnect = MagicMock()

    router = get_canvas_v3_router(mock_canvas_server, token="secret")

    # Extract the route function directly
    stream_func = router.routes[0].endpoint

    mock_ws = AsyncMock(spec=WebSocket)
    # Simulate receiving text once, then disconnecting
    mock_ws.receive_text.side_effect = [ "ping", WebSocketDisconnect() ]

    await stream_func(mock_ws, auth_token="secret")

    mock_canvas_server.connect.assert_awaited_once_with(mock_ws)
    mock_canvas_server.disconnect.assert_called_once_with(mock_ws)

@pytest.mark.asyncio
async def test_semantic_canvas_stream_unauthorized() -> None:
    """Tests unauthorized connection closing with code 1008."""
    mock_canvas_server = MagicMock(spec=SemanticCanvasServerV3)
    mock_canvas_server.connect = AsyncMock()

    router = get_canvas_v3_router(mock_canvas_server, token="secret")
    stream_func = router.routes[0].endpoint

    mock_ws = AsyncMock(spec=WebSocket)

    await stream_func(mock_ws, auth_token="wrong")

    mock_ws.close.assert_awaited_once_with(code=1008)
    mock_canvas_server.connect.assert_not_called()

@pytest.mark.asyncio
async def test_semantic_canvas_server_methods() -> None:
    """Tests SemanticCanvasServerV3 connect, disconnect, and broadcast."""
    server = SemanticCanvasServerV3()
    mock_ws = AsyncMock(spec=WebSocket)

    await server.connect(mock_ws)
    assert mock_ws in server.active_connections
    mock_ws.accept.assert_awaited_once()

    await server.broadcast_semantic_state('{"test":"data"}')
    mock_ws.send_text.assert_awaited_once_with('{"test":"data"}')

    server.disconnect(mock_ws)
    assert mock_ws not in server.active_connections

@pytest.mark.asyncio
async def test_semantic_canvas_server_broadcast_disconnect() -> None:
    """Tests SemanticCanvasServerV3 disconnects client on broadcast error."""
    server = SemanticCanvasServerV3()
    mock_ws = AsyncMock(spec=WebSocket)
    mock_ws.send_text.side_effect = Exception("Broadcast failed")

    await server.connect(mock_ws)
    assert mock_ws in server.active_connections

    await server.broadcast_semantic_state('{"test":"data"}')

    # Client should be disconnected due to exception
    assert mock_ws not in server.active_connections
