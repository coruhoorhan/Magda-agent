import asyncio
import json
from unittest.mock import MagicMock, AsyncMock
import pytest
from magda_agent.operations.live_rl_v7 import OpenClawRLCanvasMetricsV7

@pytest.mark.asyncio
async def test_broadcast_telemetry_no_clients():
    server = OpenClawRLCanvasMetricsV7()
    # Test broadcast with no clients, should return without error
    await server.broadcast_telemetry("test_event", {"value": 1})

@pytest.mark.asyncio
async def test_broadcast_telemetry_with_clients():
    server = OpenClawRLCanvasMetricsV7()

    # Mock clients
    mock_client1 = AsyncMock()
    mock_client1.closed = False
    mock_client2 = AsyncMock()
    mock_client2.closed = False
    mock_client3 = AsyncMock()
    mock_client3.closed = True # Should not receive message

    server.clients.add(mock_client1)
    server.clients.add(mock_client2)
    server.clients.add(mock_client3)

    event_data = {"key": "value"}
    await server.broadcast_telemetry("test_event", event_data)

    expected_message = json.dumps({
        "type": "test_event",
        "data": event_data
    })

    mock_client1.send.assert_awaited_once_with(expected_message)
    mock_client2.send.assert_awaited_once_with(expected_message)
    mock_client3.send.assert_not_awaited()

@pytest.mark.asyncio
async def test_broadcast_pad_shift():
    server = OpenClawRLCanvasMetricsV7()
    mock_client = AsyncMock()
    mock_client.closed = False
    server.clients.add(mock_client)

    metadata = {"source": "user_input"}
    await server.broadcast_pad_shift(0.5, -0.2, 0.1, metadata)

    expected_message = json.dumps({
        "type": "pad_shift",
        "data": {
            "pleasure": 0.5,
            "arousal": -0.2,
            "dominance": 0.1,
            "metadata": metadata
        }
    })
    mock_client.send.assert_awaited_once_with(expected_message)

@pytest.mark.asyncio
async def test_broadcast_serialization_error(caplog):
    server = OpenClawRLCanvasMetricsV7()
    mock_client = AsyncMock()
    mock_client.closed = False
    server.clients.add(mock_client)

    class UnserializableObject:
        pass

    await server.broadcast_telemetry("test", {"obj": UnserializableObject()})

    assert "Failed to serialize broadcast payload" in caplog.text
    mock_client.send.assert_not_awaited()

@pytest.mark.asyncio
async def test_start_and_stop():
    # Only test the logic to avoid real bind issues in parallel tests
    server = OpenClawRLCanvasMetricsV7(port=0) # ephemeral port

    # We will use serve mock or just test the logic internally by patching serve
    with pytest.MonkeyPatch.context() as m:
        mock_serve = AsyncMock()
        mock_server_instance = AsyncMock()
        mock_serve.return_value = mock_server_instance
        m.setattr("magda_agent.operations.live_rl_v7.serve", mock_serve)

        await server.start()
        mock_serve.assert_awaited_once_with(server._handler, server.host, server.port)

        # Test stop closes clients and server
        mock_client = AsyncMock()
        server.clients.add(mock_client)

        await server.stop()

        mock_server_instance.close.assert_called_once()
        mock_server_instance.wait_closed.assert_awaited_once()
        mock_client.close.assert_awaited_once()
        assert len(server.clients) == 0
        assert server._server is None
