import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch

from magda_agent.learning.canvas_skills_metrics_v2 import CanvasSkillsMetricsV2

@pytest.fixture
def metrics_broadcaster():
    return CanvasSkillsMetricsV2()

@pytest.mark.asyncio
async def test_register_unregister_client(metrics_broadcaster):
    mock_websocket = AsyncMock()

    await metrics_broadcaster.register_client(mock_websocket)
    assert mock_websocket in metrics_broadcaster.connected_clients

    await metrics_broadcaster.unregister_client(mock_websocket)
    assert mock_websocket not in metrics_broadcaster.connected_clients

def test_format_payload(metrics_broadcaster):
    with patch('time.time', return_value=12345.0):
        payload_str = metrics_broadcaster.format_payload('test_event', {'score': 100})

        payload_dict = json.loads(payload_str)
        assert payload_dict['timestamp'] == 12345.0
        assert payload_dict['type'] == 'test_event'
        assert payload_dict['data']['score'] == 100

@pytest.mark.asyncio
async def test_broadcast_metric(metrics_broadcaster):
    mock_websocket_1 = AsyncMock()
    mock_websocket_2 = AsyncMock()

    await metrics_broadcaster.register_client(mock_websocket_1)
    await metrics_broadcaster.register_client(mock_websocket_2)

    with patch('time.time', return_value=12345.0):
        await metrics_broadcaster.broadcast_metric('test_metric', {'key': 'value'})

        expected_payload = json.dumps({
            "timestamp": 12345.0,
            "type": "test_metric",
            "data": {"key": "value"}
        })

        mock_websocket_1.send.assert_called_once_with(expected_payload)
        mock_websocket_2.send.assert_called_once_with(expected_payload)

@pytest.mark.asyncio
async def test_handle_connection(metrics_broadcaster):
    # Simulate a websocket that yields one message then closes
    class MockWebsocket:
        def __init__(self):
            self.send = AsyncMock()

        def __aiter__(self):
            self._messages = ["msg1"]
            return self

        async def __anext__(self):
            if not self._messages:
                raise StopAsyncIteration
            return self._messages.pop(0)

    mock_ws = MockWebsocket()

    # Run the connection handler
    await metrics_broadcaster.handle_connection(mock_ws)

    # Since the iterator finishes, the client should be unregistered
    assert mock_ws not in metrics_broadcaster.connected_clients
