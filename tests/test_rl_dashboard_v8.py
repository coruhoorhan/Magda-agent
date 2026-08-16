import json
import logging
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from magda_agent.visualization.rl_dashboard_v8 import RLDashboardV8

@pytest.fixture
def mock_websocket() -> MagicMock:
    """Fixture providing a mock websocket with async send methods."""
    mock_ws = MagicMock()
    mock_ws.send_text = AsyncMock()
    mock_ws.send = AsyncMock()
    return mock_ws

@pytest.mark.asyncio
async def test_broadcast_pad_shift_with_send_text(mock_websocket: MagicMock) -> None:
    """Test broadcasting PAD shift when websocket has send_text."""
    # Delete send so we know send_text is used
    del mock_websocket.send

    dashboard = RLDashboardV8(websocket=mock_websocket)
    pad_shift = {"pleasure": 0.1, "arousal": -0.05, "dominance": 0.2}

    with patch('magda_agent.visualization.rl_dashboard_v8.time.time', return_value=12345.0):
        await dashboard.broadcast_pad_shift(user_id=42, pad_shift=pad_shift)

    mock_websocket.send_text.assert_called_once()
    sent_message = mock_websocket.send_text.call_args[0][0]
    payload = json.loads(sent_message)

    assert payload["type"] == "rl_pad_shift"
    assert payload["timestamp"] == 12345.0
    assert payload["data"]["user_id"] == 42
    assert payload["data"]["pad_shift"] == pad_shift

@pytest.mark.asyncio
async def test_broadcast_pad_shift_with_send(mock_websocket: MagicMock) -> None:
    """Test broadcasting PAD shift when websocket only has send."""
    # Delete send_text so we know send is used
    del mock_websocket.send_text

    dashboard = RLDashboardV8(websocket=mock_websocket)
    pad_shift = {"pleasure": -0.1, "arousal": 0.5, "dominance": -0.2}

    with patch('magda_agent.visualization.rl_dashboard_v8.time.time', return_value=54321.0):
        await dashboard.broadcast_pad_shift(user_id=10, pad_shift=pad_shift)

    mock_websocket.send.assert_called_once()
    sent_message = mock_websocket.send.call_args[0][0]
    payload = json.loads(sent_message)

    assert payload["type"] == "rl_pad_shift"
    assert payload["timestamp"] == 54321.0
    assert payload["data"]["user_id"] == 10
    assert payload["data"]["pad_shift"] == pad_shift

@pytest.mark.asyncio
async def test_broadcast_without_websocket(caplog: pytest.LogCaptureFixture) -> None:
    """Test broadcasting when no websocket is connected."""
    caplog.set_level(logging.DEBUG)
    dashboard = RLDashboardV8(websocket=None)
    pad_shift = {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}

    # Run the broadcast
    await dashboard.broadcast_pad_shift(user_id=99, pad_shift=pad_shift)

    # Check that debug log was written
    assert "Skipping broadcast for rl_pad_shift - no websocket connected." in caplog.text

@pytest.mark.asyncio
async def test_broadcast_error_handling(mock_websocket: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    """Test error handling during broadcast."""
    mock_websocket.send_text.side_effect = Exception("Websocket connection closed")
    dashboard = RLDashboardV8(websocket=mock_websocket)

    pad_shift = {"pleasure": 0.1, "arousal": 0.1, "dominance": 0.1}
    await dashboard.broadcast_pad_shift(user_id=1, pad_shift=pad_shift)

    assert "Failed to broadcast rl_pad_shift event: Websocket connection closed" in caplog.text
