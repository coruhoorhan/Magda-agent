import json
import pytest
from unittest.mock import AsyncMock, patch

from magda_agent.emotions.engine import PADState
from magda_agent.telemetry.live_rl_v3 import LiveRLTelemetryV3

@pytest.fixture
def mock_websocket():
    ws = AsyncMock()
    ws.send_text = AsyncMock()
    return ws

@pytest.mark.asyncio
async def test_live_rl_telemetry_emit_pad_shift(mock_websocket):
    telemetry = LiveRLTelemetryV3(websocket=mock_websocket)

    old_pad = PADState(pleasure=0.1, arousal=0.2, dominance=0.3)
    new_pad = PADState(pleasure=0.5, arousal=0.6, dominance=0.7)
    emotion_label = "Excited/Happy"
    user_id = 42
    habit_weights = {"greet": 0.8}

    with patch("time.time", return_value=12345.678):
        await telemetry.emit_pad_shift(
            old_pad=old_pad,
            new_pad=new_pad,
            emotion_label=emotion_label,
            user_id=user_id,
            habit_weights=habit_weights
        )

    mock_websocket.send_text.assert_called_once()

    # Extract the JSON payload
    call_args = mock_websocket.send_text.call_args
    message_json = call_args[0][0]
    payload = json.loads(message_json)

    # Verify top-level structure from LiveEventEmitter
    assert payload["type"] == "pad_shift_change"
    assert payload["category"] == "pad_shift"
    assert payload["timestamp"] == 12345.678

    # Verify nested data structure
    data = payload["data"]
    assert data["user_id"] == 42

    assert data["old_pad"]["pleasure"] == pytest.approx(0.1)
    assert data["old_pad"]["arousal"] == pytest.approx(0.2)
    assert data["old_pad"]["dominance"] == pytest.approx(0.3)

    assert data["new_pad"]["pleasure"] == pytest.approx(0.5)
    assert data["new_pad"]["arousal"] == pytest.approx(0.6)
    assert data["new_pad"]["dominance"] == pytest.approx(0.7)

    assert data["delta"]["pleasure"] == pytest.approx(0.4)
    assert data["delta"]["arousal"] == pytest.approx(0.4)
    assert data["delta"]["dominance"] == pytest.approx(0.4)

    assert data["emotion_label"] == "Excited/Happy"
    assert data["habit_weights"] == {"greet": 0.8}

@pytest.mark.asyncio
async def test_live_rl_telemetry_no_websocket():
    telemetry = LiveRLTelemetryV3(websocket=None)

    old_pad = PADState()
    new_pad = PADState()

    # Should not raise exception
    await telemetry.emit_pad_shift(
        old_pad=old_pad,
        new_pad=new_pad,
        emotion_label="Neutral"
    )
