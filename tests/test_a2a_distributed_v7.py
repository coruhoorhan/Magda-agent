import pytest
from unittest.mock import AsyncMock, patch
from magda_agent.telemetry.a2a_distributed_v7 import A2ADistributedTelemetryV7

@pytest.fixture
def telemetry():
    return A2ADistributedTelemetryV7()

def test_track_event(telemetry):
    telemetry.track_event("sub_1", "task_start", {"task_id": 101})
    assert len(telemetry.events) == 1
    assert telemetry.events[0] == {
        "subagent_id": "sub_1",
        "event_name": "task_start",
        "payload": {"task_id": 101}
    }

@pytest.mark.asyncio
async def test_broadcast_events_empty(telemetry):
    with patch.object(telemetry, '_mock_broadcast', new_callable=AsyncMock) as mock_broadcast:
        await telemetry.broadcast_events()
        mock_broadcast.assert_not_called()

@pytest.mark.asyncio
async def test_broadcast_events(telemetry):
    telemetry.track_event("sub_1", "task_start", {"task_id": 101})
    telemetry.track_event("sub_2", "task_end", {"task_id": 102})

    assert len(telemetry.events) == 2

    with patch.object(telemetry, '_mock_broadcast', new_callable=AsyncMock) as mock_broadcast:
        await telemetry.broadcast_events()

        # Verify broadcast was called with correct payload
        mock_broadcast.assert_called_once()
        args, _ = mock_broadcast.call_args
        payload = args[0]

        assert payload["type"] == "telemetry_broadcast"
        assert len(payload["events"]) == 2
        assert payload["events"][0]["subagent_id"] == "sub_1"
        assert payload["events"][1]["subagent_id"] == "sub_2"

        # Verify events list is cleared
        assert len(telemetry.events) == 0

@pytest.mark.asyncio
async def test_broadcast_pad_shift_to_canvas(telemetry):
    pad_shift = {"P": 0.5, "A": -0.2, "D": 0.1}

    with patch.object(telemetry, '_mock_websocket_emit', new_callable=AsyncMock) as mock_emit:
        await telemetry.broadcast_pad_shift_to_canvas("sub_1", pad_shift)

        mock_emit.assert_called_once()
        args, _ = mock_emit.call_args
        json_payload = args[0]

        import json
        payload = json.loads(json_payload)

        assert payload["type"] == "canvas_pad_shift"
        assert payload["subagent_id"] == "sub_1"
        assert payload["pad_shift"] == pad_shift

@pytest.mark.asyncio
async def test_broadcast_rl_reward_to_canvas(telemetry):
    reward_signal = 0.85
    details = {"action": "clarify_intent", "reason": "user_positive"}

    with patch.object(telemetry, '_mock_websocket_emit', new_callable=AsyncMock) as mock_emit:
        await telemetry.broadcast_rl_reward_to_canvas("sub_1", reward_signal, details)

        mock_emit.assert_called_once()
        args, _ = mock_emit.call_args
        json_payload = args[0]

        import json
        payload = json.loads(json_payload)

        assert payload["type"] == "canvas_rl_reward"
        assert payload["subagent_id"] == "sub_1"
        assert payload["reward_signal"] == reward_signal
        assert payload["details"] == details

@pytest.mark.asyncio
async def test_broadcast_rl_reward_to_canvas_no_details(telemetry):
    reward_signal = 0.85

    with patch.object(telemetry, '_mock_websocket_emit', new_callable=AsyncMock) as mock_emit:
        await telemetry.broadcast_rl_reward_to_canvas("sub_1", reward_signal)

        mock_emit.assert_called_once()
        args, _ = mock_emit.call_args
        json_payload = args[0]

        import json
        payload = json.loads(json_payload)

        assert payload["type"] == "canvas_rl_reward"
        assert payload["subagent_id"] == "sub_1"
        assert payload["reward_signal"] == reward_signal
        assert payload["details"] == {}

@pytest.mark.asyncio
async def test_broadcast_tool_execution_trace(telemetry):
    tool_name = "search_web"
    arguments = {"query": "OpenClaw AI trends"}
    result = "Found 10 results"
    success = True

    with patch.object(telemetry, '_mock_websocket_emit', new_callable=AsyncMock) as mock_emit:
        await telemetry.broadcast_tool_execution_trace("sub_1", tool_name, arguments, result, success)

        mock_emit.assert_called_once()
        args, _ = mock_emit.call_args
        json_payload = args[0]

        import json
        payload = json.loads(json_payload)

        assert payload["type"] == "canvas_tool_trace"
        assert payload["subagent_id"] == "sub_1"
        assert payload["tool_name"] == tool_name
        assert payload["arguments"] == arguments
        assert payload["result"] == result
        assert payload["success"] == success
