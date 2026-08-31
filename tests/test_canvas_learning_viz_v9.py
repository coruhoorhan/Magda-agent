import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.visualization.canvas_learning_viz_v9 import (
    CanvasLearningMetricsExporterV9,
    CanvasLearningBroadcasterV9,
)


@pytest.fixture
def exporter():
    """Fixture providing a fresh exporter instance."""
    return CanvasLearningMetricsExporterV9(max_events=5)


@pytest.fixture
def mock_websocket():
    """Fixture providing a mocked FastAPI/Starlette style websocket."""
    ws = MagicMock()
    ws.send_text = AsyncMock()
    return ws


def test_exporter_record_learning_shift(exporter):
    """Test that the exporter correctly records learning shifts."""
    event = exporter.record_learning_shift(
        user_id="user_123",
        user_model_parameters={"habit_weight": 0.5},
        metric_shifts={"habit_weight_shift": +0.1},
        metadata={"source": "feedback_loop"},
    )

    assert event["user_id"] == "user_123"
    assert event["user_model_parameters"] == {"habit_weight": 0.5}
    assert event["metric_shifts"] == {"habit_weight_shift": +0.1}
    assert event["metadata"] == {"source": "feedback_loop"}
    assert "event_id" in event
    assert "timestamp" in event

    assert len(exporter.events) == 1


def test_exporter_max_events(exporter):
    """Test that the exporter limits the number of stored events."""
    for i in range(10):
        exporter.record_learning_shift(
            user_id=f"user_{i}",
            user_model_parameters={"val": i},
            metric_shifts={"shift": 0.1},
        )

    assert len(exporter.events) == 5
    assert exporter.events[-1]["user_id"] == "user_9"
    assert exporter.events[0]["user_id"] == "user_5"


def test_get_canvas_payload(exporter):
    """Test that get_canvas_payload returns the correct JSON structure."""
    exporter.record_learning_shift(
        user_id="user_1",
        user_model_parameters={},
        metric_shifts={"shift": 0.2},
    )

    payload = exporter.get_canvas_payload()

    assert payload["type"] == "learning_metric_shift"
    assert "event_id" in payload
    assert payload["status"] == "active"
    assert "data" in payload
    assert "summary" in payload["data"]
    assert "recent_events" in payload["data"]

    assert payload["data"]["summary"]["total_shifts_recorded"] == 1
    assert len(payload["data"]["recent_events"]) == 1


@pytest.mark.asyncio
async def test_broadcaster_sends_payload(mock_websocket):
    """Test that the broadcaster constructs and sends the payload via websocket."""
    broadcaster = CanvasLearningBroadcasterV9(websocket=mock_websocket)

    event = await broadcaster.broadcast_learning_shift(
        user_id="user_abc",
        user_model_parameters={"x": 10},
        metric_shifts={"x_shift": 2},
    )

    assert mock_websocket.send_text.called

    # Extract the payload that was sent
    call_args = mock_websocket.send_text.call_args[0][0]
    sent_payload = json.loads(call_args)

    assert sent_payload["type"] == "learning_metric_shift"
    assert sent_payload["data"]["latest_event"]["user_id"] == "user_abc"
    assert sent_payload["data"]["latest_event"]["metric_shifts"]["x_shift"] == 2


@pytest.mark.asyncio
async def test_broadcaster_no_websocket():
    """Test that the broadcaster silently skips sending if no websocket is provided."""
    broadcaster = CanvasLearningBroadcasterV9(websocket=None)

    # Should not raise any errors
    event = await broadcaster.broadcast_learning_shift(
        user_id="user_none",
        user_model_parameters={},
        metric_shifts={},
    )

    assert event["user_id"] == "user_none"
    assert len(broadcaster.exporter.events) == 1
