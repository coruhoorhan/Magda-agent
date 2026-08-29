import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.skills.canvas_metrics_v8 import (
    CanvasMemoryMetricsExporterV8,
    CanvasMetricsBroadcasterV8,
)


def test_exporter_initialization() -> None:
    """Test default initialization of CanvasMemoryMetricsExporterV8."""
    exporter = CanvasMemoryMetricsExporterV8(max_events=5)
    assert exporter.events == []
    assert exporter.max_events == 5

    summary = exporter.get_metrics_summary()
    assert summary["total_events"] == 0
    assert summary["total_consolidated_memories"] == 0
    assert summary["global_average_importance"] == 0.0
    assert summary["global_average_compression_ratio"] == 0.0


def test_exporter_record_consolidation() -> None:
    """Test recording memory consolidation events and metric aggregation."""
    exporter = CanvasMemoryMetricsExporterV8(max_events=3)

    memories_1 = [{"id": "m1", "text": "hello"}, {"id": "m2", "text": "world"}]
    event_1 = exporter.record_consolidation(
        user_id="user_123",
        memories=memories_1,
        importance_scores=[0.8, 0.6],
        compression_ratio=0.5,
        metadata={"category": "user_preferences"},
    )

    assert event_1["user_id"] == "user_123"
    assert event_1["consolidated_count"] == 2
    assert event_1["average_importance"] == 0.7
    assert event_1["compression_ratio"] == 0.5
    assert len(exporter.events) == 1

    memories_2 = [{"id": "m3", "text": "foo"}]
    event_2 = exporter.record_consolidation(
        user_id="user_123",
        memories=memories_2,
        importance_scores=[0.9],
        compression_ratio=0.3,
    )

    summary = exporter.get_metrics_summary()
    assert summary["total_events"] == 2
    assert summary["total_consolidated_memories"] == 3
    assert summary["global_average_importance"] == 0.8
    assert summary["global_average_compression_ratio"] == 0.4


def test_exporter_max_events_trimming() -> None:
    """Test that events beyond max_events are trimmed."""
    exporter = CanvasMemoryMetricsExporterV8(max_events=2)

    for i in range(5):
        exporter.record_consolidation(
            user_id=f"user_{i}",
            memories=[{"id": f"m_{i}"}],
            importance_scores=[0.5],
            compression_ratio=1.0,
        )

    assert len(exporter.events) == 2
    assert exporter.events[0]["user_id"] == "user_3"
    assert exporter.events[1]["user_id"] == "user_4"


def test_exporter_payload_and_json() -> None:
    """Test Canvas UI payload generation and JSON export."""
    exporter = CanvasMemoryMetricsExporterV8()
    exporter.record_consolidation(
        user_id="u1",
        memories=[{"id": "m1"}],
        importance_scores=[0.7],
        compression_ratio=0.6,
    )

    payload = exporter.get_canvas_payload()
    assert payload["type"] == "memory_consolidation_metrics"
    assert payload["status"] == "active"
    assert "event_id" in payload
    assert "timestamp" in payload
    assert "data" in payload
    assert payload["data"]["summary"]["total_events"] == 1
    assert len(payload["data"]["recent_events"]) == 1

    json_str = exporter.export_json()
    parsed = json.loads(json_str)
    assert parsed["type"] == "memory_consolidation_metrics"


@pytest.mark.asyncio
async def test_broadcaster_send_text() -> None:
    """Test broadcasting via WebSocket with send_text interface."""
    mock_websocket = AsyncMock()
    mock_websocket.send_text = AsyncMock()

    broadcaster = CanvasMetricsBroadcasterV8(websocket=mock_websocket)

    memories = [{"id": "m1", "text": "consolidated text"}]
    with patch("time.time", return_value=1700000000.0):
        event = await broadcaster.broadcast_consolidation_metrics(
            user_id="user_abc",
            memories=memories,
            importance_scores=[0.85],
            compression_ratio=0.4,
        )

    assert event["user_id"] == "user_abc"
    assert mock_websocket.send_text.await_count == 1

    sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
    assert sent_data["type"] == "memory_consolidation_metrics"
    assert sent_data["data"]["latest_event"]["user_id"] == "user_abc"


@pytest.mark.asyncio
async def test_broadcaster_fallback_send() -> None:
    """Test broadcasting via WebSocket with fallback send interface."""
    mock_websocket = AsyncMock()
    del mock_websocket.send_text
    mock_websocket.send = AsyncMock()

    broadcaster = CanvasMetricsBroadcasterV8(websocket=mock_websocket)

    success = await broadcaster.broadcast_metrics()
    assert success is True
    assert mock_websocket.send.await_count == 1


@pytest.mark.asyncio
async def test_broadcaster_no_websocket() -> None:
    """Test broadcasting when no WebSocket connection is provided."""
    broadcaster = CanvasMetricsBroadcasterV8(websocket=None)
    broadcaster.logger = MagicMock()

    success = await broadcaster.broadcast_metrics()
    assert success is False
    broadcaster.logger.debug.assert_called_once()


@pytest.mark.asyncio
async def test_broadcaster_exception_handling() -> None:
    """Test handling of exceptions raised during WebSocket broadcasting."""
    mock_websocket = AsyncMock()
    mock_websocket.send_text = AsyncMock(side_effect=RuntimeError("Connection closed"))

    broadcaster = CanvasMetricsBroadcasterV8(websocket=mock_websocket)
    broadcaster.logger = MagicMock()

    success = await broadcaster.broadcast_metrics()
    assert success is False
    broadcaster.logger.error.assert_called_once()
