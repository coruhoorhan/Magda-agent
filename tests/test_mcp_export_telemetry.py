import pytest
import logging
from magda_agent.telemetry.mcp_export_telemetry import MCPExportTelemetryTracker

def test_mcp_export_telemetry_tracker_initialization():
    """Test that the tracker initializes with an empty list of exports."""
    tracker = MCPExportTelemetryTracker()
    assert tracker.exports == []

def test_track_export():
    """Test that tracking an export appends the correct data to exports list."""
    tracker = MCPExportTelemetryTracker()
    metadata = {"timestamp": "2026-06-25T12:00:00Z", "version": "1.0"}
    tracker.track_export("my_tool", metadata)

    assert len(tracker.exports) == 1
    record = tracker.exports[0]
    assert record["tool_name"] == "my_tool"
    assert record["metadata"] == metadata

def test_track_export_logging(caplog):
    """Test that tracking an export logs the correct message."""
    tracker = MCPExportTelemetryTracker()

    with caplog.at_level(logging.INFO):
        tracker.track_export("test_tool", {"key": "value"})

    assert "Tracking MCP tool export: test_tool" in caplog.text

def test_track_multiple_exports():
    """Test tracking multiple tools sequentially."""
    tracker = MCPExportTelemetryTracker()
    tracker.track_export("tool_A", {"a": 1})
    tracker.track_export("tool_B", {"b": 2})

    assert len(tracker.exports) == 2
    assert tracker.exports[0]["tool_name"] == "tool_A"
    assert tracker.exports[1]["tool_name"] == "tool_B"
