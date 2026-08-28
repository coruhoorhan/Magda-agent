import pytest
import json
from unittest.mock import patch, MagicMock
from magda_agent.integration.a2a_tracing_v6 import A2ATracerV6

def test_start_trace():
    tracer = A2ATracerV6()
    trace_id = tracer.start_trace("test_task")
    assert trace_id in tracer.traces
    assert tracer.traces[trace_id]["task_name"] == "test_task"
    assert tracer.traces[trace_id]["parent_trace_id"] is None
    assert tracer.traces[trace_id]["status"] == "in_progress"

def test_start_trace_with_parent():
    tracer = A2ATracerV6()
    trace_id = tracer.start_trace("child_task", "parent_123")
    assert tracer.traces[trace_id]["parent_trace_id"] == "parent_123"

def test_end_trace():
    tracer = A2ATracerV6()
    trace_id = tracer.start_trace("test_task")
    tracer.end_trace(trace_id, "success")
    assert tracer.traces[trace_id]["status"] == "success"
    assert tracer.traces[trace_id]["end_time"] is not None

def test_end_trace_not_found():
    tracer = A2ATracerV6()
    with pytest.raises(KeyError):
        tracer.end_trace("invalid_id")

def test_add_span():
    tracer = A2ATracerV6()
    trace_id = tracer.start_trace("test_task")
    span_id = tracer.add_span(trace_id, "db_query", 15.5)

    assert len(tracer.traces[trace_id]["spans"]) == 1
    span = tracer.traces[trace_id]["spans"][0]
    assert span["span_id"] == span_id
    assert span["span_name"] == "db_query"
    assert span["duration_ms"] == 15.5

def test_log_monitoring_event():
    tracer = A2ATracerV6()
    trace_id = tracer.start_trace("test_task")
    tracer.log_monitoring_event(trace_id, "network_latency", {"latency_ms": 120})

    assert len(tracer.traces[trace_id]["monitoring_events"]) == 1
    event = tracer.traces[trace_id]["monitoring_events"][0]
    assert event["event_type"] == "network_latency"
    assert event["details"]["latency_ms"] == 120

def test_attach_delegated_task():
    tracer = A2ATracerV6()
    trace_id = tracer.start_trace("test_task")
    tracer.attach_delegated_task(trace_id, "task_001")

    assert tracer.delegated_tasks["task_001"] == trace_id

def test_get_trace_for_task():
    tracer = A2ATracerV6()
    trace_id = tracer.start_trace("test_task")
    tracer.attach_delegated_task(trace_id, "task_001")

    trace = tracer.get_trace_for_task("task_001")
    assert trace is not None
    assert trace["task_name"] == "test_task"

def test_get_trace_for_task_not_found():
    tracer = A2ATracerV6()
    trace = tracer.get_trace_for_task("invalid_task")
    assert trace is None

def test_get_trace():
    tracer = A2ATracerV6()
    trace_id = tracer.start_trace("test_task")
    trace = tracer.get_trace(trace_id)
    assert trace["task_name"] == "test_task"

def test_export_traces():
    tracer = A2ATracerV6()
    trace_id = tracer.start_trace("test_task")
    exported = tracer.export_traces()
    data = json.loads(exported)
    assert trace_id in data
    assert data[trace_id]["task_name"] == "test_task"
