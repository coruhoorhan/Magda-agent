import pytest
from unittest.mock import patch, MagicMock
from magda_agent.integration.a2a_tracing_v5 import A2ATracerV5
import json

@pytest.fixture
def tracer():
    return A2ATracerV5()

@patch('magda_agent.integration.a2a_tracing_v5.time.time')
@patch('magda_agent.integration.a2a_tracing_v5.uuid.uuid4')
def test_start_trace(mock_uuid, mock_time, tracer):
    mock_uuid.return_value = "fake-uuid-1"
    mock_time.return_value = 1000.0

    trace_id = tracer.start_trace("test_task")

    assert trace_id == "fake-uuid-1"
    assert "fake-uuid-1" in tracer.traces
    assert tracer.traces["fake-uuid-1"] == {
        "task_name": "test_task",
        "start_time": 1000.0,
        "end_time": None,
        "status": "in_progress",
        "spans": []
    }

@patch('magda_agent.integration.a2a_tracing_v5.time.time')
def test_end_trace(mock_time, tracer):
    mock_time.return_value = 1005.0
    tracer.traces["fake-trace-id"] = {
        "task_name": "test_task",
        "start_time": 1000.0,
        "end_time": None,
        "status": "in_progress",
        "spans": []
    }

    tracer.end_trace("fake-trace-id", status="success")

    assert tracer.traces["fake-trace-id"]["end_time"] == 1005.0
    assert tracer.traces["fake-trace-id"]["status"] == "success"

def test_end_trace_not_found(tracer):
    with pytest.raises(KeyError, match="Trace ID fake-trace-id not found."):
        tracer.end_trace("fake-trace-id")

@patch('magda_agent.integration.a2a_tracing_v5.time.time')
@patch('magda_agent.integration.a2a_tracing_v5.uuid.uuid4')
def test_add_span(mock_uuid, mock_time, tracer):
    mock_uuid.return_value = "fake-span-id"
    mock_time.return_value = 1002.0
    tracer.traces["fake-trace-id"] = {
        "task_name": "test_task",
        "start_time": 1000.0,
        "end_time": None,
        "status": "in_progress",
        "spans": []
    }

    span_id = tracer.add_span("fake-trace-id", "test_span", 150.0)

    assert span_id == "fake-span-id"
    assert len(tracer.traces["fake-trace-id"]["spans"]) == 1
    assert tracer.traces["fake-trace-id"]["spans"][0] == {
        "span_id": "fake-span-id",
        "span_name": "test_span",
        "duration_ms": 150.0,
        "timestamp": 1002.0
    }

def test_add_span_not_found(tracer):
    with pytest.raises(KeyError, match="Trace ID fake-trace-id not found."):
        tracer.add_span("fake-trace-id", "test_span", 150.0)

def test_get_trace(tracer):
    tracer.traces["fake-trace-id"] = {
        "task_name": "test_task"
    }
    trace = tracer.get_trace("fake-trace-id")
    assert trace == {"task_name": "test_task"}

def test_get_trace_not_found(tracer):
    with pytest.raises(KeyError, match="Trace ID fake-trace-id not found."):
        tracer.get_trace("fake-trace-id")

def test_export_traces(tracer):
    tracer.traces["fake-trace-id"] = {
        "task_name": "test_task"
    }
    exported = tracer.export_traces()
    assert json.loads(exported) == {"fake-trace-id": {"task_name": "test_task"}}
