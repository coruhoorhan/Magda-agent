import pytest
import json
from magda_agent.integration.a2a_tracing_v8 import A2ATracerV8

def test_start_trace_sampled():
    tracer = A2ATracerV8(sample_rate=1.0)
    trace_id = tracer.start_trace("test_task")
    assert trace_id is not None
    assert trace_id in tracer.traces
    assert tracer.traces[trace_id]["task_name"] == "test_task"
    assert tracer.traces[trace_id]["parent_trace_id"] is None
    assert tracer.traces[trace_id]["status"] == "in_progress"

def test_start_trace_not_sampled():
    tracer = A2ATracerV8(sample_rate=0.0)
    trace_id = tracer.start_trace("test_task")
    assert trace_id is None

def test_start_trace_not_sampled_with_parent():
    tracer = A2ATracerV8(sample_rate=0.0)
    # Even if sample rate is 0, if there's a parent, we want to trace it.
    # We might have an explicit parent passed.
    # But current implementation says `not self.should_sample() and not parent_trace_id` -> returns None
    # So if there is a parent trace id, it should return a trace id.
    trace_id = tracer.start_trace("test_task", parent_trace_id="parent_123")
    assert trace_id is not None
    assert tracer.traces[trace_id]["parent_trace_id"] == "parent_123"

def test_end_trace():
    tracer = A2ATracerV8(sample_rate=1.0)
    trace_id = tracer.start_trace("test_task")
    assert trace_id is not None
    tracer.end_trace(trace_id, "success")
    assert tracer.traces[trace_id]["status"] == "success"
    assert tracer.traces[trace_id]["end_time"] is not None

def test_end_trace_not_found():
    tracer = A2ATracerV8()
    with pytest.raises(KeyError):
        tracer.end_trace("invalid_id")

def test_add_span():
    tracer = A2ATracerV8(sample_rate=1.0)
    trace_id = tracer.start_trace("test_task")
    assert trace_id is not None
    span_id = tracer.add_span(trace_id, "db_query", 15.5)

    assert len(tracer.traces[trace_id]["spans"]) == 1
    span = tracer.traces[trace_id]["spans"][0]
    assert span["span_id"] == span_id
    assert span["span_name"] == "db_query"
    assert span["duration_ms"] == 15.5

def test_inject_context():
    tracer = A2ATracerV8(sample_rate=1.0)
    trace_id = tracer.start_trace("test_task")
    assert trace_id is not None
    tracer.set_baggage(trace_id, "user_id", "456")

    carrier = {}
    tracer.inject_context(trace_id, carrier)

    assert carrier.get("x-a2a-trace-id") == trace_id
    baggage_json = carrier.get("x-a2a-baggage")
    assert baggage_json is not None
    baggage = json.loads(baggage_json)
    assert baggage.get("user_id") == "456"

def test_extract_context():
    tracer = A2ATracerV8(sample_rate=1.0)
    carrier = {"x-a2a-trace-id": "trace_999"}
    extracted_id = tracer.extract_context(carrier)
    assert extracted_id == "trace_999"

def test_set_and_get_baggage():
    tracer = A2ATracerV8(sample_rate=1.0)
    trace_id = tracer.start_trace("test_task")
    assert trace_id is not None
    tracer.set_baggage(trace_id, "tenant_id", "t-123")

    val = tracer.get_baggage(trace_id, "tenant_id")
    assert val == "t-123"

def test_attach_delegated_task():
    tracer = A2ATracerV8(sample_rate=1.0)
    trace_id = tracer.start_trace("test_task")
    assert trace_id is not None
    tracer.attach_delegated_task(trace_id, "task_001")

    assert tracer.delegated_tasks["task_001"] == trace_id

def test_get_trace_for_task():
    tracer = A2ATracerV8(sample_rate=1.0)
    trace_id = tracer.start_trace("test_task")
    assert trace_id is not None
    tracer.attach_delegated_task(trace_id, "task_001")

    trace = tracer.get_trace_for_task("task_001")
    assert trace is not None
    assert trace["task_name"] == "test_task"

def test_export_traces():
    tracer = A2ATracerV8(sample_rate=1.0)
    trace_id = tracer.start_trace("test_task")
    assert trace_id is not None
    exported = tracer.export_traces()
    data = json.loads(exported)
    assert trace_id in data
    assert data[trace_id]["task_name"] == "test_task"