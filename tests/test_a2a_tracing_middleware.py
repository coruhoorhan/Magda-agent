import pytest
from unittest.mock import patch
from magda_agent.integration.a2a_tracing_middleware import A2ATracingMiddleware
from magda_agent.integration.a2a_tracing import A2ATracer, TRACE_HEADER

def test_inject_trace_id_new_trace():
    """Test injecting trace ID when no trace is currently active."""
    A2ATracer.set_trace_id(None)

    headers = {"Authorization": "Bearer token"}
    injected_headers = A2ATracingMiddleware.inject_trace_id(headers)

    assert TRACE_HEADER in injected_headers
    assert injected_headers["Authorization"] == "Bearer token"
    assert A2ATracer.get_current_trace_id() is not None
    assert injected_headers[TRACE_HEADER] == A2ATracer.get_current_trace_id()

def test_inject_trace_id_existing_trace():
    """Test injecting trace ID when a trace is already active."""
    test_trace_id = "test_trace_12345"
    A2ATracer.set_trace_id(test_trace_id)

    headers = {"Content-Type": "application/json"}
    injected_headers = A2ATracingMiddleware.inject_trace_id(headers)

    assert TRACE_HEADER in injected_headers
    assert injected_headers[TRACE_HEADER] == test_trace_id
    assert A2ATracer.get_current_trace_id() == test_trace_id

def test_extract_trace_id_success():
    """Test extracting trace ID from headers."""
    A2ATracer.set_trace_id(None)
    test_trace_id = "extracted_trace_67890"

    headers = {TRACE_HEADER: test_trace_id, "User-Agent": "test-agent"}
    extracted_id = A2ATracingMiddleware.extract_trace_id(headers)

    assert extracted_id == test_trace_id
    assert A2ATracer.get_current_trace_id() == test_trace_id

def test_extract_trace_id_not_found():
    """Test extracting trace ID when it's not present in headers."""
    A2ATracer.set_trace_id(None)

    headers = {"User-Agent": "test-agent"}
    extracted_id = A2ATracingMiddleware.extract_trace_id(headers)

    assert extracted_id is None
    assert A2ATracer.get_current_trace_id() is None
