"""Unit tests for MCP Tool Execution Isolation Wrapper."""
import pytest
from unittest.mock import MagicMock

from magda_agent.safety.mcp_isolation_wrapper import (
    IsolatedContextNamespace,
    MCPIsolationWrapper,
)
from magda_agent.safety.taint_tracking_v2 import (
    PolicyViolationError,
    TaintTrackerV2,
    get_origins,
    is_tainted,
)


def test_isolated_context_namespace_basics() -> None:
    """Test IsolatedContextNamespace variable management and history recording."""
    parent_ctx = {"initial": "value"}
    ns = IsolatedContextNamespace(name="test_ns", parent_context=parent_ctx)

    assert ns.name == "test_ns"
    assert ns.get_variable("initial") == "value"
    assert not ns.has_variable("foo")

    ns.set_variable("foo", 123)
    assert ns.get_variable("foo") == 123
    assert ns.has_variable("foo")

    ns.record_execution("mock_tool", {"a": 1}, "result", success=True)
    assert len(ns.history) == 1
    assert ns.history[0]["tool_name"] == "mock_tool"
    assert ns.history[0]["output"] == "result"

    ns.clear()
    assert not ns.has_variable("foo")
    assert not ns.has_variable("initial")


@pytest.mark.asyncio
async def test_isolated_context_namespace_context_managers() -> None:
    """Test sync and async context managers for IsolatedContextNamespace."""
    with IsolatedContextNamespace(name="sync_ns") as ns_sync:
        ns_sync.set_variable("k", "v")
        assert ns_sync.get_variable("k") == "v"

    async with IsolatedContextNamespace(name="async_ns") as ns_async:
        ns_async.set_variable("a", "b")
        assert ns_async.get_variable("a") == "b"


def test_mcp_isolation_wrapper_sync_execution() -> None:
    """Test synchronous execution of MCP tool in an isolated namespace."""
    wrapper = MCPIsolationWrapper()
    ns = wrapper.create_namespace(name="exec_ns")

    def sample_tool(text: str, multiplier: int = 2) -> str:
        return text * multiplier

    result = wrapper.execute_tool(sample_tool, "sample_tool", {"text": "abc", "multiplier": 3}, namespace=ns)

    assert result == "abcabcabc"
    assert ns.get_variable("last_result_sample_tool") == "abcabcabc"
    assert len(ns.history) == 1
    assert ns.history[0]["success"] is True


@pytest.mark.asyncio
async def test_mcp_isolation_wrapper_async_execution() -> None:
    """Test asynchronous execution of MCP tool in an isolated namespace."""
    wrapper = MCPIsolationWrapper()
    ns = wrapper.create_namespace(name="async_ns")

    async def async_sample_tool(query: str) -> str:
        return f"processed_{query}"

    result = await wrapper.execute_tool_async(async_sample_tool, "async_tool", {"query": "test"}, namespace=ns)

    assert result == "processed_test"
    assert ns.get_variable("last_result_async_tool") == "processed_test"
    assert len(ns.history) == 1


def test_mcp_isolation_wrapper_taint_propagation() -> None:
    """Test that input taint propagates to tool output."""
    wrapper = MCPIsolationWrapper()
    tainted_input = wrapper.tracker.taint("unsafe_data", "untrusted_web_search")

    def echo_tool(data: str) -> str:
        return f"echo: {data}"

    result = wrapper.execute_tool(echo_tool, "echo_tool", {"data": tainted_input})

    assert is_tainted(result)
    assert "untrusted_web_search" in get_origins(result)


def test_mcp_isolation_wrapper_blocks_sensitive_tool_with_tainted_input() -> None:
    """Test that tainted inputs to sensitive tools raise PolicyViolationError."""
    wrapper = MCPIsolationWrapper(sensitive_tools={"file_writer"})
    tainted_param = wrapper.tracker.taint("/etc/passwd", "user_prompt")

    def file_writer(filepath: str) -> str:
        return f"written {filepath}"

    with pytest.raises(PolicyViolationError) as exc_info:
        wrapper.execute_tool(file_writer, "file_writer", {"filepath": tainted_param})

    assert "Tainted input from origins" in str(exc_info.value)
    assert "user_prompt" in str(exc_info.value)

    # Test explicit is_sensitive=True flag
    with pytest.raises(PolicyViolationError):
        wrapper.execute_tool(file_writer, "other_tool", {"filepath": tainted_param}, is_sensitive=True)


def test_mcp_isolation_wrapper_blocks_blocked_origins() -> None:
    """Test that inputs from blocked origins raise PolicyViolationError."""
    wrapper = MCPIsolationWrapper(blocked_origins={"malicious_ip"})
    tainted_input = wrapper.tracker.taint("some_payload", "malicious_ip")

    def harmless_tool(payload: str) -> str:
        return payload

    with pytest.raises(PolicyViolationError) as exc_info:
        wrapper.execute_tool(harmless_tool, "harmless_tool", {"payload": tainted_input})

    assert "Input taint origin(s)" in str(exc_info.value)
    assert "malicious_ip" in str(exc_info.value)


def test_mcp_isolation_wrapper_context_isolation() -> None:
    """Test that tool execution isolates state and does not pollute outer variables."""
    wrapper = MCPIsolationWrapper()
    outer_inputs = {"data": "hello"}

    def mutating_tool(data: str) -> str:
        return data.upper()

    ns = wrapper.create_namespace("isolated_scope")
    ns.set_variable("data", "hello")

    wrapper.execute_tool(mutating_tool, "mutating_tool", outer_inputs, namespace=ns)

    # Inputs dictionary passed by caller remains unchanged
    assert outer_inputs == {"data": "hello"}
    # Namespace history captured the execution
    assert len(ns.history) == 1


def test_mcp_isolation_wrapper_handles_tool_failure() -> None:
    """Test that tool failures raise RuntimeError and record failure in history."""
    wrapper = MCPIsolationWrapper()
    ns = wrapper.create_namespace("fail_ns")

    def failing_tool(arg: str) -> None:
        raise ValueError("Network connection lost")

    with pytest.raises(RuntimeError) as exc_info:
        wrapper.execute_tool(failing_tool, "failing_tool", {"arg": "val"}, namespace=ns)

    assert "Isolated execution of tool 'failing_tool' failed" in str(exc_info.value)
    assert len(ns.history) == 1
    assert ns.history[0]["success"] is False
    assert "Network connection lost" in ns.history[0]["error"]
