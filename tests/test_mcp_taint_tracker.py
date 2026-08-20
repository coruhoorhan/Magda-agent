"""
Tests for the MCPTaintTracker module.
"""
import pytest
from magda_agent.security.mcp_taint_tracker import MCPTaintTracker, TaintError, TaintedString


def test_taint_labeling() -> None:
    """Test that data is correctly labeled with a taint."""
    tracker = MCPTaintTracker()
    data = tracker.label_data("mcp_output", "tool_a")

    assert isinstance(data, TaintedString)
    assert data == "mcp_output"
    assert "tool_a" in data.taints


def test_taint_propagation_add() -> None:
    """Test that taint propagates through string concatenation."""
    tracker = MCPTaintTracker()
    tainted = tracker.label_data("hello", "tool_a")

    result = tainted + " world"
    assert isinstance(result, TaintedString)
    assert result == "hello world"
    assert "tool_a" in result.taints

    result2 = "say " + tainted
    assert isinstance(result2, TaintedString)
    assert result2 == "say hello"
    assert "tool_a" in result2.taints

    tainted2 = tracker.label_data(" world", "tool_b")
    result3 = tainted + tainted2
    assert isinstance(result3, TaintedString)
    assert result3 == "hello world"
    assert "tool_a" in result3.taints
    assert "tool_b" in result3.taints


def test_taint_propagation_join() -> None:
    """Test that taint propagates through string join."""
    tracker = MCPTaintTracker()
    tainted = tracker.label_data("hello", "tool_a")

    # We can only track join when called ON a TaintedString
    tainted_sep = tracker.label_data("-", "tool_sep")
    result = tainted_sep.join([tainted, "world"])
    assert isinstance(result, TaintedString)
    assert result == "hello-world"
    assert "tool_sep" in result.taints
    assert "tool_a" in result.taints


def test_taint_propagation_replace() -> None:
    """Test that taint propagates through string replacement."""
    tracker = MCPTaintTracker()
    tainted = tracker.label_data("hello world", "tool_a")

    result = tainted.replace("world", "there")
    assert isinstance(result, TaintedString)
    assert result == "hello there"
    assert "tool_a" in result.taints


def test_taint_propagation_format() -> None:
    """Test that taint propagates through string formatting."""
    tracker = MCPTaintTracker()
    tainted_format = tracker.label_data("hello {}", "tool_format")

    result = tainted_format.format("world")
    assert isinstance(result, TaintedString)
    assert result == "hello world"
    assert "tool_format" in result.taints

    tainted_arg = tracker.label_data("world", "tool_arg")
    result2 = "hello {}".format(tainted_arg)
    # The normal str.format doesn't know about TaintedString, it will call str()
    # So we can't easily track this without overriding str (which is not possible)
    # However, if we call format ON the TaintedString:
    result3 = TaintedString("hello {}").format(tainted_arg)
    assert isinstance(result3, TaintedString)
    assert result3 == "hello world"
    assert "tool_arg" in result3.taints


def test_blocking_execution() -> None:
    """Test that execution is blocked when tainted data reaches sensitive endpoints."""
    tracker = MCPTaintTracker()
    tracker.register_sensitive_endpoint("system.execute")

    tainted = tracker.label_data("rm -rf /", "tool_a")

    # Execution should be blocked
    with pytest.raises(TaintError, match="Tainted data from {'tool_a'} attempted to access sensitive endpoint 'system.execute'"):
        tracker.check_execution("system.execute", tainted)

    # Should not block non-sensitive endpoints
    tracker.check_execution("logger.log", tainted)

    # Should block if tainted data is nested in a list
    with pytest.raises(TaintError):
        tracker.check_execution("system.execute", ["echo", tainted])

    # Should block if tainted data is nested in a dict
    with pytest.raises(TaintError):
        tracker.check_execution("system.execute", {"cmd": tainted})

    # Should block if key is tainted
    with pytest.raises(TaintError):
        tracker.check_execution("system.execute", {tainted: "value"})

    # Should NOT block untainted data even on sensitive endpoints
    tracker.check_execution("system.execute", "ls -la")
