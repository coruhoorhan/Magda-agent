import pytest
from unittest.mock import MagicMock, patch
from magda_agent.security.mcp_taint_v6 import TaintedValueV6, MCPKernelTaintTrackerV6
from magda_agent.security.mcp_kernel import SecurityError
from magda_agent.security.mcp_kernel_taint import mark_tainted, is_tainted

def test_tainted_value_unwrap():
    """Test that TaintedValueV6 unwraps the correct value."""
    tv = TaintedValueV6(value="test_value", tainted=True)
    assert tv.unwrap() == "test_value"
    assert tv.tainted is True

def test_tainted_value_set_tainted():
    """Test setting the tainted flag manually."""
    tv = TaintedValueV6(value="test_value", tainted=False)
    assert tv.tainted is False
    tv.set_tainted(True)
    assert tv.tainted is True

@patch('magda_agent.security.mcp_kernel.MCPKernel.execute')
def test_execute_plugin_with_untainted_inputs(mock_execute):
    """Test execution with completely safe/untainted inputs."""
    mock_execute.return_value = {"x": 42}
    tracker = MCPKernelTaintTrackerV6()

    code = "x = input_val * 2"
    inputs = {"input_val": 21}

    result = tracker.execute_plugin(code, inputs)

    assert result == {"x": 42}
    assert not is_tainted(result)
    mock_execute.assert_called_once_with(code, locals_dict={"input_val": 21})

@patch('magda_agent.security.mcp_kernel.MCPKernel.execute')
def test_execute_plugin_with_tainted_code(mock_execute):
    """Test that providing tainted code raises SecurityError."""
    tracker = MCPKernelTaintTrackerV6()

    code = mark_tainted("x = 1")
    inputs = {}

    with pytest.raises(SecurityError, match="Code is tainted and unsafe to execute."):
        tracker.execute_plugin(code, inputs)

    mock_execute.assert_not_called()

@patch('magda_agent.security.mcp_kernel.MCPKernel.execute')
def test_execute_plugin_propagates_taint(mock_execute):
    """Test that providing tainted inputs correctly taints the resulting variables."""
    mock_execute.return_value = {"x": 42}
    tracker = MCPKernelTaintTrackerV6()

    code = "x = input_val * 2"
    inputs = mark_tainted({"input_val": 21})

    result = tracker.execute_plugin(code, inputs)

    # Check that output gets tainted
    assert is_tainted(result)

@patch('magda_agent.security.mcp_kernel.MCPKernel.execute')
def test_execute_plugin_with_tainted_value_class(mock_execute):
    """Test execution with TaintedValueV6 wrapping an input."""
    mock_execute.return_value = {"x": 42}
    tracker = MCPKernelTaintTrackerV6()

    code = "x = input_val * 2"
    inputs = {"input_val": TaintedValueV6(value=21, tainted=True)}

    result = tracker.execute_plugin(code, inputs)

    assert is_tainted(result)
    mock_execute.assert_called_once_with(code, locals_dict={"input_val": 21})

@patch('magda_agent.security.mcp_kernel.MCPKernel.execute')
def test_execute_plugin_with_untainted_value_class(mock_execute):
    """Test execution with TaintedValueV6 wrapping an untainted input."""
    mock_execute.return_value = {"x": 42}
    tracker = MCPKernelTaintTrackerV6()

    code = "x = input_val * 2"
    inputs = {"input_val": TaintedValueV6(value=21, tainted=False)}

    result = tracker.execute_plugin(code, inputs)

    assert not is_tainted(result)
    mock_execute.assert_called_once_with(code, locals_dict={"input_val": 21})
