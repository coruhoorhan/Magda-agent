"""Tests for MCPKernel Taint Tracking Isolation V4."""

import pytest

from magda_agent.security.mcp_kernel_v4 import MCPKernelV4
from magda_agent.security.mcp_kernel import SecurityError
from magda_agent.security.mcp_kernel_taint import mark_tainted, is_tainted

def test_mcp_kernel_v4_clean_execution() -> None:
    """Test execution of clean plugin with clean inputs."""
    kernel = MCPKernelV4()
    code = "out = inp + ' suffix'"
    inputs = {"inp": "clean"}

    result = kernel.execute_plugin(code, inputs)

    assert "out" in result
    assert result["out"] == "clean suffix"
    assert not is_tainted(result)

def test_mcp_kernel_v4_tainted_input_execution() -> None:
    """Test execution of clean plugin with tainted inputs propagates taint."""
    kernel = MCPKernelV4()
    code = "out = inp + ' suffix'"
    inputs = mark_tainted({"inp": "tainted"})

    result = kernel.execute_plugin(code, inputs)

    assert "out" in result
    assert result["out"] == "tainted suffix"
    assert is_tainted(result)

def test_mcp_kernel_v4_tainted_plugin_execution() -> None:
    """Test execution of a tainted plugin code blocks and raises error."""
    kernel = MCPKernelV4()
    code = mark_tainted("out = 'bad'")
    inputs = {"inp": "clean"}

    with pytest.raises(SecurityError, match="Code is tainted and unsafe to execute."):
        kernel.execute_plugin(code, inputs)
