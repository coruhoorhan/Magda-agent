"""Tests for MCPKernel Taint Tracking Sandbox V4."""
import pytest

from magda_agent.safety.taint import PolicyViolationError
from magda_agent.safety.taint_sandbox_v4 import (
    MCPKernelV4,
    SandboxExecutionEnvironmentV4,
    TaintTrackerV4,
)
from magda_agent.safety.taint_tracking_v2 import TaintedString


def test_sandbox_execute_sync_taint_v4() -> None:
    """Test that SandboxExecutionEnvironmentV4 properly deep taints synchronous list outputs."""
    tracker = TaintTrackerV4()
    sandbox = SandboxExecutionEnvironmentV4(tracker)

    def mock_tool(input_str: str) -> list:
        return ["hello", input_str]

    # Create tainted input
    tainted_input = tracker.taint("world", "user_input")

    # Execute
    result = sandbox.execute(mock_tool, tainted_input)

    # Verify result is a list and elements are tainted
    assert isinstance(result, list)
    assert len(result) == 2

    assert tracker.is_tainted(result[0])
    assert tracker.is_tainted(result[1])

    origins_0 = tracker.get_origins(result[0])
    origins_1 = tracker.get_origins(result[1])

    assert "user_input" in origins_0
    assert "user_input" in origins_1


@pytest.mark.asyncio
async def test_sandbox_execute_async_taint_v4() -> None:
    """Test that SandboxExecutionEnvironmentV4 properly deep taints async dict outputs."""
    tracker = TaintTrackerV4()
    sandbox = SandboxExecutionEnvironmentV4(tracker)

    async def mock_async_tool(input_str: str) -> dict:
        return {"data": input_str, "status": "ok"}

    # Create tainted input
    tainted_input = tracker.taint("sensitive_data", "network_call")

    # Execute async
    result = await sandbox.execute_async(mock_async_tool, tainted_input)

    # Verify dict elements are tainted
    assert isinstance(result, dict)

    assert tracker.is_tainted(result)
    assert tracker.is_tainted(result["data"])
    assert tracker.is_tainted(result["status"])

    assert "network_call" in tracker.get_origins(result["data"])
    assert "network_call" in tracker.get_origins(result["status"])


def test_mcp_kernel_v4_sensitive_block() -> None:
    """Test that MCPKernelV4 blocks sensitive executions with tainted data."""
    kernel = MCPKernelV4()

    def sensitive_tool(data: str) -> str:
        return f"Processed {data}"

    # Tainted input
    tainted_dict = {"data": kernel.tracker.taint("bad_data", "malicious_actor")}

    # Should raise PolicyViolationError because tool is marked sensitive and input is tainted
    with pytest.raises(PolicyViolationError, match="Tainted input to sensitive tool call from origins"):
        kernel.execute_tool(sensitive_tool, tainted_dict, is_sensitive=True)

    # Should succeed if not sensitive
    result = kernel.execute_tool(sensitive_tool, tainted_dict, is_sensitive=False)
    assert isinstance(result, TaintedString)
    assert "malicious_actor" in kernel.tracker.get_origins(result)
