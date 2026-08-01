import asyncio
import time
import pytest
from unittest.mock import AsyncMock
from magda_agent.execution.mcp_concurrent import MCPConcurrentExecutor

async def mock_tool_success(delay: float = 0.1, result: str = "success") -> str:
    """A mock asynchronous tool that succeeds after a delay."""
    await asyncio.sleep(delay)
    return result

async def mock_tool_failure(delay: float = 0.1, error_msg: str = "error") -> str:
    """A mock asynchronous tool that fails after a delay."""
    await asyncio.sleep(delay)
    raise ValueError(error_msg)

def test_execute_tools_concurrently_success():
    """Test that multiple successful tools run concurrently."""
    executor = MCPConcurrentExecutor()

    tasks = [
        (mock_tool_success, {"delay": 0.2, "result": "res1"}),
        (mock_tool_success, {"delay": 0.2, "result": "res2"}),
        (mock_tool_success, {"delay": 0.2, "result": "res3"}),
    ]

    start_time = time.time()
    results = asyncio.run(executor.execute_tools_concurrently(tasks))
    end_time = time.time()

    duration = end_time - start_time

    assert results == ["res1", "res2", "res3"]
    # If sequential, it would take > 0.6 seconds. Concurrently, it should take ~0.2 seconds.
    assert duration < 0.4, f"Execution took too long ({duration}s), possibly not concurrent."

def test_execute_tools_concurrently_with_exceptions():
    """Test that exceptions do not crash the gather and are returned as results."""
    executor = MCPConcurrentExecutor()

    tasks = [
        (mock_tool_success, {"delay": 0.1, "result": "ok"}),
        (mock_tool_failure, {"delay": 0.1, "error_msg": "failed"}),
        (mock_tool_success, {"delay": 0.1, "result": "ok2"}),
    ]

    results = asyncio.run(executor.execute_tools_concurrently(tasks))

    assert len(results) == 3
    assert results[0] == "ok"
    assert isinstance(results[1], ValueError)
    assert str(results[1]) == "failed"
    assert results[2] == "ok2"
