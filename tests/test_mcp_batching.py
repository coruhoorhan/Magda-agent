import asyncio
import time
import pytest
from magda_agent.execution.mcp_batching import MCPBatchingExecutor

async def mock_tool_success(delay: float = 0.1, result: str = "success") -> str:
    """A mock asynchronous tool that succeeds after a delay."""
    await asyncio.sleep(delay)
    return result

async def mock_tool_failure(delay: float = 0.1, error_msg: str = "error") -> str:
    """A mock asynchronous tool that fails after a delay."""
    await asyncio.sleep(delay)
    raise ValueError(error_msg)

def test_execute_in_batches_success():
    """Test that multiple successful tools batched by server run concurrently and return correct results."""
    executor = MCPBatchingExecutor()

    tasks = [
        ("server_a", mock_tool_success, {"delay": 0.2, "result": "res_a1"}),
        ("server_b", mock_tool_success, {"delay": 0.2, "result": "res_b1"}),
        ("server_a", mock_tool_success, {"delay": 0.2, "result": "res_a2"}),
        ("server_c", mock_tool_success, {"delay": 0.2, "result": "res_c1"}),
    ]

    start_time = time.time()
    results = asyncio.run(executor.execute_in_batches(tasks))
    end_time = time.time()

    duration = end_time - start_time

    assert results == ["res_a1", "res_b1", "res_a2", "res_c1"]
    # If sequential, it would take ~0.8 seconds. Concurrently across all, it should take ~0.2 seconds.
    assert duration < 0.4, f"Execution took too long ({duration}s), possibly not concurrent."

def test_execute_in_batches_with_exceptions():
    """Test that exceptions do not crash the executor and are returned as results in correct order."""
    executor = MCPBatchingExecutor()

    tasks = [
        ("server_a", mock_tool_success, {"delay": 0.1, "result": "ok_a"}),
        ("server_b", mock_tool_failure, {"delay": 0.1, "error_msg": "failed_b"}),
        ("server_a", mock_tool_success, {"delay": 0.1, "result": "ok_a2"}),
        ("server_a", mock_tool_failure, {"delay": 0.1, "error_msg": "failed_a"}),
    ]

    results = asyncio.run(executor.execute_in_batches(tasks))

    assert len(results) == 4
    assert results[0] == "ok_a"
    assert isinstance(results[1], ValueError)
    assert str(results[1]) == "failed_b"
    assert results[2] == "ok_a2"
    assert isinstance(results[3], ValueError)
    assert str(results[3]) == "failed_a"

def test_execute_in_batches_empty():
    """Test executing an empty list of tasks."""
    executor = MCPBatchingExecutor()
    results = asyncio.run(executor.execute_in_batches([]))
    assert results == []

def test_execute_in_batches_single_server():
    """Test executing tasks that all target the same server."""
    executor = MCPBatchingExecutor()

    tasks = [
        ("server_a", mock_tool_success, {"delay": 0.1, "result": "a1"}),
        ("server_a", mock_tool_success, {"delay": 0.1, "result": "a2"}),
        ("server_a", mock_tool_success, {"delay": 0.1, "result": "a3"}),
    ]

    results = asyncio.run(executor.execute_in_batches(tasks))
    assert results == ["a1", "a2", "a3"]
