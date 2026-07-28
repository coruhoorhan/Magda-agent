import asyncio
import pytest
from magda_agent.skills.mcp_concurrent_v2 import MCPConcurrentExecutorV2

@pytest.mark.asyncio
async def test_execute_tools_concurrently():
    """Test that multiple tools are executed concurrently and return expected results."""
    executor = MCPConcurrentExecutorV2()

    async def mock_tool(name: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return f"result_{name}"

    tools = [mock_tool, mock_tool, mock_tool]
    kwargs_list = [
        {"name": "A", "delay": 0.1},
        {"name": "B", "delay": 0.2},
        {"name": "C", "delay": 0.1}
    ]

    results = await executor.execute_tools(tools, kwargs_list)
    assert results == ["result_A", "result_B", "result_C"]

@pytest.mark.asyncio
async def test_execute_tools_mismatched_lengths():
    """Test that mismatched lists raise ValueError."""
    executor = MCPConcurrentExecutorV2()
    async def mock_tool(): pass

    with pytest.raises(ValueError, match="Lengths of tools and kwargs_list must match."):
        await executor.execute_tools([mock_tool], [])

@pytest.mark.asyncio
async def test_execute_tools_with_exceptions():
    """Test that exceptions are caught and returned when return_exceptions=True."""
    executor = MCPConcurrentExecutorV2()

    async def mock_tool_success(val: str) -> str:
        return val

    async def mock_tool_fail() -> None:
        raise RuntimeError("tool failed")

    tools = [mock_tool_success, mock_tool_fail, mock_tool_success]
    kwargs_list = [{"val": "1"}, {}, {"val": "3"}]

    results = await executor.execute_tools(tools, kwargs_list)

    assert results[0] == "1"
    assert isinstance(results[1], RuntimeError)
    assert str(results[1]) == "tool failed"
    assert results[2] == "3"

@pytest.mark.asyncio
async def test_execute_tools_with_semaphore():
    """Test that the concurrency limit is respected."""
    executor = MCPConcurrentExecutorV2(max_concurrency=2)
    active_tasks = 0
    max_active_tasks = 0

    async def mock_tool() -> int:
        nonlocal active_tasks, max_active_tasks
        active_tasks += 1
        max_active_tasks = max(max_active_tasks, active_tasks)
        await asyncio.sleep(0.1)
        active_tasks -= 1
        return 1

    tools = [mock_tool] * 5
    kwargs_list = [{}] * 5

    results = await executor.execute_tools(tools, kwargs_list)

    assert sum(results) == 5
    assert max_active_tasks <= 2
