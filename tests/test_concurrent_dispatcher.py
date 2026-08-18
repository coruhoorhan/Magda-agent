import pytest
import asyncio
from magda_agent.execution.concurrent_dispatcher import ConcurrentDispatcher

@pytest.mark.asyncio
async def test_concurrent_execution_success():
    dispatcher = ConcurrentDispatcher()

    async def mock_executor(tool_name: str, arguments: dict):
        await asyncio.sleep(0.1)
        return f"{tool_name}_result"

    tools = [
        {"tool_name": "tool_1", "arguments": {}},
        {"tool_name": "tool_2", "arguments": {}},
        {"tool_name": "tool_3", "arguments": {}}
    ]

    results = await dispatcher.execute_concurrently(tools, mock_executor)

    assert len(results) == 3
    assert results == ["tool_1_result", "tool_2_result", "tool_3_result"]

@pytest.mark.asyncio
async def test_concurrent_execution_with_failure():
    dispatcher = ConcurrentDispatcher()

    async def mock_executor(tool_name: str, arguments: dict):
        await asyncio.sleep(0.1)
        if tool_name == "tool_2":
            raise ValueError("Intentional failure")
        return f"{tool_name}_result"

    tools = [
        {"tool_name": "tool_1", "arguments": {}},
        {"tool_name": "tool_2", "arguments": {}},
        {"tool_name": "tool_3", "arguments": {}}
    ]

    results = await dispatcher.execute_concurrently(tools, mock_executor)

    assert len(results) == 3
    assert results[0] == "tool_1_result"
    assert isinstance(results[1], ValueError)
    assert str(results[1]) == "Intentional failure"
    assert results[2] == "tool_3_result"
