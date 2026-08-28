import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import Any, List, Dict

from magda_agent.architecture.mcp_subagent_v2 import MCPSubagentV2

class MockExecutor:
    """Mock executor for testing MCP subagent."""
    def __init__(self, expected_result: Any, should_fail: bool = False):
        self.expected_result = expected_result
        self.should_fail = should_fail

    async def execute(self, context: List[Dict[str, Any]], **kwargs) -> Any:
        if self.should_fail:
            raise RuntimeError("Mock executor failed as requested")
        return self.expected_result

@pytest.fixture
def mock_spawner():
    """Provides a mocked SubagentSpawner."""
    spawner = MagicMock()
    # We will simulate the spawner's spawn_subagent method using a side_effect
    # that delegates back to the executor's execute method.
    async def mock_spawn_subagent(task_description, full_context, agent_executor, **kwargs):
        if hasattr(agent_executor, "execute"):
            return await agent_executor.execute(full_context, **kwargs)
        else:
            return await agent_executor(full_context, **kwargs)

    spawner.spawn_subagent = AsyncMock(side_effect=mock_spawn_subagent)
    return spawner


@pytest.mark.asyncio
async def test_mcp_subagent_successful_execution(mock_spawner):
    """
    Test that MCPSubagentV2 successfully orchestrates the execution of an MCP tool.
    """
    mcp_subagent = MCPSubagentV2(spawner=mock_spawner)

    tool_name = "test_tool"
    tool_kwargs = {"param1": "value1"}
    context = [{"role": "system", "content": "test context"}]
    expected_result = {"status": "success", "data": "test_data"}

    mock_executor = MockExecutor(expected_result=expected_result)

    result = await mcp_subagent.execute_mcp_tool_isolated(
        mcp_tool_name=tool_name,
        tool_kwargs=tool_kwargs,
        context=context,
        agent_executor=mock_executor,
        agent_id="test_agent_123"
    )

    assert result == expected_result
    mock_spawner.spawn_subagent.assert_called_once()

    call_kwargs = mock_spawner.spawn_subagent.call_args.kwargs
    assert f"'{tool_name}'" in call_kwargs["task_description"]
    assert str(tool_kwargs) in call_kwargs["task_description"]
    assert call_kwargs["full_context"] == context
    assert call_kwargs["agent_executor"] == mock_executor
    assert call_kwargs["agent_id"] == "test_agent_123"
    assert call_kwargs["merge_results"] is False


@pytest.mark.asyncio
async def test_mcp_subagent_executor_failure(mock_spawner):
    """
    Test that MCPSubagentV2 handles an exception raised by the agent executor (via spawner).
    """
    mcp_subagent = MCPSubagentV2(spawner=mock_spawner)

    mock_executor = MockExecutor(expected_result=None, should_fail=True)

    result = await mcp_subagent.execute_mcp_tool_isolated(
        mcp_tool_name="failing_tool",
        tool_kwargs={},
        context=[],
        agent_executor=mock_executor
    )

    assert isinstance(result, Exception)
    assert str(result) == "Mock executor failed as requested"
    mock_spawner.spawn_subagent.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_subagent_spawner_failure():
    """
    Test that MCPSubagentV2 handles an exception raised by the spawner itself.
    """
    spawner = MagicMock()
    spawner.spawn_subagent = AsyncMock(side_effect=RuntimeError("Spawner failed to create worktree"))

    mcp_subagent = MCPSubagentV2(spawner=spawner)
    mock_executor = MockExecutor(expected_result="should_not_reach")

    result = await mcp_subagent.execute_mcp_tool_isolated(
        mcp_tool_name="tool",
        tool_kwargs={},
        context=[],
        agent_executor=mock_executor
    )

    assert isinstance(result, Exception)
    assert str(result) == "Spawner failed to create worktree"
    spawner.spawn_subagent.assert_called_once()
