import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from magda_agent.architecture.parallel_subagents_v7 import ParallelSubagentManagerV7

class MockExecutor:
    """Mock agent executor that simulates async work."""
    def __init__(self, delay: float, result_val: dict):
        self.delay = delay
        self.result_val = result_val

    async def execute(self, context: list, **kwargs) -> dict:
        await asyncio.sleep(self.delay)
        return self.result_val

@pytest.mark.asyncio
async def test_parallel_execution_and_merge():
    """
    Test that ParallelSubagentManagerV7 can run tasks in parallel and merge the returned state.
    """
    # Create mock SubagentSpawner
    mock_spawner = MagicMock()

    # We will simulate the spawner's spawn_subagent method using a side_effect
    # that just delegates back to the executor's execute method to simulate the real behavior.
    async def mock_spawn_subagent(task_description, full_context, agent_executor, **kwargs):
        # The spawner normally calls agent_executor.execute or similar.
        if hasattr(agent_executor, "execute"):
            return await agent_executor.execute(full_context, **kwargs)
        else:
            return await agent_executor(full_context, **kwargs)

    mock_spawner.spawn_subagent = AsyncMock(side_effect=mock_spawn_subagent)

    manager = ParallelSubagentManagerV7(spawner=mock_spawner)
    tasks = ["Task A", "Task B"]
    base_context = [{"role": "system", "content": "base"}]

    delay = 0.1
    # We will return a different dict depending on the factory call index
    factory_calls = 0
    def factory():
        nonlocal factory_calls
        res = {"result_a": "value_a"} if factory_calls == 0 else {"result_b": "value_b"}
        factory_calls += 1
        return MockExecutor(delay=delay, result_val=res)

    start_time = asyncio.get_event_loop().time()
    results = await manager.run_parallel_tasks(tasks, base_context, factory)
    end_time = asyncio.get_event_loop().time()

    elapsed = end_time - start_time

    assert len(results) == 2
    assert results[0] == {"result_a": "value_a"}
    assert results[1] == {"result_b": "value_b"}

    # If sequential, it would take ~0.2s. Parallel should be ~0.1s.
    assert elapsed < 0.2, f"Execution took too long ({elapsed}s), might be sequential."

    # Test state merging
    merged_state = manager.merge_state(results)
    assert merged_state == {"result_a": "value_a", "result_b": "value_b"}

@pytest.mark.asyncio
async def test_exception_handling():
    """
    Test that an exception in one subagent does not crash the entire process,
    and that state merging skips the exception safely.
    """
    mock_spawner = MagicMock()

    async def mock_spawn_subagent(task_description, full_context, agent_executor, **kwargs):
        if hasattr(agent_executor, "execute"):
            return await agent_executor.execute(full_context, **kwargs)
        return await agent_executor(full_context, **kwargs)

    mock_spawner.spawn_subagent = AsyncMock(side_effect=mock_spawn_subagent)
    manager = ParallelSubagentManagerV7(spawner=mock_spawner)

    factory_calls = 0
    def factory():
        nonlocal factory_calls
        class FaultyExecutor:
            async def execute(self, ctx, **kwargs):
                if "Task Error" in ctx[-1]["content"] if ctx else True:
                    raise ValueError("Failed on purpose")
                return {"success": True}

        class GoodExecutor:
            async def execute(self, ctx, **kwargs):
                return {"good_key": "good_value"}

        executor = FaultyExecutor() if factory_calls == 0 else GoodExecutor()
        factory_calls += 1
        return executor

    # Instead of appending to context inside tests, we just assume the context is sufficient
    # We will pass tasks that our mock will distinguish if we need to,
    # but here we distinguish by factory_calls order
    results = await manager.run_parallel_tasks(["Task Error", "Task Good"], [], factory)

    assert len(results) == 2
    assert isinstance(results[0], ValueError)
    assert results[1] == {"good_key": "good_value"}

    # Merge should ignore the exception
    merged = manager.merge_state(results)
    assert merged == {"good_key": "good_value"}

@pytest.mark.asyncio
async def test_merge_results_branch_names():
    """
    Test that when merge_results=True, the correct branch names are passed to the spawner.
    """
    mock_spawner = MagicMock()
    mock_spawner.spawn_subagent = AsyncMock(return_value={"test": "ok"})

    manager = ParallelSubagentManagerV7(spawner=mock_spawner)

    def factory():
        class MockExec:
            async def execute(self, *args, **kwargs):
                return {}
        return MockExec()

    await manager.run_parallel_tasks(["Task 1", "Task 2"], [], factory, merge_results=True)

    assert mock_spawner.spawn_subagent.call_count == 2

    call_args_1 = mock_spawner.spawn_subagent.call_args_list[0].kwargs
    call_args_2 = mock_spawner.spawn_subagent.call_args_list[1].kwargs

    assert call_args_1["branch_name"] == "subagent-task-0"
    assert call_args_2["branch_name"] == "subagent-task-1"
    assert call_args_1["merge_results"] is True
    assert call_args_2["merge_results"] is True
