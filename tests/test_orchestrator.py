import asyncio
import pytest
from magda_agent.agents.orchestrator import MultiAgentOrchestrator


@pytest.mark.asyncio
async def test_run_parallel_tasks_success() -> None:
    """
    Test that the orchestrator can execute multiple tasks in parallel
    and return their results successfully using a mocked spawner.
    """
    async def mock_spawner(task: dict) -> dict:
        # Simulate some asynchronous work
        await asyncio.sleep(0.05)
        return {"result": f"processed_{task['id']}"}

    orchestrator = MultiAgentOrchestrator(agent_spawner=mock_spawner)

    tasks = [
        {"id": "task_1", "data": "foo"},
        {"id": "task_2", "data": "bar"},
        {"id": "task_3", "data": "baz"},
    ]

    results = await orchestrator.run_parallel_tasks(tasks, timeout=2.0)

    assert len(results) == 3
    assert results[0]["result"] == "processed_task_1"
    assert results[1]["result"] == "processed_task_2"
    assert results[2]["result"] == "processed_task_3"


@pytest.mark.asyncio
async def test_run_parallel_tasks_timeout() -> None:
    """
    Test that the orchestrator enforces the timeout safety mechanism,
    raising an asyncio.TimeoutError if tasks take too long.
    """
    async def mock_slow_spawner(task: dict) -> dict:
        # This will exceed the 0.1s timeout
        await asyncio.sleep(0.5)
        return {"result": "too_slow"}

    orchestrator = MultiAgentOrchestrator(agent_spawner=mock_slow_spawner)

    tasks = [{"id": "slow_task_1"}]

    with pytest.raises(asyncio.TimeoutError):
        await orchestrator.run_parallel_tasks(tasks, timeout=0.1)


@pytest.mark.asyncio
async def test_run_parallel_tasks_empty() -> None:
    """
    Test that the orchestrator handles an empty task list gracefully.
    """
    orchestrator = MultiAgentOrchestrator()
    results = await orchestrator.run_parallel_tasks([], timeout=5.0)
    assert results == []


@pytest.mark.asyncio
async def test_run_parallel_tasks_no_spawner() -> None:
    """
    Test the fallback behavior when no agent_spawner is provided.
    """
    orchestrator = MultiAgentOrchestrator()

    tasks = [{"id": "fallback_task_1"}]

    results = await orchestrator.run_parallel_tasks(tasks, timeout=2.0)

    assert len(results) == 1
    assert results[0]["task_id"] == "fallback_task_1"
    assert results[0]["status"] == "completed"
