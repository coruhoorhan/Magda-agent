import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from magda_agent.agents.worktree_isolation_v8 import SubagentWorktreeSpawnerV8
from magda_agent.llm_client import LLMClient

def test_spawn_parallel_isolation():
    """
    Test that spawn_parallel correctly executes subtasks in parallel
    while utilizing the GitWorktreeManager for isolation and creating
    isolated memories.
    """
    mock_llm = AsyncMock(spec=LLMClient)

    tasks = [
        {"description": "Task 1", "system_prompt": "Sys 1"},
        {"description": "Task 2", "system_prompt": "Sys 2"}
    ]

    base_context = "Global context"

    spawner = SubagentWorktreeSpawnerV8(llm=mock_llm, base_dir="/tmp/fake_dir")

    # We need to mock GitWorktreeManager's isolated_environment.
    # It's an @asynccontextmanager, we can mock it by creating a context manager that yields fake paths
    import contextlib

    @contextlib.asynccontextmanager
    async def mock_isolated_env(*args, **kwargs):
        # Using a generator state to yield different paths
        yield f"/tmp/fake/worktree_{hash(str(args))}"

    # We also need to mock SubAgent.execute since we don't want real LLM calls
    with patch("magda_agent.agents.worktree_isolation_v8.GitWorktreeManager.isolated_environment", new=mock_isolated_env), \
         patch("magda_agent.agents.worktree_isolation_v8.WorkingMemory") as MockWorkingMemory, \
         patch("magda_agent.agents.worktree_isolation_v8.SubAgent") as MockSubAgent, \
         patch("magda_agent.agents.worktree_isolation_v8.MemoryEntry") as MockMemoryEntry, \
         patch("magda_agent.agents.worktree_isolation_v8.PADState"):

        mock_memory_instance = AsyncMock()
        MockWorkingMemory.return_value = mock_memory_instance

        mock_subagent_instance = AsyncMock()
        # Ensure execute returns a string like it normally would
        mock_subagent_instance.execute.return_value = "Mocked Result"
        MockSubAgent.return_value = mock_subagent_instance

        results = asyncio.run(spawner.spawn_parallel(tasks, base_context))

        assert len(results) == 2
        for result in results:
            assert result["status"] == "success"
            assert result["result"] == "Mocked Result"
            assert "worktree" in result
            assert "task" in result

        assert MockWorkingMemory.call_count == 2
        assert MockSubAgent.call_count == 2
        assert mock_subagent_instance.execute.call_count == 2
        # Assert memory injection happened
        assert mock_subagent_instance.working_memory == mock_memory_instance
        # Assert memory entry added
        assert mock_memory_instance.add.call_count == 2

def test_spawn_parallel_error_handling():
    """
    Test that if a single task fails, it does not crash the whole spawner
    and returns an error status for that specific task.
    """
    mock_llm = AsyncMock(spec=LLMClient)

    tasks = [
        {"description": "Fail Task", "system_prompt": "Sys 1"}
    ]

    spawner = SubagentWorktreeSpawnerV8(llm=mock_llm)

    import contextlib
    @contextlib.asynccontextmanager
    async def mock_isolated_env(*args, **kwargs):
        yield "/fake/worktree_fail"

    with patch("magda_agent.agents.worktree_isolation_v8.GitWorktreeManager.isolated_environment", new=mock_isolated_env), \
         patch("magda_agent.agents.worktree_isolation_v8.SubAgent") as MockSubAgent:

        mock_subagent_instance = AsyncMock()
        mock_subagent_instance.execute.side_effect = Exception("Simulated Error")
        MockSubAgent.return_value = mock_subagent_instance

        results = asyncio.run(spawner.spawn_parallel(tasks, "context"))

        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "Simulated Error" in results[0]["result"]
