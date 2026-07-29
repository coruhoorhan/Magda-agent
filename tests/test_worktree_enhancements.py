import asyncio
import os
import shutil
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from magda_agent.agents.worktree_enhancements import ParallelWorktreeSubagentSpawner
from magda_agent.llm_client import LLMClient
from magda_agent.isolation.git_worktree_v13 import GitWorktreeManagerV13
from magda_agent.memory.virtual_context import VirtualContextManager

@pytest.fixture
def mock_llm() -> MagicMock:
    """Fixture to create a mocked LLMClient."""
    llm = MagicMock(spec=LLMClient)
    llm.chat_completion = AsyncMock(side_effect=lambda messages, **kwargs: f"Completed: {messages[-1]['content'][:40]}")
    return llm

@pytest.fixture
def mock_worktree_manager() -> MagicMock:
    """Fixture to create a mocked GitWorktreeManagerV13."""
    manager = MagicMock(spec=GitWorktreeManagerV13)
    # Return unique paths to simulate distinct worktrees
    manager.create_worktree_async = AsyncMock(side_effect=lambda: f"/tmp/mock_worktrees/worktree_{id(MagicMock())}")
    manager.remove_worktree_async = AsyncMock()
    return manager

@pytest.mark.asyncio
async def test_parallel_execution_and_isolation(mock_llm, mock_worktree_manager) -> None:
    """
    Tests that the ParallelWorktreeSubagentSpawner properly executes parallel tasks,
    spawns isolated git worktrees, cleans them up, and sets up isolated contexts.
    """
    spawner = ParallelWorktreeSubagentSpawner(llm=mock_llm, worktree_manager=mock_worktree_manager)

    tasks = [
        {"description": "Analyze performance logs", "system_prompt": "You are a Performance Analyst."},
        {"description": "Validate database schema", "system_prompt": "You are a Database Engineer."}
    ]

    base_context = "System is running v2.1.0."

    results = await spawner.run_parallel_tasks(tasks, base_context)

    # 1. Verify results are returned correctly for both tasks
    assert len(results) == 2
    assert "Completed: " in results[0]
    assert "Completed: " in results[1]

    # 2. Verify LLM chat_completion was called exactly once per task (total 2 times)
    assert mock_llm.chat_completion.call_count == 2

    # 3. Verify git worktree creation and removal were called exactly once per task
    assert mock_worktree_manager.create_worktree_async.call_count == 2
    assert mock_worktree_manager.remove_worktree_async.call_count == 2

    # Verify each call to chat_completion received isolated worktree path and base_context
    chat_calls = mock_llm.chat_completion.call_args_list
    assert len(chat_calls) == 2

    for call in chat_calls:
        messages = call[0][0]
        # Inspect user prompt content
        user_prompt = messages[-1]["content"]
        assert "System is running v2.1.0" in user_prompt
        assert "Isolated Git Worktree Path:" in user_prompt
        assert "Assigned Task:" in user_prompt


@pytest.mark.asyncio
async def test_memory_isolation_concurrent_tasks(mock_llm, mock_worktree_manager) -> None:
    """
    Verifies that memories and VirtualContextManager states do not bleed across parallel tasks.
    Each task should run within its own private VirtualContextManager instance.
    """
    # Keep the databases on disk so we can inspect them after run completes
    spawner = ParallelWorktreeSubagentSpawner(
        llm=mock_llm,
        worktree_manager=mock_worktree_manager,
        cleanup_memory_dirs=False
    )

    # We will patch VirtualContextManager init to track created instances and verify isolation
    vc_instances = []
    original_init = VirtualContextManager.__init__

    def tracked_init(self, *args, **kwargs):
        vc_instances.append(self)
        original_init(self, *args, **kwargs)

    tasks = [
        {"description": "Task Alpha"},
        {"description": "Task Beta"}
    ]

    try:
        with patch.object(VirtualContextManager, "__init__", tracked_init):
            await spawner.run_parallel_tasks(tasks, "Common base context.")

        # There should be exactly two separate VirtualContextManager instances created (one per task)
        assert len(vc_instances) == 2
        vc1, vc2 = vc_instances[0], vc_instances[1]

        # The memory spaces must be completely disjoint instances
        assert vc1 is not vc2
        assert vc1.working_memory is not vc2.working_memory
        assert vc1.episodic_memory is not vc2.episodic_memory

        # Verify each instance has its own isolated records in its EpisodicMemory
        events1 = vc1.episodic_memory.get_all_events()
        events2 = vc2.episodic_memory.get_all_events()

        # Each should have only recorded its own assigned task, proving absolute memory separation
        assert any("Task Alpha" in ev["text"] for ev in events1)
        assert not any("Task Beta" in ev["text"] for ev in events1)

        assert any("Task Beta" in ev["text"] for ev in events2)
        assert not any("Task Alpha" in ev["text"] for ev in events2)

    finally:
        # Perform manual cleanup of the persistent directories used in the test
        for vc in vc_instances:
            if hasattr(vc, "episodic_memory") and hasattr(vc.episodic_memory, "client"):
                # Close/reset connections if possible, or just delete directories safely
                pass
            path = getattr(vc, "semantic_memory", None) and getattr(vc.semantic_memory, "persist_directory", None)
            if path and os.path.exists(path):
                try:
                    shutil.rmtree(path)
                except Exception:
                    pass


@pytest.mark.asyncio
async def test_isolated_task_failure_cleanup(mock_llm, mock_worktree_manager) -> None:
    """
    Verifies that even if one or all subtasks fail during execution,
    the spawner guarantees cleanup of all spawned git worktrees.
    """
    spawner = ParallelWorktreeSubagentSpawner(llm=mock_llm, worktree_manager=mock_worktree_manager)

    # Force LLM chat completion to raise an error
    mock_llm.chat_completion.side_effect = RuntimeError("LLM service offline")

    tasks = [{"description": "Failing Task"}]

    results = await spawner.run_parallel_tasks(tasks, "System context.")

    # Verify that execution was gracefully intercepted and error description returned
    assert len(results) == 1
    assert "Error: Task execution failed - LLM service offline" in results[0]

    # Verify that worktree creation and cleanup still occurred safely
    assert mock_worktree_manager.create_worktree_async.call_count == 1
    assert mock_worktree_manager.remove_worktree_async.call_count == 1
