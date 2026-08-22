"""
Tests for the SubagentSpawner module.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.architecture.subagent_spawning import SubagentSpawner

def test_compress_context_short():
    """Test compressing a context that is already short enough."""
    spawner = SubagentSpawner()
    context = [
        {"role": "system", "content": "You are an AI."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    compressed = spawner.compress_context(context)
    assert len(compressed) == 3
    assert compressed == context

def test_compress_context_long():
    """Test compressing a context that is longer than 5 messages."""
    spawner = SubagentSpawner()
    context = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Msg 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Msg 2"},
        {"role": "assistant", "content": "Reply 2"},
        {"role": "user", "content": "Msg 3"},
        {"role": "assistant", "content": "Reply 3"},
    ]
    compressed = spawner.compress_context(context)
    # Should keep first and last 4
    assert len(compressed) == 5
    assert compressed[0] == context[0]
    assert compressed[1:] == context[-4:]

def test_compress_context_empty():
    """Test compressing an empty context."""
    spawner = SubagentSpawner()
    compressed = spawner.compress_context([])
    assert compressed == []

@pytest.mark.asyncio
@patch("magda_agent.architecture.subagent_spawning.AgentWorktreeIsolationV4")
async def test_spawn_subagent_callable(MockIsolationV4):
    """Test spawning a subagent using a callable executor."""
    mock_isolation = MockIsolationV4.return_value
    mock_isolation.create_worktree = AsyncMock(return_value=("/mock/path", {"MOCK": "ENV"}))
    mock_isolation.remove_worktree = AsyncMock()

    spawner = SubagentSpawner()
    context = [{"role": "system", "content": "System"}]

    async def mock_executor(ctx, **kwargs):
        mock_executor.called = True
        mock_executor.call_args = ctx
        mock_executor.kwargs = kwargs
        return "Task Complete"

    mock_executor.called = False

    result = await spawner.spawn_subagent("Do something", context, mock_executor)

    assert result == "Task Complete"
    assert mock_executor.called
    assert mock_executor.kwargs["worktree_path"] == "/mock/path"
    assert mock_executor.kwargs["isolated_env"] == {"MOCK": "ENV"}

    mock_isolation.create_worktree.assert_called_once()
    mock_isolation.remove_worktree.assert_called_once()

    # Check the execution context passed
    called_context = mock_executor.call_args
    assert len(called_context) == 2
    assert called_context[0] == context[0]
    assert called_context[1]["role"] == "user"
    assert "Task: Do something" in called_context[1]["content"]

@pytest.mark.asyncio
@patch("magda_agent.architecture.subagent_spawning.AgentWorktreeIsolationV4")
async def test_spawn_subagent_with_execute_method(MockIsolationV4):
    """Test spawning a subagent using an executor with an execute method."""
    mock_isolation = MockIsolationV4.return_value
    mock_isolation.create_worktree = AsyncMock(return_value=("/mock/path2", {"MOCK": "ENV2"}))
    mock_isolation.remove_worktree = AsyncMock()

    spawner = SubagentSpawner()
    context = [{"role": "system", "content": "System"}]

    executor = MagicMock()
    executor.execute = AsyncMock(return_value="Task Executed")
    # To properly mock inspect.signature accepting kwargs, we need a spec
    async def mock_execute(ctx, **kwargs):
        return "Task Executed"
    executor.execute = AsyncMock(side_effect=mock_execute)

    result = await spawner.spawn_subagent("Do something else", context, executor)

    assert result == "Task Executed"

    # Verify execute was called with correct context and kwargs
    assert executor.execute.call_count == 1
    call_args, call_kwargs = executor.execute.call_args
    assert len(call_args) == 1
    assert call_args[0][0] == context[0]
    assert call_kwargs["worktree_path"] == "/mock/path2"
    assert call_kwargs["isolated_env"] == {"MOCK": "ENV2"}

    mock_isolation.create_worktree.assert_called_once()
    mock_isolation.remove_worktree.assert_called_once()

@pytest.mark.asyncio
@patch("magda_agent.architecture.subagent_spawning.AgentWorktreeIsolationV4")
@patch("magda_agent.architecture.subagent_spawning.asyncio.create_subprocess_exec")
async def test_spawn_subagent_with_merge_results(mock_subprocess_exec, MockIsolationV4):
    """Test spawning a subagent that merges results."""
    mock_isolation = MockIsolationV4.return_value
    mock_isolation.create_worktree = AsyncMock(return_value=("/mock/path", {"MOCK": "ENV"}))
    mock_isolation.remove_worktree = AsyncMock()

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"stdout", b"")
    mock_process.returncode = 0
    mock_subprocess_exec.return_value = mock_process

    spawner = SubagentSpawner()
    context = []

    async def mock_executor(ctx, **kwargs):
        return "Task Merged"

    result = await spawner.spawn_subagent(
        "Do merge",
        context,
        mock_executor,
        branch_name="feature-branch",
        merge_results=True
    )

    assert result == "Task Merged"
    mock_subprocess_exec.assert_called_once_with(
        "git", "merge", "feature-branch", "--no-edit",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    mock_isolation.create_worktree.assert_called_once()
    mock_isolation.remove_worktree.assert_called_once()


@pytest.mark.asyncio
@patch("magda_agent.architecture.subagent_spawning.AgentWorktreeIsolationV4")
async def test_spawn_subagent_invalid_executor(MockIsolationV4):
    """Test spawning a subagent with an invalid executor raises TypeError."""
    mock_isolation = MockIsolationV4.return_value
    mock_isolation.create_worktree = AsyncMock(return_value=("/mock/path", {"MOCK": "ENV"}))
    mock_isolation.remove_worktree = AsyncMock()
    spawner = SubagentSpawner()
    context = []

    # Invalid executor (neither callable nor has execute method)
    invalid_executor = "not a callable"

    with pytest.raises(TypeError):
        await spawner.spawn_subagent("Fail task", context, invalid_executor)
