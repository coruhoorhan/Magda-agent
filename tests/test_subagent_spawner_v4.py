"""
Tests for the SubagentSpawnerV4 module.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.architecture.subagent_spawner_v4 import SubagentSpawnerV4
from magda_agent.architecture.agent_teams_v4 import AgentTeamManagerV4

def test_compress_context_short():
    """Test compressing a context that is already short enough."""
    spawner = SubagentSpawnerV4()
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
    spawner = SubagentSpawnerV4()
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
    spawner = SubagentSpawnerV4()
    compressed = spawner.compress_context([])
    assert compressed == []

@pytest.mark.asyncio
async def test_spawn_subagent_callable():
    """Test spawning a subagent using a callable executor."""
    mock_team_manager = MagicMock(spec=AgentTeamManagerV4)
    mock_team_manager.spawn_agent = AsyncMock(return_value=("/mock/path", {"MOCK": "ENV"}))
    mock_team_manager.disband_agent = AsyncMock()

    spawner = SubagentSpawnerV4(team_manager=mock_team_manager)
    context = [{"role": "system", "content": "System"}]

    async def mock_executor(ctx, **kwargs):
        mock_executor.called = True
        mock_executor.call_args = ctx
        mock_executor.kwargs = kwargs
        return "Task Complete"

    mock_executor.called = False

    result = await spawner.spawn_subagent("Do something", context, mock_executor, agent_id="agent-1")

    assert result == "Task Complete"
    assert mock_executor.called
    assert mock_executor.kwargs["worktree_path"] == "/mock/path"
    assert mock_executor.kwargs["isolated_env"] == {"MOCK": "ENV"}

    mock_team_manager.spawn_agent.assert_called_once_with("agent-1", branch_name=None)
    mock_team_manager.disband_agent.assert_called_once_with("agent-1")

    # Check the execution context passed
    called_context = mock_executor.call_args
    assert len(called_context) == 2
    assert called_context[0] == context[0]
    assert called_context[1]["role"] == "user"
    assert "Task: Do something" in called_context[1]["content"]

@pytest.mark.asyncio
async def test_spawn_subagent_with_execute_method():
    """Test spawning a subagent using an executor with an execute method."""
    mock_team_manager = MagicMock(spec=AgentTeamManagerV4)
    mock_team_manager.spawn_agent = AsyncMock(return_value=("/mock/path2", {"MOCK": "ENV2"}))
    mock_team_manager.disband_agent = AsyncMock()

    spawner = SubagentSpawnerV4(team_manager=mock_team_manager)
    context = [{"role": "system", "content": "System"}]

    executor = MagicMock()
    # To properly mock inspect.signature accepting kwargs, we need a spec
    async def mock_execute(ctx, **kwargs):
        return "Task Executed"
    executor.execute = AsyncMock(side_effect=mock_execute)

    result = await spawner.spawn_subagent("Do something else", context, executor, agent_id="agent-2")

    assert result == "Task Executed"

    # Verify execute was called with correct context and kwargs
    assert executor.execute.call_count == 1
    call_args, call_kwargs = executor.execute.call_args
    assert len(call_args) == 1
    assert call_args[0][0] == context[0]
    assert call_kwargs["worktree_path"] == "/mock/path2"
    assert call_kwargs["isolated_env"] == {"MOCK": "ENV2"}

    mock_team_manager.spawn_agent.assert_called_once_with("agent-2", branch_name=None)
    mock_team_manager.disband_agent.assert_called_once_with("agent-2")

@pytest.mark.asyncio
@patch("magda_agent.architecture.subagent_spawner_v4.asyncio.create_subprocess_exec")
async def test_spawn_subagent_with_merge_results(mock_subprocess_exec):
    """Test spawning a subagent that merges results."""
    mock_team_manager = MagicMock(spec=AgentTeamManagerV4)
    mock_team_manager.spawn_agent = AsyncMock(return_value=("/mock/path", {"MOCK": "ENV"}))
    mock_team_manager.disband_agent = AsyncMock()

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"stdout", b"")
    mock_process.returncode = 0
    mock_subprocess_exec.return_value = mock_process

    spawner = SubagentSpawnerV4(team_manager=mock_team_manager)
    context = []

    async def mock_executor(ctx, **kwargs):
        return "Task Merged"

    result = await spawner.spawn_subagent(
        "Do merge",
        context,
        mock_executor,
        agent_id="agent-3",
        branch_name="feature-branch",
        merge_results=True
    )

    assert result == "Task Merged"
    mock_subprocess_exec.assert_called_once_with(
        "git", "merge", "feature-branch", "--no-edit",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    mock_team_manager.spawn_agent.assert_called_once_with("agent-3", branch_name="feature-branch")
    mock_team_manager.disband_agent.assert_called_once_with("agent-3")


@pytest.mark.asyncio
async def test_spawn_subagent_invalid_executor():
    """Test spawning a subagent with an invalid executor raises TypeError."""
    mock_team_manager = MagicMock(spec=AgentTeamManagerV4)
    mock_team_manager.spawn_agent = AsyncMock(return_value=("/mock/path", {"MOCK": "ENV"}))
    mock_team_manager.disband_agent = AsyncMock()

    spawner = SubagentSpawnerV4(team_manager=mock_team_manager)
    context = []

    # Invalid executor (neither callable nor has execute method)
    invalid_executor = "not a callable"

    with pytest.raises(TypeError):
        await spawner.spawn_subagent("Fail task", context, invalid_executor, agent_id="agent-4")

    mock_team_manager.spawn_agent.assert_called_once_with("agent-4", branch_name=None)
    mock_team_manager.disband_agent.assert_called_once_with("agent-4")

@pytest.mark.asyncio
async def test_spawn_subagent_executor_exception():
    """Test spawning a subagent calls disband_agent even when executor raises exception."""
    mock_team_manager = MagicMock(spec=AgentTeamManagerV4)
    mock_team_manager.spawn_agent = AsyncMock(return_value=("/mock/path", {"MOCK": "ENV"}))
    mock_team_manager.disband_agent = AsyncMock()

    spawner = SubagentSpawnerV4(team_manager=mock_team_manager)
    context = []

    async def mock_executor(ctx, **kwargs):
        raise ValueError("Executor failed")

    with pytest.raises(ValueError, match="Executor failed"):
        await spawner.spawn_subagent("Fail task", context, mock_executor, agent_id="agent-5")

    mock_team_manager.spawn_agent.assert_called_once_with("agent-5", branch_name=None)
    mock_team_manager.disband_agent.assert_called_once_with("agent-5")
