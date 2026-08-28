"""
Tests for the SubagentSpawnerV5 module.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.architecture.subagent_spawner_v5 import SubagentSpawnerV5
from magda_agent.architecture.agent_teams_v4 import AgentTeamManagerV4
from magda_agent.memory.compression_v8 import ClaudeContextCompressorV8
from magda_agent.memory.working import MemoryEntry

@pytest.mark.asyncio
async def test_compress_context_empty():
    """Test compressing an empty context."""
    spawner = SubagentSpawnerV5()
    compressed = await spawner.compress_context([])
    assert compressed == []

@pytest.mark.asyncio
async def test_compress_context_success():
    """Test successful context compression using ClaudeContextCompressorV8."""
    mock_compressor = MagicMock(spec=ClaudeContextCompressorV8)
    # compress_entries is async
    mock_compressor.compress_entries = AsyncMock(
        return_value=MemoryEntry(
            content="Summarized past context.",
            importance=0.5,
            emotional_state="neutral",
            tags=[],
            user_id="subagent_user"
        )
    )

    spawner = SubagentSpawnerV5(compressor=mock_compressor)
    context = [
        {"role": "system", "content": "You are an AI."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    compressed = await spawner.compress_context(context)

    # Should return a single system message with the compressed context
    assert len(compressed) == 1
    assert compressed[0]["role"] == "system"
    assert "Compressed Context:\nSummarized past context." in compressed[0]["content"]

    # Ensure the compressor was called with the correct converted entries
    mock_compressor.compress_entries.assert_called_once()
    args, kwargs = mock_compressor.compress_entries.call_args
    entries = args[0]
    assert len(entries) == 3
    assert entries[0].content == "system: You are an AI."

@pytest.mark.asyncio
async def test_compress_context_fallback():
    """Test fallback to naive compression if ClaudeContextCompressorV8 raises exception."""
    mock_compressor = MagicMock(spec=ClaudeContextCompressorV8)
    mock_compressor.compress_entries = AsyncMock(side_effect=Exception("Compression failed"))

    spawner = SubagentSpawnerV5(compressor=mock_compressor)
    context = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Msg 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Msg 2"},
        {"role": "assistant", "content": "Reply 2"},
        {"role": "user", "content": "Msg 3"},
        {"role": "assistant", "content": "Reply 3"},
    ]
    compressed = await spawner.compress_context(context)

    # Should fallback to keeping first and last 4
    assert len(compressed) == 5
    assert compressed[0] == context[0]
    assert compressed[1:] == context[-4:]

@pytest.mark.asyncio
async def test_spawn_subagent_callable():
    """Test spawning a subagent using a callable executor."""
    mock_team_manager = MagicMock(spec=AgentTeamManagerV4)
    mock_team_manager.spawn_agent = AsyncMock(return_value=("/mock/path", {"MOCK": "ENV"}))
    mock_team_manager.disband_agent = AsyncMock()

    mock_compressor = MagicMock(spec=ClaudeContextCompressorV8)
    mock_compressor.compress_entries = AsyncMock(
        return_value=MemoryEntry(
            content="Compressed",
            importance=0.5,
            emotional_state="neutral",
            tags=[],
            user_id="subagent_user"
        )
    )

    spawner = SubagentSpawnerV5(team_manager=mock_team_manager, compressor=mock_compressor)
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
    assert called_context[0]["role"] == "system"
    assert called_context[1]["role"] == "user"
    assert "Task: Do something" in called_context[1]["content"]

@pytest.mark.asyncio
async def test_spawn_subagent_with_execute_method():
    """Test spawning a subagent using an executor with an execute method."""
    mock_team_manager = MagicMock(spec=AgentTeamManagerV4)
    mock_team_manager.spawn_agent = AsyncMock(return_value=("/mock/path2", {"MOCK": "ENV2"}))
    mock_team_manager.disband_agent = AsyncMock()

    mock_compressor = MagicMock(spec=ClaudeContextCompressorV8)
    mock_compressor.compress_entries = AsyncMock(
        return_value=MemoryEntry(
            content="Compressed",
            importance=0.5,
            emotional_state="neutral",
            tags=[],
            user_id="subagent_user"
        )
    )

    spawner = SubagentSpawnerV5(team_manager=mock_team_manager, compressor=mock_compressor)
    context = [{"role": "system", "content": "System"}]

    executor = MagicMock()
    async def mock_execute(ctx, **kwargs):
        return "Task Executed"
    executor.execute = AsyncMock(side_effect=mock_execute)

    result = await spawner.spawn_subagent("Do something else", context, executor, agent_id="agent-2")

    assert result == "Task Executed"

    assert executor.execute.call_count == 1
    call_args, call_kwargs = executor.execute.call_args
    assert len(call_args) == 1
    assert call_kwargs["worktree_path"] == "/mock/path2"
    assert call_kwargs["isolated_env"] == {"MOCK": "ENV2"}

    mock_team_manager.spawn_agent.assert_called_once_with("agent-2", branch_name=None)
    mock_team_manager.disband_agent.assert_called_once_with("agent-2")

@pytest.mark.asyncio
@patch("magda_agent.architecture.subagent_spawner_v5.asyncio.create_subprocess_exec")
async def test_spawn_subagent_with_merge_results(mock_subprocess_exec):
    """Test spawning a subagent that merges results."""
    mock_team_manager = MagicMock(spec=AgentTeamManagerV4)
    mock_team_manager.spawn_agent = AsyncMock(return_value=("/mock/path", {"MOCK": "ENV"}))
    mock_team_manager.disband_agent = AsyncMock()

    mock_compressor = MagicMock(spec=ClaudeContextCompressorV8)
    mock_compressor.compress_entries = AsyncMock(
        return_value=MemoryEntry(
            content="Compressed",
            importance=0.5,
            emotional_state="neutral",
            tags=[],
            user_id="subagent_user"
        )
    )

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"stdout", b"")
    mock_process.returncode = 0
    mock_subprocess_exec.return_value = mock_process

    spawner = SubagentSpawnerV5(team_manager=mock_team_manager, compressor=mock_compressor)
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

    mock_compressor = MagicMock(spec=ClaudeContextCompressorV8)
    mock_compressor.compress_entries = AsyncMock(
        return_value=MemoryEntry(
            content="Compressed",
            importance=0.5,
            emotional_state="neutral",
            tags=[],
            user_id="subagent_user"
        )
    )

    spawner = SubagentSpawnerV5(team_manager=mock_team_manager, compressor=mock_compressor)
    context = []

    # Invalid executor
    invalid_executor = "not a callable"

    with pytest.raises(TypeError):
        await spawner.spawn_subagent("Fail task", context, invalid_executor, agent_id="agent-4")

    mock_team_manager.spawn_agent.assert_called_once_with("agent-4", branch_name=None)
    mock_team_manager.disband_agent.assert_called_once_with("agent-4")
