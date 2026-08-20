import asyncio
import os
import shutil
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magda_agent.architecture.agent_teams_v4 import AgentWorktreeIsolationV4, AgentTeamManagerV4

@pytest.fixture
def isolation_manager() -> AgentWorktreeIsolationV4:
    return AgentWorktreeIsolationV4(base_dir="/tmp/test_agent_teams_v4")

@pytest.fixture
def team_manager(isolation_manager: AgentWorktreeIsolationV4) -> AgentTeamManagerV4:
    return AgentTeamManagerV4(isolation_manager=isolation_manager)


@pytest.mark.asyncio
async def test_create_worktree_success(isolation_manager: AgentWorktreeIsolationV4) -> None:
    agent_id = "agent_123"

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"output", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        path, env = await isolation_manager.create_worktree(agent_id)

        assert mock_exec.called
        assert agent_id in isolation_manager.active_worktrees
        assert path == isolation_manager.active_worktrees[agent_id]

        # Verify isolated env vars
        assert env["MAGDA_AGENT_ID"] == agent_id
        assert env["MAGDA_WORKTREE_PATH"] == path
        assert env["MAGDA_ISOLATED"] == "true"


@pytest.mark.asyncio
async def test_create_worktree_failure(isolation_manager: AgentWorktreeIsolationV4) -> None:
    agent_id = "agent_fail"

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"git error")
    mock_process.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="Git worktree creation failed"):
            await isolation_manager.create_worktree(agent_id)


@pytest.mark.asyncio
async def test_remove_worktree_success(isolation_manager: AgentWorktreeIsolationV4) -> None:
    agent_id = "agent_remove"
    dummy_path = "/tmp/test_agent_teams_v4/agent_remove_path"
    isolation_manager.active_worktrees[agent_id] = dummy_path

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec, \
         patch("os.path.exists", return_value=False):

        await isolation_manager.remove_worktree(agent_id)

        mock_exec.assert_called_with("git", "worktree", "remove", "--force", dummy_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        assert agent_id not in isolation_manager.active_worktrees


@pytest.mark.asyncio
async def test_aggressive_cleanup_fallback(isolation_manager: AgentWorktreeIsolationV4) -> None:
    agent_id = "agent_cleanup"
    dummy_path = "/tmp/test_agent_teams_v4/agent_cleanup_path"
    isolation_manager.active_worktrees[agent_id] = dummy_path

    # Mock git command failure
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"git error")
    mock_process.returncode = 128

    # Mock os.path.exists to say the directory still exists after git fails, then doesn't exist after aggressive cleanup
    with patch("asyncio.create_subprocess_exec", return_value=mock_process), \
         patch("os.path.exists", side_effect=[True, False]), \
         patch("shutil.rmtree") as mock_rmtree:

        await isolation_manager.remove_worktree(agent_id)

        mock_rmtree.assert_called_once_with(dummy_path, ignore_errors=True)
        assert agent_id not in isolation_manager.active_worktrees


@pytest.mark.asyncio
async def test_team_manager_spawn_and_disband(team_manager: AgentTeamManagerV4) -> None:
    agent_id = "agent_manager"

    # Mock the isolation manager's create and remove methods
    mock_path = "/tmp/mock_path"
    mock_env = {"MAGDA_AGENT_ID": agent_id}
    team_manager.isolation_manager.create_worktree = AsyncMock(return_value=(mock_path, mock_env))  # type: ignore
    team_manager.isolation_manager.remove_worktree = AsyncMock()  # type: ignore

    # Test spawn
    path, env = await team_manager.spawn_agent(agent_id)
    assert path == mock_path
    assert env == mock_env
    assert agent_id in team_manager.agents
    assert team_manager.get_agent_env(agent_id) == mock_env

    # Test duplicate spawn
    with pytest.raises(ValueError):
        await team_manager.spawn_agent(agent_id)

    # Test disband
    await team_manager.disband_agent(agent_id)
    assert agent_id not in team_manager.agents
    assert team_manager.get_agent_env(agent_id) is None
    team_manager.isolation_manager.remove_worktree.assert_called_once_with(agent_id)


@pytest.mark.asyncio
async def test_team_manager_disband_all(team_manager: AgentTeamManagerV4) -> None:
    # Setup multiple agents
    team_manager.agents = ["agent1", "agent2"]
    team_manager.agent_envs = {"agent1": {}, "agent2": {}}

    # Mock disband_agent
    with patch.object(team_manager, "disband_agent", new_callable=AsyncMock) as mock_disband:
        await team_manager.disband_all()

        assert mock_disband.call_count == 2
        mock_disband.assert_any_call("agent1")
        mock_disband.assert_any_call("agent2")
