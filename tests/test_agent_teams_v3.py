import asyncio
import os
import shutil
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from magda_agent.architecture.agent_teams_v3 import AgentWorktreeIsolationV3, AgentTeamManagerV3, GitWorktreeError


@pytest.fixture
def base_dir(tmp_path):
    path = str(tmp_path / "test_worktrees")
    os.makedirs(path, exist_ok=True)
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.mark.asyncio
async def test_agent_worktree_isolation_create(base_dir):
    """
    Tests that a worktree is created correctly.
    """
    isolation = AgentWorktreeIsolationV3(base_dir=base_dir)

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        path = await isolation.create_worktree("agent123", branch_name="feature-x")

        assert "agent123" in isolation.active_worktrees
        assert isolation.active_worktrees["agent123"] == path

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "git" in args
        assert "worktree" in args
        assert "add" in args
        assert "-b" in args
        assert "feature-x" in args


@pytest.mark.asyncio
async def test_agent_worktree_isolation_create_failure(base_dir):
    """
    Tests worktree creation failure.
    """
    isolation = AgentWorktreeIsolationV3(base_dir=base_dir)

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"fatal: error")
    mock_process.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(GitWorktreeError, match="Git worktree creation failed"):
            await isolation.create_worktree("agent123")


@pytest.mark.asyncio
async def test_agent_worktree_isolation_remove(base_dir):
    """
    Tests that a worktree is removed cleanly.
    """
    isolation = AgentWorktreeIsolationV3(base_dir=base_dir)

    # Manually setup active worktree state
    test_path = os.path.join(base_dir, "test_agent_path")
    os.makedirs(test_path, exist_ok=True)
    isolation.active_worktrees["agent123"] = test_path

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        await isolation.remove_worktree("agent123")

        assert "agent123" not in isolation.active_worktrees
        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "git" in args
        assert "worktree" in args
        assert "remove" in args
        assert "--force" in args
        assert test_path in args


@pytest.mark.asyncio
async def test_agent_team_manager_spawn_and_disband(base_dir):
    """
    Tests AgentTeamManagerV3 spawning and disbanding agents.
    """
    manager = AgentTeamManagerV3(isolation_manager=AgentWorktreeIsolationV3(base_dir=base_dir))

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        # Test Spawn
        path1 = await manager.spawn_agent("worker_1")
        assert "worker_1" in manager.agents
        assert path1 is not None

        path2 = await manager.spawn_agent("worker_2", branch_name="test-branch")
        assert "worker_2" in manager.agents
        assert path2 is not None

        # Test duplicate spawn
        with pytest.raises(ValueError, match="already exists"):
            await manager.spawn_agent("worker_1")

        # Test Disband single
        await manager.disband_agent("worker_1")
        assert "worker_1" not in manager.agents

        # Test Disband all
        await manager.disband_all()
        assert len(manager.agents) == 0
        assert "worker_2" not in manager.agents


@pytest.mark.asyncio
async def test_agent_team_manager_getters(base_dir):
    """
    Tests AgentTeamManagerV3 getters get_active_agents and get_worktree_path.
    """
    manager = AgentTeamManagerV3(isolation_manager=AgentWorktreeIsolationV3(base_dir=base_dir))

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        path1 = await manager.spawn_agent("agentA")
        path2 = await manager.spawn_agent("agentB")

        active = manager.get_active_agents()
        assert "agentA" in active
        assert "agentB" in active
        assert len(active) == 2

        assert manager.get_worktree_path("agentA") == path1
        assert manager.get_worktree_path("agentB") == path2
        assert manager.get_worktree_path("nonexistent") is None

@pytest.mark.asyncio
async def test_agent_evaluator_team():
    """
    Test the AgentEvaluatorTeam class.
    """
    from magda_agent.architecture.agent_teams_v3 import AgentEvaluatorTeam
    team = AgentEvaluatorTeam(evaluators=["eval1", "eval2"])

    with patch("magda_agent.llm_client.LLMClient.generate", new_callable=AsyncMock) as mock_generate:
        mock_generate.side_effect = [
            "PASSED: Code is solid",
            "PASSED: Looks excellent"
        ]

        result = await team.evaluate_code("def foo(): pass", {"context": "test"})

        assert result["passed"] is True
        assert result["overall_score"] == 100
        assert len(result["results"]) == 2
        assert mock_generate.call_count == 2
