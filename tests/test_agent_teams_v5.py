"""
Unit tests for Agent Teams Git Worktree Synchronization V5.
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

try:
    from magda_agent.architecture.agent_teams_v5 import (
        AgentTeamManagerV5,
        AgentWorktreeIsolationV5,
        GitSyncError,
        GitSyncStrategy,
        GitWorktreeError,
        SyncResult,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "architecture"
        / "agent_teams_v5.py"
    )
    spec = importlib.util.spec_from_file_location("agent_teams_v5", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    AgentTeamManagerV5 = module.AgentTeamManagerV5
    AgentWorktreeIsolationV5 = module.AgentWorktreeIsolationV5
    GitSyncError = module.GitSyncError
    GitSyncStrategy = module.GitSyncStrategy
    GitWorktreeError = module.GitWorktreeError
    SyncResult = module.SyncResult


class TestAgentTeamsV5(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.isolation = AgentWorktreeIsolationV5(base_dir="/tmp/test_wt_v5")
        self.team_mgr = AgentTeamManagerV5(isolation_manager=self.isolation)

    async def test_create_worktree_success(self):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            path, env = await self.team_mgr.spawn_agent("agent_sync_1")
            self.assertIn("agent_sync_1", self.team_mgr.agents)
            self.assertEqual(env["MAGDA_AGENT_ID"], "agent_sync_1")
            self.assertIn("wt_agent_sync_1", path)

    async def test_commit_changes(self):
        self.isolation.active_worktrees["agent_c"] = "/tmp/test_wt_v5/wt_agent_c"
        self.isolation.active_branches["agent_c"] = "agent/agent_c_123"

        mock_proc_add = AsyncMock()
        mock_proc_add.communicate.return_value = (b"", b"")
        mock_proc_add.returncode = 0

        mock_proc_status = AsyncMock()
        mock_proc_status.communicate.return_value = (b"M file.py", b"")
        mock_proc_status.returncode = 0

        mock_proc_commit = AsyncMock()
        mock_proc_commit.communicate.return_value = (b"[master 12345] committed", b"")
        mock_proc_commit.returncode = 0

        with patch("asyncio.create_subprocess_exec", side_effect=[mock_proc_add, mock_proc_status, mock_proc_commit]):
            committed = await self.isolation.commit_changes("agent_c", message="Fix bug")
            self.assertTrue(committed)

    async def test_sync_worktree_rebase_success(self):
        self.isolation.active_worktrees["agent_s"] = "/tmp/test_wt_v5/wt_agent_s"
        self.isolation.active_branches["agent_s"] = "agent/agent_s_123"

        mock_proc_add = AsyncMock(returncode=0)
        mock_proc_add.communicate.return_value = (b"", b"")

        mock_proc_status = AsyncMock(returncode=0)
        mock_proc_status.communicate.return_value = (b"", b"")  # clean

        mock_proc_revlist = AsyncMock(returncode=0)
        mock_proc_revlist.communicate.return_value = (b"2", b"")

        mock_proc_rebase = AsyncMock(returncode=0)
        mock_proc_rebase.communicate.return_value = (b"Successfully rebased", b"")

        with patch("asyncio.create_subprocess_exec", side_effect=[mock_proc_add, mock_proc_status, mock_proc_revlist, mock_proc_rebase]):
            res = await self.isolation.sync_worktree("agent_s", target_branch="main", strategy=GitSyncStrategy.REBASE)
            self.assertTrue(res.success)
            self.assertEqual(res.commits_synced, 2)
            self.assertFalse(res.conflict_detected)
            self.assertEqual(res.strategy_used, GitSyncStrategy.REBASE)

    async def test_sync_worktree_rebase_conflict(self):
        self.isolation.active_worktrees["agent_conflict"] = "/tmp/test_wt_v5/wt_agent_conflict"
        self.isolation.active_branches["agent_conflict"] = "agent/agent_conflict_123"

        mock_proc_add = AsyncMock(returncode=0)
        mock_proc_add.communicate.return_value = (b"", b"")

        mock_proc_status = AsyncMock(returncode=0)
        mock_proc_status.communicate.return_value = (b"", b"")

        mock_proc_revlist = AsyncMock(returncode=0)
        mock_proc_revlist.communicate.return_value = (b"1", b"")

        # Rebase fails due to conflict
        mock_proc_rebase_fail = AsyncMock(returncode=1)
        mock_proc_rebase_fail.communicate.return_value = (b"", b"CONFLICT: merge conflict in foo.py")

        # Abort rebase succeeds
        mock_proc_abort = AsyncMock(returncode=0)
        mock_proc_abort.communicate.return_value = (b"", b"")

        with patch("asyncio.create_subprocess_exec", side_effect=[
            mock_proc_add, mock_proc_status, mock_proc_revlist, mock_proc_rebase_fail, mock_proc_abort
        ]):
            res = await self.isolation.sync_worktree("agent_conflict", target_branch="main", strategy=GitSyncStrategy.REBASE)
            self.assertFalse(res.success)
            self.assertTrue(res.conflict_detected)
            self.assertIn("Rebase conflict", res.error_message)

    async def test_synchronize_all_team_manager(self):
        self.team_mgr.agents = ["agent_1", "agent_2"]
        mock_sync_1 = SyncResult(agent_id="agent_1", branch_name="b1", success=True, commits_synced=1)
        mock_sync_2 = SyncResult(agent_id="agent_2", branch_name="b2", success=True, commits_synced=3)

        with patch.object(self.isolation, "sync_worktree", side_effect=[mock_sync_1, mock_sync_2]):
            results = await self.team_mgr.synchronize_all(target_branch="main")
            self.assertEqual(len(results), 2)
            self.assertTrue(results["agent_1"].success)
            self.assertTrue(results["agent_2"].success)
            self.assertEqual(len(self.team_mgr.sync_history), 2)

    async def test_disband_cleanup(self):
        self.team_mgr.agents = ["agent_d"]
        self.isolation.active_worktrees["agent_d"] = "/tmp/test_wt_v5/wt_agent_d"
        self.isolation.active_branches["agent_d"] = "b_d"

        mock_proc_rm = AsyncMock(returncode=0)
        mock_proc_rm.communicate.return_value = (b"", b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc_rm):
            await self.team_mgr.disband_agent("agent_d")
            self.assertNotIn("agent_d", self.team_mgr.agents)
            self.assertNotIn("agent_d", self.isolation.active_worktrees)


if __name__ == "__main__":
    unittest.main()
