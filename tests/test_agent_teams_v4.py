"""
Unit tests for Agent Teams Git Worktree Isolation V4.
"""

import asyncio
import os
import shutil
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

try:
    from magda_agent.architecture.agent_teams_v4 import (
        AgentEvaluatorTeamV4,
        AgentTeamManagerV4,
        AgentWorktreeIsolationV4,
        GitWorktreeError,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "architecture"
        / "agent_teams_v4.py"
    )
    spec = importlib.util.spec_from_file_location("agent_teams_v4", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    AgentEvaluatorTeamV4 = module.AgentEvaluatorTeamV4
    AgentTeamManagerV4 = module.AgentTeamManagerV4
    AgentWorktreeIsolationV4 = module.AgentWorktreeIsolationV4
    GitWorktreeError = module.GitWorktreeError


class TestAgentTeamsGitWorktreeIsolationV4(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.isolation_manager = AgentWorktreeIsolationV4(base_dir="/tmp/test_agent_teams_v4")
        self.team_manager = AgentTeamManagerV4(isolation_manager=self.isolation_manager)

    async def test_create_worktree_success(self):
        agent_id = "agent_123"

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"output", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            path, env = await self.isolation_manager.create_worktree(agent_id)

            self.assertTrue(mock_exec.called)
            self.assertIn(agent_id, self.isolation_manager.active_worktrees)
            self.assertEqual(path, self.isolation_manager.active_worktrees[agent_id])

            # Verify isolated env vars
            self.assertEqual(env["MAGDA_AGENT_ID"], agent_id)
            self.assertEqual(env["MAGDA_WORKTREE_PATH"], path)
            self.assertEqual(env["MAGDA_ISOLATED"], "true")

    async def test_create_worktree_failure(self):
        agent_id = "agent_fail"

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"git error")
        mock_process.returncode = 128

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with self.assertRaises(GitWorktreeError):
                await self.isolation_manager.create_worktree(agent_id)

    async def test_remove_worktree_success(self):
        agent_id = "agent_remove"
        dummy_path = "/tmp/test_agent_teams_v4/agent_remove_path"
        self.isolation_manager.active_worktrees[agent_id] = dummy_path

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec, \
             patch("os.path.exists", return_value=False):

            await self.isolation_manager.remove_worktree(agent_id)

            mock_exec.assert_called_with(
                "git", "worktree", "remove", "--force", dummy_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            self.assertNotIn(agent_id, self.isolation_manager.active_worktrees)

    async def test_aggressive_cleanup_fallback(self):
        agent_id = "agent_cleanup"
        dummy_path = "/tmp/test_agent_teams_v4/agent_cleanup_path"
        self.isolation_manager.active_worktrees[agent_id] = dummy_path

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"git error")
        mock_process.returncode = 128

        with patch("asyncio.create_subprocess_exec", return_value=mock_process), \
             patch("os.path.exists", side_effect=[True, False]), \
             patch("shutil.rmtree") as mock_rmtree:

            await self.isolation_manager.remove_worktree(agent_id)

            mock_rmtree.assert_called_once_with(dummy_path, ignore_errors=True)
            self.assertNotIn(agent_id, self.isolation_manager.active_worktrees)

    async def test_team_manager_spawn_and_disband(self):
        agent_id = "agent_manager"

        mock_path = "/tmp/mock_path"
        mock_env = {"MAGDA_AGENT_ID": agent_id}
        self.team_manager.isolation_manager.create_worktree = AsyncMock(return_value=(mock_path, mock_env))
        self.team_manager.isolation_manager.remove_worktree = AsyncMock()

        # Test spawn
        path, env = await self.team_manager.spawn_agent(agent_id)
        self.assertEqual(path, mock_path)
        self.assertEqual(env, mock_env)
        self.assertIn(agent_id, self.team_manager.agents)
        self.assertEqual(self.team_manager.get_agent_env(agent_id), mock_env)

        # Test duplicate spawn
        with self.assertRaises(ValueError):
            await self.team_manager.spawn_agent(agent_id)

        # Test disband
        await self.team_manager.disband_agent(agent_id)
        self.assertNotIn(agent_id, self.team_manager.agents)
        self.assertIsNone(self.team_manager.get_agent_env(agent_id))
        self.team_manager.isolation_manager.remove_worktree.assert_called_once_with(agent_id)

    async def test_team_manager_disband_all(self):
        self.team_manager.agents = ["agent1", "agent2"]
        self.team_manager.agent_envs = {"agent1": {}, "agent2": {}}

        with patch.object(self.team_manager, "disband_agent", new_callable=AsyncMock) as mock_disband:
            await self.team_manager.disband_all()

            self.assertEqual(mock_disband.call_count, 2)
            mock_disband.assert_any_call("agent1")
            mock_disband.assert_any_call("agent2")

    async def test_agent_evaluator_team_v4(self):
        evaluators = ["eval1", "eval2"]
        team = AgentEvaluatorTeamV4(evaluators=evaluators, team_manager=self.team_manager)

        with patch("magda_agent.llm_client.LLMClient.generate", new_callable=AsyncMock) as mock_generate, \
             patch.object(self.team_manager, "spawn_agent", new_callable=AsyncMock) as mock_spawn, \
             patch.object(self.team_manager, "disband_agent", new_callable=AsyncMock) as mock_disband:

            mock_generate.side_effect = [
                "PASSED: Code is good",
                "PASSED: Looks fine",
            ]

            mock_spawn.side_effect = [
                ("/tmp/eval1", {"MAGDA_WORKTREE_PATH": "/tmp/eval1"}),
                ("/tmp/eval2", {"MAGDA_WORKTREE_PATH": "/tmp/eval2"}),
            ]

            result = await team.evaluate_code("def foo(): pass", {"context": "test"})

            self.assertTrue(result["passed"])
            self.assertEqual(result["overall_score"], 100)
            self.assertEqual(len(result["results"]), 2)

            self.assertEqual(mock_generate.call_count, 2)
            self.assertEqual(mock_spawn.call_count, 2)
            self.assertEqual(mock_disband.call_count, 2)

            mock_spawn.assert_any_call("eval1")
            mock_spawn.assert_any_call("eval2")
            mock_disband.assert_any_call("eval1")
            mock_disband.assert_any_call("eval2")

    async def test_agent_evaluator_team_v4_failed_eval(self):
        evaluators = ["eval1", "eval2"]
        team = AgentEvaluatorTeamV4(evaluators=evaluators, team_manager=self.team_manager)

        with patch("magda_agent.llm_client.LLMClient.generate", new_callable=AsyncMock) as mock_generate, \
             patch.object(self.team_manager, "spawn_agent", new_callable=AsyncMock) as mock_spawn, \
             patch.object(self.team_manager, "disband_agent", new_callable=AsyncMock) as mock_disband:

            mock_generate.side_effect = [
                "PASSED: Good",
                "FAILED: Syntax Error",
            ]

            mock_spawn.side_effect = [
                ("/tmp/eval1", {"MAGDA_WORKTREE_PATH": "/tmp/eval1"}),
                ("/tmp/eval2", {"MAGDA_WORKTREE_PATH": "/tmp/eval2"}),
            ]

            result = await team.evaluate_code("def foo() pass", {"context": "test"})

            self.assertFalse(result["passed"])
            self.assertEqual(result["overall_score"], 75.0)  # (100 + 50) / 2
            self.assertEqual(len(result["results"]), 2)

            self.assertEqual(mock_disband.call_count, 2)


if __name__ == "__main__":
    unittest.main()
