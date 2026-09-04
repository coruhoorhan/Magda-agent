"""
Unit tests for Agent Teams Parallel Subagents V6.
"""

import asyncio
import os
import shutil
import tempfile
import time
import unittest

try:
    from magda_agent.architecture.parallel_execution_v6 import (
        AgentTeamsParallelExecutionManagerV6,
        SubagentExecutionOutcomeV6,
        SubagentExecutionTaskV6,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "architecture"
        / "parallel_execution_v6.py"
    )
    spec = importlib.util.spec_from_file_location("parallel_execution_v6", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    AgentTeamsParallelExecutionManagerV6 = module.AgentTeamsParallelExecutionManagerV6
    SubagentExecutionOutcomeV6 = module.SubagentExecutionOutcomeV6
    SubagentExecutionTaskV6 = module.SubagentExecutionTaskV6


class TestParallelExecutionV6(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = AgentTeamsParallelExecutionManagerV6(
            base_worktree_dir=self.temp_dir,
            max_parallel_subagents=4,
            cleanup_on_completion=False,  # Keep for inspecting paths
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_concurrent_execution_and_isolation(self):
        async def run_async():
            async def subagent_worker(task, wt_path, env):
                # Verify environment variables
                assert env["MAGDA_AGENT_ID"] == task.subagent_id
                assert env["MAGDA_ISOLATED"] == "true"
                assert env["MAGDA_WORKTREE_PATH"] == wt_path

                # Write isolated output file in worktree
                out_file = os.path.join(wt_path, "sub_result.txt")
                with open(out_file, "w") as f:
                    f.write(f"Work from {task.subagent_id}")

                await asyncio.sleep(0.04)
                return f"Done {task.subagent_id}"

            tasks = [
                SubagentExecutionTaskV6(subagent_id=f"agent_{i}", task_name=f"task_{i}")
                for i in range(4)
            ]

            t0 = time.perf_counter()
            outcomes = await self.manager.execute_parallel_tasks_async(tasks, subagent_worker)
            elapsed = time.perf_counter() - t0

            # Parallel: 4 * 0.04s sequential is ~0.16s; parallel should take < 0.12s
            self.assertLess(elapsed, 0.14)

            # Check distinct worktree paths
            worktree_paths = [o.worktree_path for o in outcomes]
            self.assertEqual(len(set(worktree_paths)), 4)

            # Check successful outputs
            self.assertEqual(len(outcomes), 4)
            for i, o in enumerate(outcomes):
                self.assertTrue(o.success)
                self.assertEqual(o.output, f"Done agent_{i}")
                self.assertEqual(o.isolated_env["MAGDA_ISOLATED"], "true")

        asyncio.run(run_async())

    def test_timeout_handling(self):
        async def run_async():
            async def slow_subagent(task, wt, env):
                await asyncio.sleep(0.5)
                return "slow_done"

            task = SubagentExecutionTaskV6(
                subagent_id="slow_agent",
                task_name="slow_task",
                timeout_seconds=0.02,
            )

            outcomes = await self.manager.execute_parallel_tasks_async([task], slow_subagent)
            self.assertEqual(len(outcomes), 1)
            self.assertFalse(outcomes[0].success)
            self.assertTrue(outcomes[0].timed_out)
            self.assertIn("timed out", outcomes[0].error)

        asyncio.run(run_async())

    def test_partial_failure_isolation(self):
        async def run_async():
            def failing_subagent(task, wt, env):
                if task.subagent_id == "failing_agent":
                    raise RuntimeError("Syntax error in generated code")
                return "Good execution"

            tasks = [
                {"subagent_id": "failing_agent", "task_name": "task_fail"},
                {"subagent_id": "healthy_agent", "task_name": "task_good"},
            ]

            outcomes = await self.manager.execute_parallel_tasks_async(tasks, failing_subagent)

            self.assertEqual(len(outcomes), 2)
            self.assertFalse(outcomes[0].success)
            self.assertIn("Syntax error", outcomes[0].error)

            self.assertTrue(outcomes[1].success)
            self.assertEqual(outcomes[1].output, "Good execution")

        asyncio.run(run_async())

    def test_sync_execution_wrapper(self):
        tasks = [{"subagent_id": "sync_agent", "task_name": "sync_task"}]
        outcomes = self.manager.execute_parallel_tasks(tasks)
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0].success)


if __name__ == "__main__":
    unittest.main()
