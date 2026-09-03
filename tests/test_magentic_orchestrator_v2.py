"""
Unit tests for Magentic-One Pattern Agent Teams V2.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.architecture.magentic_orchestrator_v2 import (
        BlackboardEntry,
        MagenticOrchestratorV2,
        MagenticSubagentV2,
        ThreadSafeBlackboardState,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "architecture"
        / "magentic_orchestrator_v2.py"
    )
    spec = importlib.util.spec_from_file_location("magentic_orchestrator_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    BlackboardEntry = module.BlackboardEntry
    MagenticOrchestratorV2 = module.MagenticOrchestratorV2
    MagenticSubagentV2 = module.MagenticSubagentV2
    ThreadSafeBlackboardState = module.ThreadSafeBlackboardState


class TestMagenticOrchestratorV2(unittest.TestCase):
    def test_blackboard_get_set_and_revisions(self):
        async def run_async():
            bb = ThreadSafeBlackboardState(initial_state={"init_key": 100})

            # Check initial
            v1 = await bb.get("init_key")
            self.assertEqual(v1, 100)

            # Set
            rev = await bb.set("init_key", 200, agent_id="agent_1")
            self.assertEqual(rev, 1)
            v2 = await bb.get("init_key")
            self.assertEqual(v2, 200)

            # Metadata snapshot
            meta = await bb.get_full_metadata_snapshot()
            self.assertEqual(meta["init_key"]["updated_by"], "agent_1")
            self.assertEqual(meta["init_key"]["revision"], 1)

        asyncio.run(run_async())

    def test_concurrent_atomic_increments_avoid_race_conditions(self):
        async def run_async():
            bb = ThreadSafeBlackboardState(initial_state={"counter": 0})

            async def increment_worker(agent_id: str):
                for _ in range(25):
                    await bb.atomic_update("counter", lambda val: (val or 0) + 1, agent_id=agent_id)
                    await asyncio.sleep(0.001)

            # Run 4 concurrent workers, each incrementing 25 times -> total must be exactly 100
            await asyncio.gather(
                increment_worker("worker_1"),
                increment_worker("worker_2"),
                increment_worker("worker_3"),
                increment_worker("worker_4"),
            )

            final_val = await bb.get("counter")
            self.assertEqual(final_val, 100)

        asyncio.run(run_async())

    def test_concurrent_list_appends(self):
        async def run_async():
            bb = ThreadSafeBlackboardState()

            async def append_worker(agent_id: str, item: str):
                await asyncio.sleep(0.001)
                await bb.append_to_list("items", f"{agent_id}:{item}", agent_id=agent_id)

            await asyncio.gather(*(
                append_worker(f"agent_{i}", f"val_{i}") for i in range(20)
            ))

            items = await bb.get("items")
            self.assertEqual(len(items), 20)

        asyncio.run(run_async())

    def test_orchestrator_parallel_subagent_execution(self):
        async def run_async():
            async def coder_handler(task, bb):
                await bb.set("code_artifact", "def solve(): return 42", agent_id="coder")
                return "Code created"

            async def reviewer_handler(task, bb):
                # Read code from blackboard
                code = await bb.get("code_artifact")
                await bb.set("review_status", "PASSED" if "42" in (code or "") else "FAILED", agent_id="reviewer")
                return "Review finished"

            coder = MagenticSubagentV2("coder", "Coder Subagent", "Coder", coder_handler)
            reviewer = MagenticSubagentV2("reviewer", "Reviewer Subagent", "Reviewer", reviewer_handler)

            orchestrator = MagenticOrchestratorV2(subagents=[coder, reviewer])

            # Execute coder first, then reviewer
            res1 = await orchestrator.execute_parallel_subtasks([("coder", "Write solution")])
            self.assertEqual(res1[0]["output"], "Code created")

            res2 = await orchestrator.execute_parallel_subtasks([("reviewer", "Review solution")])
            self.assertEqual(res2[0]["output"], "Review finished")

            snapshot = await orchestrator.blackboard.get_snapshot()
            self.assertEqual(snapshot["code_artifact"], "def solve(): return 42")
            self.assertEqual(snapshot["review_status"], "PASSED")

        asyncio.run(run_async())

    def test_full_orchestration_loop(self):
        agent_a = MagenticSubagentV2("agent_a", "Agent A", "Searcher")
        agent_b = MagenticSubagentV2("agent_b", "Agent B", "Summarizer")

        orchestrator = MagenticOrchestratorV2(subagents=[agent_a, agent_b])

        res = orchestrator.run_orchestration_sync("Complete user research")
        self.assertEqual(res["status"], "COMPLETED")
        self.assertEqual(len(res["results"]), 2)
        self.assertIn("task_results", res["blackboard_snapshot"])
        self.assertEqual(len(res["blackboard_snapshot"]["task_results"]), 2)


if __name__ == "__main__":
    unittest.main()
