"""
Tests for Magentic-One Task Dependency Resolution Engine v4.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock

try:
    from magda_agent.architecture.dependency_router_v4 import (
        MagenticOneDependencyRouterV4,
        DAGTaskNode,
        DAGCycleDetectedError,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "architecture" / "dependency_router_v4.py"
    spec = importlib.util.spec_from_file_location("dependency_router_v4", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    MagenticOneDependencyRouterV4 = module.MagenticOneDependencyRouterV4
    DAGTaskNode = module.DAGTaskNode
    DAGCycleDetectedError = module.DAGCycleDetectedError


class TestDependencyRouterV4(unittest.IsolatedAsyncioTestCase):
    """
    Comprehensive test suite verifying DAG topological resolution,
    execution wave computation, concurrency grouping, and artifact propagation.
    """

    def setUp(self):
        self.router = MagenticOneDependencyRouterV4()

    # -------------------------------------------------------------------------
    # 1. Topological Sorting & Ordering
    # -------------------------------------------------------------------------
    def test_topological_sort_linear(self):
        """Sequential dependency chain A -> B -> C should sort in exact order."""
        tasks = [
            DAGTaskNode(task_id="step_c", name="Deploy", dependencies=["step_b"]),
            DAGTaskNode(task_id="step_a", name="Build", dependencies=[]),
            DAGTaskNode(task_id="step_b", name="Test", dependencies=["step_a"]),
        ]

        sorted_tasks = self.router.topological_sort(tasks)
        ids = [t.task_id for t in sorted_tasks]
        self.assertEqual(ids, ["step_a", "step_b", "step_c"])

    def test_topological_sort_diamond_dag(self):
        """Diamond DAG: A -> (B, C) -> D should have A first and D last."""
        tasks = [
            DAGTaskNode(task_id="D", name="Merge", dependencies=["B", "C"]),
            DAGTaskNode(task_id="B", name="Worker 1", dependencies=["A"]),
            DAGTaskNode(task_id="A", name="Root", dependencies=[]),
            DAGTaskNode(task_id="C", name="Worker 2", dependencies=["A"]),
        ]

        sorted_tasks = self.router.topological_sort(tasks)
        ids = [t.task_id for t in sorted_tasks]
        self.assertEqual(ids[0], "A")
        self.assertEqual(ids[-1], "D")
        self.assertIn("B", ids[1:3])
        self.assertIn("C", ids[1:3])

    def test_topological_sort_cycle_detected(self):
        """Cyclical dependencies (A -> B -> A) must raise DAGCycleDetectedError."""
        cyclic_tasks = [
            DAGTaskNode(task_id="A", name="Task A", dependencies=["B"]),
            DAGTaskNode(task_id="B", name="Task B", dependencies=["A"]),
        ]

        with self.assertRaises(DAGCycleDetectedError):
            self.router.topological_sort(cyclic_tasks)

    # -------------------------------------------------------------------------
    # 2. Execution Waves & Concurrency Grouping
    # -------------------------------------------------------------------------
    def test_compute_execution_waves(self):
        """Tasks should be grouped into parallel waves where independent tasks share a wave."""
        tasks = [
            DAGTaskNode(task_id="A1", name="Fetch A1", dependencies=[]),
            DAGTaskNode(task_id="A2", name="Fetch A2", dependencies=[]),
            DAGTaskNode(task_id="B1", name="Process A1", dependencies=["A1"]),
            DAGTaskNode(task_id="B2", name="Process A2", dependencies=["A2"]),
            DAGTaskNode(task_id="C1", name="Aggregate B1 & B2", dependencies=["B1", "B2"]),
        ]

        waves = self.router.compute_execution_waves(tasks)
        self.assertEqual(len(waves), 3)

        # Wave 0: A1, A2 (concurrent root tasks)
        wave_0_ids = {t.task_id for t in waves[0]}
        self.assertEqual(wave_0_ids, {"A1", "A2"})

        # Wave 1: B1, B2 (concurrent second tier)
        wave_1_ids = {t.task_id for t in waves[1]}
        self.assertEqual(wave_1_ids, {"B1", "B2"})

        # Wave 2: C1 (final sink task)
        wave_2_ids = {t.task_id for t in waves[2]}
        self.assertEqual(wave_2_ids, {"C1"})

    # -------------------------------------------------------------------------
    # 3. End-to-End DAG Routing & Subagent Dispatching
    # -------------------------------------------------------------------------
    async def test_route_and_execute_dag_with_mock_subagents(self):
        """Subagents should execute DAG tasks and pass upstream artifacts downstream."""
        dispatched_calls = []

        async def mock_subagent_dispatcher(task: DAGTaskNode, ctx: dict) -> dict:
            dispatched_calls.append(task.task_id)
            upstream = ctx.get("upstream_artifacts", {})
            return {
                "task_name": task.name,
                "role": task.assigned_role,
                "received_upstream": upstream,
            }

        router = MagenticOneDependencyRouterV4(subagent_dispatcher=mock_subagent_dispatcher)

        plan = [
            {"id": "fetch_data", "name": "Fetch API data", "role": "surfer", "deps": []},
            {"id": "parse_data", "name": "Parse raw data", "role": "coder", "deps": ["fetch_data"]},
            {"id": "save_db", "name": "Save to DB", "role": "writer", "deps": ["parse_data"]},
        ]

        result = await router.route_and_execute_dag(plan)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total_tasks"], 3)
        self.assertEqual(result["completed_count"], 3)
        self.assertEqual(result["failed_count"], 0)
        self.assertEqual(result["waves_count"], 3)

        # Verify downstream task received upstream output
        save_db_output = result["results"]["save_db"]
        self.assertIn("parse_data", save_db_output["received_upstream"])

    async def test_dependency_failure_propagation(self):
        """When an upstream task fails, downstream dependent tasks should be skipped."""
        async def failing_subagent_dispatcher(task: DAGTaskNode, ctx: dict) -> str:
            if task.task_id == "step_fail":
                raise RuntimeError("Network timeout during fetch")
            return "Success"

        router = MagenticOneDependencyRouterV4(subagent_dispatcher=failing_subagent_dispatcher)

        tasks = [
            DAGTaskNode(task_id="step_fail", name="Failing Step", dependencies=[]),
            DAGTaskNode(task_id="step_dep", name="Dependent Step", dependencies=["step_fail"]),
            DAGTaskNode(task_id="step_indep", name="Independent Step", dependencies=[]),
        ]

        result = await router.route_and_execute_dag(tasks)

        self.assertEqual(result["status"], "partial_failure")
        self.assertEqual(result["completed_count"], 1)  # step_indep completed
        self.assertEqual(result["failed_count"], 2)     # step_fail (failed) + step_dep (skipped)

        step_dep_dict = next(t for t in result["tasks"] if t["task_id"] == "step_dep")
        self.assertEqual(step_dep_dict["status"], "skipped")
        self.assertIn("Prerequisite dependency failed", step_dep_dict["error"])

    # -------------------------------------------------------------------------
    # 4. Worktree Isolation Path Assignment
    # -------------------------------------------------------------------------
    async def test_worktree_isolation_path_provisioning(self):
        """Tasks with worktree_isolation=True should have distinct worktree paths."""
        router = MagenticOneDependencyRouterV4()

        tasks = [
            DAGTaskNode(task_id="t1", name="Task 1", assigned_role="coder", worktree_isolation=True),
            DAGTaskNode(task_id="t2", name="Task 2", assigned_role="reviewer", worktree_isolation=False),
        ]

        result = await router.route_and_execute_dag(tasks)
        t1_dict = next(t for t in result["tasks"] if t["task_id"] == "t1")
        t2_dict = next(t for t in result["tasks"] if t["task_id"] == "t2")

        self.assertIsNotNone(t1_dict["worktree_path"])
        self.assertIn("/tmp/worktrees/t1_coder", t1_dict["worktree_path"])
        self.assertIsNone(t2_dict["worktree_path"])


if __name__ == "__main__":
    unittest.main()
