"""
Unit tests for Claude Subagent Dependency Graph V2.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.architecture.dependency_graph_v2 import (
        ClaudeSubagentDependencyGraphV2,
        DependencyGraphError,
        DependencyGraphValidationResult,
        SubagentTaskNode,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "architecture"
        / "dependency_graph_v2.py"
    )
    spec = importlib.util.spec_from_file_location("dependency_graph_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ClaudeSubagentDependencyGraphV2 = module.ClaudeSubagentDependencyGraphV2
    DependencyGraphError = module.DependencyGraphError
    DependencyGraphValidationResult = module.DependencyGraphValidationResult
    SubagentTaskNode = module.SubagentTaskNode


class TestClaudeSubagentDependencyGraphV2(unittest.TestCase):
    def setUp(self):
        self.graph = ClaudeSubagentDependencyGraphV2()

    def test_valid_dag_validation_and_topological_sort(self):
        self.graph.add_task("task_plan", name="Create execution plan", dependencies=[])
        self.graph.add_task("task_code", name="Write implementation", dependencies=["task_plan"])
        self.graph.add_task("task_test", name="Run test suite", dependencies=["task_code"])

        res = self.graph.validate_graph()

        self.assertTrue(res.is_valid)
        self.assertFalse(res.has_cycles)
        self.assertEqual(res.topological_order, ["task_plan", "task_code", "task_test"])
        self.assertEqual(len(res.parallel_waves), 3)

    def test_cycle_detection(self):
        # A -> B -> C -> A (Cycle)
        self.graph.add_task("A", dependencies=["C"])
        self.graph.add_task("B", dependencies=["A"])
        self.graph.add_task("C", dependencies=["B"])

        res = self.graph.validate_graph()

        self.assertFalse(res.is_valid)
        self.assertTrue(res.has_cycles)
        self.assertTrue(any("Cyclic dependency" in e for e in res.errors))

    def test_missing_dependency_detection(self):
        self.graph.add_task("task_1", dependencies=["non_existent_task"])

        res = self.graph.validate_graph()

        self.assertFalse(res.is_valid)
        self.assertIn("task_1", res.missing_dependencies)
        self.assertIn("non_existent_task", res.missing_dependencies["task_1"])

    def test_parallel_waves_generation(self):
        # T1 (root) -> T2, T3 (parallel) -> T4 (join)
        self.graph.add_task("T1", dependencies=[])
        self.graph.add_task("T2", dependencies=["T1"])
        self.graph.add_task("T3", dependencies=["T1"])
        self.graph.add_task("T4", dependencies=["T2", "T3"])

        res = self.graph.validate_graph()

        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.parallel_waves), 3)
        self.assertEqual(res.parallel_waves[0], ["T1"])
        self.assertEqual(set(res.parallel_waves[1]), {"T2", "T3"})
        self.assertEqual(res.parallel_waves[2], ["T4"])

    def test_step_by_step_execution_tracking(self):
        self.graph.add_task("A", dependencies=[])
        self.graph.add_task("B", dependencies=["A"])
        self.graph.add_task("C", dependencies=["B"])

        # Step 1: Initial executable task is A
        next_tasks = self.graph.get_next_executable_tasks()
        self.assertEqual([t.id for t in next_tasks], ["A"])

        # Complete A
        self.graph.mark_task_completed("A", output="A_done")
        self.assertFalse(self.graph.is_complete())

        # Step 2: Next executable is B
        next_tasks = self.graph.get_next_executable_tasks()
        self.assertEqual([t.id for t in next_tasks], ["B"])

        # Complete B
        self.graph.mark_task_completed("B", output="B_done")

        # Step 3: Next executable is C
        next_tasks = self.graph.get_next_executable_tasks()
        self.assertEqual([t.id for t in next_tasks], ["C"])

        # Complete C
        self.graph.mark_task_completed("C", output="C_done")
        self.assertTrue(self.graph.is_complete())

    def test_async_subagent_pipeline_execution(self):
        async def run_async():
            self.graph.add_task("build", dependencies=[])
            self.graph.add_task("lint", dependencies=["build"])
            self.graph.add_task("test", dependencies=["build"])

            async def mock_subagent(task):
                await asyncio.sleep(0.01)
                return f"Result for {task.id}"

            results = await self.graph.execute_subagent_pipeline_async(mock_subagent)

            self.assertEqual(len(results), 3)
            self.assertEqual(results["build"]["output"], "Result for build")
            self.assertEqual(results["lint"]["output"], "Result for lint")
            self.assertEqual(results["test"]["output"], "Result for test")
            self.assertTrue(self.graph.is_complete())

        asyncio.run(run_async())


if __name__ == "__main__":
    unittest.main()
