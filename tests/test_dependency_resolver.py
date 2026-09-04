"""
Tests for Hierarchical Planner Subagent Dependency Resolver.
"""

import unittest

try:
    from magda_agent.architecture.dependency_resolver import (
        SubagentDependencyResolver,
        PlannerTask,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "architecture" / "dependency_resolver.py"
    spec = importlib.util.spec_from_file_location("dependency_resolver", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    SubagentDependencyResolver = module.SubagentDependencyResolver
    PlannerTask = module.PlannerTask


class TestSubagentDependencyResolver(unittest.TestCase):
    """
    Test suite verifying DAG resolution, cycle detection, parallel batch grouping,
    and critical path analysis for hierarchical subagents.
    """

    # -------------------------------------------------------------------------
    # 1. Topological Sort & Resolution
    # -------------------------------------------------------------------------
    def test_resolve_dependencies_linear(self):
        """Sequential tasks should resolve in topological sequence."""
        tasks = [
            PlannerTask(id="task3", dependencies=["task2"]),
            PlannerTask(id="task1", dependencies=[]),
            PlannerTask(id="task2", dependencies=["task1"]),
        ]

        resolved = SubagentDependencyResolver.resolve_dependencies(tasks)
        ids = [t.id for t in resolved]
        self.assertEqual(ids, ["task1", "task2", "task3"])

    def test_resolve_dependencies_complex_dag(self):
        """Branching and joining DAG should place predecessors before consumers."""
        tasks = [
            {"id": "test", "deps": ["build_backend", "build_frontend"]},
            {"id": "build_frontend", "deps": ["init"]},
            {"id": "build_backend", "deps": ["init"]},
            {"id": "deploy", "deps": ["test"]},
            {"id": "init", "deps": []},
        ]

        resolved = SubagentDependencyResolver.resolve_dependencies(tasks)
        ids = [t.id for t in resolved]

        self.assertEqual(ids[0], "init")
        self.assertEqual(ids[-1], "deploy")
        self.assertLess(ids.index("init"), ids.index("build_backend"))
        self.assertLess(ids.index("init"), ids.index("build_frontend"))
        self.assertLess(ids.index("build_backend"), ids.index("test"))
        self.assertLess(ids.index("build_frontend"), ids.index("test"))

    def test_resolve_dependencies_cycle_detection(self):
        """Cycles in dependency plans must raise ValueError."""
        cyclic_tasks = [
            PlannerTask(id="A", dependencies=["B"]),
            PlannerTask(id="B", dependencies=["C"]),
            PlannerTask(id="C", dependencies=["A"]),
        ]

        with self.assertRaises(ValueError) as ctx:
            SubagentDependencyResolver.resolve_dependencies(cyclic_tasks)
        self.assertIn("Cycle detected", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 2. Parallel Task Batch Grouping
    # -------------------------------------------------------------------------
    def test_group_independent_tasks_batches(self):
        """Independent tasks should be clustered into execution tiers for subagents."""
        tasks = [
            PlannerTask(id="scrape_a", dependencies=[]),
            PlannerTask(id="scrape_b", dependencies=[]),
            PlannerTask(id="scrape_c", dependencies=[]),
            PlannerTask(id="merge", dependencies=["scrape_a", "scrape_b", "scrape_c"]),
            PlannerTask(id="generate_report", dependencies=["merge"]),
        ]

        batches = SubagentDependencyResolver.group_independent_tasks(tasks)
        self.assertEqual(len(batches), 3)

        # Batch 0: 3 parallel scraping tasks
        batch_0_ids = {t.id for t in batches[0]}
        self.assertEqual(batch_0_ids, {"scrape_a", "scrape_b", "scrape_c"})

        # Batch 1: merge task
        batch_1_ids = {t.id for t in batches[1]}
        self.assertEqual(batch_1_ids, {"merge"})

        # Batch 2: generate report
        batch_2_ids = {t.id for t in batches[2]}
        self.assertEqual(batch_2_ids, {"generate_report"})

    # -------------------------------------------------------------------------
    # 3. Next Executable Tasks Query
    # -------------------------------------------------------------------------
    def test_get_next_executable_tasks(self):
        """Should yield unblocked tasks progressively as completed_task_ids updates."""
        tasks = [
            PlannerTask(id="t1", dependencies=[]),
            PlannerTask(id="t2", dependencies=["t1"]),
            PlannerTask(id="t3", dependencies=["t1"]),
            PlannerTask(id="t4", dependencies=["t2", "t3"]),
        ]

        # Initial state: only t1 is ready
        ready_0 = SubagentDependencyResolver.get_next_executable_tasks(tasks, set())
        self.assertEqual([t.id for t in ready_0], ["t1"])

        # After t1 completes: t2 and t3 become ready
        ready_1 = SubagentDependencyResolver.get_next_executable_tasks(tasks, {"t1"})
        self.assertEqual({t.id for t in ready_1}, {"t2", "t3"})

        # After t2 completes: t4 not yet ready because t3 is pending
        ready_2 = SubagentDependencyResolver.get_next_executable_tasks(tasks, {"t1", "t2"})
        self.assertEqual([t.id for t in ready_2], ["t3"])

        # After t2 and t3 complete: t4 is ready
        ready_3 = SubagentDependencyResolver.get_next_executable_tasks(tasks, {"t1", "t2", "t3"})
        self.assertEqual([t.id for t in ready_3], ["t4"])

    # -------------------------------------------------------------------------
    # 4. Critical Path Calculation
    # -------------------------------------------------------------------------
    def test_calculate_critical_path(self):
        """Should correctly find the longest path through the DAG."""
        tasks = [
            PlannerTask(id="init", estimated_duration=2.0, dependencies=[]),
            PlannerTask(id="short_branch", estimated_duration=1.0, dependencies=["init"]),
            PlannerTask(id="long_branch", estimated_duration=5.0, dependencies=["init"]),
            PlannerTask(id="finish", estimated_duration=3.0, dependencies=["short_branch", "long_branch"]),
        ]

        path, duration = SubagentDependencyResolver.calculate_critical_path(tasks)
        # Longest path is init (2) -> long_branch (5) -> finish (3) = 10.0
        self.assertEqual(path, ["init", "long_branch", "finish"])
        self.assertEqual(duration, 10.0)


if __name__ == "__main__":
    unittest.main()
