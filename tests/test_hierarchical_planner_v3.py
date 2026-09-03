"""
Unit tests for Claude Agent SDK Hierarchical Planner V3.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.architecture.hierarchical_planner_v3 import (
        ClaudeHierarchicalPlannerV3,
        DecompositionPlan,
        PlanSubTask,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "architecture"
        / "hierarchical_planner_v3.py"
    )
    spec = importlib.util.spec_from_file_location("hierarchical_planner_v3", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ClaudeHierarchicalPlannerV3 = module.ClaudeHierarchicalPlannerV3
    DecompositionPlan = module.DecompositionPlan
    PlanSubTask = module.PlanSubTask


class TestHierarchicalPlannerV3(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.planner = ClaudeHierarchicalPlannerV3(llm_client=self.mock_llm)

    def test_heuristic_goal_decomposition(self):
        # Without LLM
        planner_no_llm = ClaudeHierarchicalPlannerV3(llm_client=None)

        plan = planner_no_llm.decompose_goal("Build distributed cache")

        self.assertEqual(plan.original_goal, "Build distributed cache")
        self.assertEqual(len(plan.subtasks), 5)
        self.assertEqual(plan.topological_order[0], "step_1_research")
        self.assertEqual(plan.topological_order[-1], "step_5_review")
        self.assertEqual(len(plan.parallel_stages), 5)

    def test_mock_llm_goal_decomposition(self):
        async def run_async():
            mock_plan_json = json.dumps({
                "subtasks": [
                    {
                        "task_id": "sub1_spec",
                        "title": "Write Spec",
                        "assigned_role": "architect",
                        "dependencies": [],
                    },
                    {
                        "task_id": "sub2_db",
                        "title": "DB Migration",
                        "assigned_role": "coder",
                        "dependencies": ["sub1_spec"],
                    },
                    {
                        "task_id": "sub3_api",
                        "title": "API Routes",
                        "assigned_role": "coder",
                        "dependencies": ["sub1_spec"],
                    },
                    {
                        "task_id": "sub4_test",
                        "title": "Integration Test",
                        "assigned_role": "tester",
                        "dependencies": ["sub2_db", "sub3_api"],
                    },
                ]
            })

            self.mock_llm.generate = AsyncMock(return_value=mock_plan_json)

            plan = await self.planner.decompose_goal_async("Implement user auth service")

            self.assertEqual(len(plan.subtasks), 4)
            self.assertEqual(plan.topological_order[0], "sub1_spec")
            self.assertEqual(plan.topological_order[-1], "sub4_test")

            # sub2_db and sub3_api should be in the same parallel stage
            self.assertEqual(len(plan.parallel_stages), 3)
            self.assertEqual(plan.parallel_stages[0], ["sub1_spec"])
            self.assertEqual(set(plan.parallel_stages[1]), {"sub2_db", "sub3_api"})
            self.assertEqual(plan.parallel_stages[2], ["sub4_test"])

        asyncio.run(run_async())

    def test_get_executable_subtasks_incremental(self):
        planner_no_llm = ClaudeHierarchicalPlannerV3(llm_client=None)
        plan = planner_no_llm.decompose_goal("Refactor core")

        # Initial executable task is step_1_research
        next_tasks = planner_no_llm.get_executable_subtasks(plan)
        self.assertEqual(len(next_tasks), 1)
        self.assertEqual(next_tasks[0].task_id, "step_1_research")

        # Complete step 1
        planner_no_llm.update_subtask_status(plan, "step_1_research", "completed", "Research done")

        # Next executable is step_2_design
        next_tasks = planner_no_llm.get_executable_subtasks(plan)
        self.assertEqual(len(next_tasks), 1)
        self.assertEqual(next_tasks[0].task_id, "step_2_design")

    def test_simulate_plan_execution(self):
        planner_no_llm = ClaudeHierarchicalPlannerV3(llm_client=None)
        plan = planner_no_llm.decompose_goal("Deploy auth microservice")

        def custom_mock_executor(st):
            return f"Role {st.assigned_role} successfully completed {st.task_id}"

        sim_res = planner_no_llm.simulate_plan_execution(plan, custom_mock_executor)

        self.assertTrue(sim_res["is_complete"])
        self.assertEqual(sim_res["total_subtasks"], 5)
        self.assertIn("step_3_implement", sim_res["results"])
        self.assertEqual(
            sim_res["results"]["step_3_implement"]["output"],
            "Role coder successfully completed step_3_implement",
        )

    def test_cycle_detection_in_subtasks(self):
        planner_no_llm = ClaudeHierarchicalPlannerV3(llm_client=None)

        # Invalid cyclic tasks
        cyclic_tasks = [
            PlanSubTask(task_id="A", title="Task A", dependencies=["B"]),
            PlanSubTask(task_id="B", title="Task B", dependencies=["A"]),
        ]

        with self.assertRaises(ValueError) as ctx:
            planner_no_llm._topological_sort_and_stages(cyclic_tasks)

        self.assertIn("Cycle detected", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
