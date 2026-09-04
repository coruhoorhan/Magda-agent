import asyncio
import json
import unittest
from unittest.mock import AsyncMock

from magda_agent.llm_client import LLMClient
from magda_agent.architecture.magentic_one_v3 import (
    MagenticOneWorkerV3,
    MagenticOneStateMergerV3,
    MagenticOneOrchestratorV3,
)


class TestMagenticOneArchitectureV3Comprehensive(unittest.IsolatedAsyncioTestCase):
    """
    Comprehensive E2E and unit test suite for Magentic-One Architecture V3.
    """

    async def asyncSetUp(self):
        self.mock_llm = AsyncMock(spec=LLMClient)

    # -------------------------------------------------------------------------
    # 1. Worker Execution & State Awareness
    # -------------------------------------------------------------------------
    async def test_worker_execution_success(self):
        """Worker formats prompt with role, specialties, shared state and context, returns structured result."""
        self.mock_llm.chat_completion.return_value = "def solve(): return 42"
        worker = MagenticOneWorkerV3(
            name="Coder",
            description="Writes robust code",
            llm=self.mock_llm,
            specialties=["python", "algorithms"],
        )

        shared_state = {
            "artifacts": {"WebSurfer": {"spec": "Return 42"}},
            "task": "Implement solver",
        }
        context = ["[WebSurfer]: Spec discovered."]

        result = await worker.execute_subtask(
            subtask="Implement the solve function",
            context=context,
            shared_state=shared_state,
        )

        self.assertEqual(result["worker"], "Coder")
        self.assertEqual(result["subtask"], "Implement the solve function")
        self.assertEqual(result["result"], "def solve(): return 42")
        self.assertEqual(result["status"], "completed")

        self.mock_llm.chat_completion.assert_called_once()
        prompt_call = self.mock_llm.chat_completion.call_args[0][0][0]["content"]
        self.assertIn("Coder", prompt_call)
        self.assertIn("python, algorithms", prompt_call)
        self.assertIn("Implement solver", prompt_call)
        self.assertIn("Spec discovered", prompt_call)

    async def test_worker_execution_error_handling(self):
        """Worker catches exceptions during LLM chat and returns error status dict without raising."""
        self.mock_llm.chat_completion.side_effect = RuntimeError("LLM service unavailable")
        worker = MagenticOneWorkerV3(
            name="FileSurfer",
            description="Inspects files",
            llm=self.mock_llm,
        )

        result = await worker.execute_subtask("Read /etc/config", context=[])
        self.assertEqual(result["worker"], "FileSurfer")
        self.assertEqual(result["status"], "error")
        self.assertIn("Error in worker FileSurfer", result["result"])

    # -------------------------------------------------------------------------
    # 2. State Merger Logic & Artifact Preservation
    # -------------------------------------------------------------------------
    async def test_state_merger_aggregation_and_artifacts(self):
        """Merger cleanly accumulates history, maps artifacts by worker and subtask, and sets last_outputs."""
        merger = MagenticOneStateMergerV3()
        state = {
            "task": "Refactor subsystem",
            "history": [],
            "artifacts": {},
        }

        batch_1 = [
            {"worker": "WebSurfer", "subtask": "Search docs", "result": "Found v3 API docs", "status": "completed"},
            {"worker": "FileSurfer", "subtask": "Inspect src/", "result": "3 source files found", "status": "completed"},
        ]
        state = merger.merge_results(state, batch_1)

        self.assertEqual(len(state["history"]), 2)
        self.assertEqual(state["artifacts"]["WebSurfer"]["Search docs"], "Found v3 API docs")
        self.assertEqual(state["artifacts"]["FileSurfer"]["Inspect src/"], "3 source files found")
        self.assertEqual(state["last_outputs"], batch_1)

        batch_2 = [
            {"worker": "Coder", "subtask": "Implement patch", "result": "diff --git ...", "status": "completed"},
        ]
        state = merger.merge_results(state, batch_2)

        self.assertEqual(len(state["history"]), 3)
        self.assertEqual(state["artifacts"]["Coder"]["Implement patch"], "diff --git ...")
        self.assertEqual(state["last_outputs"], batch_2)
        self.assertEqual(state["artifacts"]["WebSurfer"]["Search docs"], "Found v3 API docs")

    # -------------------------------------------------------------------------
    # 3. Intelligent Routing & Worker Matching
    # -------------------------------------------------------------------------
    async def test_routing_by_explicit_hints(self):
        """Routing honors explicit worker hints with case insensitivity."""
        orchestrator = MagenticOneOrchestratorV3(llm=self.mock_llm)
        self.mock_llm.chat_completion.return_value = "Done"

        subtasks = [
            {"subtask": "Check git status", "worker": "filesurfer"},
            {"subtask": "Write unit test", "assigned_worker": "CODER"},
            {"subtask": "Query web index", "worker": "WebSurfer"},
        ]

        results = await orchestrator.route_and_execute_parallel(subtasks, context=[], shared_state={})
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["worker"], "FileSurfer")
        self.assertEqual(results[1]["worker"], "Coder")
        self.assertEqual(results[2]["worker"], "WebSurfer")

    async def test_routing_by_specialty_heuristics(self):
        """Routing matches subtask text to worker specialties when no explicit hint is given."""
        orchestrator = MagenticOneOrchestratorV3(llm=self.mock_llm)
        self.mock_llm.chat_completion.return_value = "Heuristic match executed"

        subtasks = [
            {"subtask": "Debug and fix python software implementation error"},
            {"subtask": "Scrape documentation from research website"},
            {"subtask": "Inspect filesystem directories and disk files"},
            {"subtask": "Reason about high level synthesis and planning"},
        ]

        results = await orchestrator.route_and_execute_parallel(subtasks, context=[], shared_state={})
        self.assertEqual(results[0]["worker"], "Coder")
        self.assertEqual(results[1]["worker"], "WebSurfer")
        self.assertEqual(results[2]["worker"], "FileSurfer")
        self.assertEqual(results[3]["worker"], "Orchestrator")

    # -------------------------------------------------------------------------
    # 4. Plan Generation & Robust Parsing
    # -------------------------------------------------------------------------
    async def test_plan_parsing_markdown_wrapped_json(self):
        """Plan generator extracts JSON when enclosed in markdown ```json ... ``` fences."""
        orchestrator = MagenticOneOrchestratorV3(llm=self.mock_llm)
        fenced_json = "```json\n[\n  {\"subtask\": \"Inspect schemas\", \"worker\": \"FileSurfer\"},\n  {\"subtask\": \"Generate code\", \"worker\": \"Coder\"}\n]\n```"
        self.mock_llm.chat_completion.return_value = fenced_json

        plan = await orchestrator._plan("Design schemas and code", context=[], shared_state={})
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["subtask"], "Inspect schemas")
        self.assertEqual(plan[0]["worker"], "FileSurfer")
        self.assertEqual(plan[1]["subtask"], "Generate code")
        self.assertEqual(plan[1]["worker"], "Coder")

    async def test_plan_parsing_fallback_on_corrupt_response(self):
        """Plan generator returns safe fallback plan when response is non-JSON or invalid."""
        orchestrator = MagenticOneOrchestratorV3(llm=self.mock_llm)
        self.mock_llm.chat_completion.return_value = "I suggest we first look at files then code."

        plan = await orchestrator._plan("Build module", context=[], shared_state={})
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["worker"], "Coder")
        self.assertIn("Build module", plan[0]["subtask"])

    async def test_plan_parsing_string_array_items(self):
        """Plan generator converts raw string items into subtask dicts."""
        orchestrator = MagenticOneOrchestratorV3(llm=self.mock_llm)
        self.mock_llm.chat_completion.return_value = json.dumps(["Task A", "Task B"])

        plan = await orchestrator._plan("Dual tasks", context=[], shared_state={})
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["subtask"], "Task A")
        self.assertEqual(plan[1]["subtask"], "Task B")

    # -------------------------------------------------------------------------
    # 5. Review Step & Criteria
    # -------------------------------------------------------------------------
    async def test_review_success_and_rejection(self):
        """Review evaluates completion based on 'YES' prefix and handles errors."""
        orchestrator = MagenticOneOrchestratorV3(llm=self.mock_llm)

        self.mock_llm.chat_completion.return_value = "YES, all deliverables verified."
        complete, feedback = await orchestrator._review("Build X", context=[], shared_state={})
        self.assertTrue(complete)
        self.assertIn("YES", feedback)

        self.mock_llm.chat_completion.return_value = "NO: Unit tests are still failing."
        complete, feedback = await orchestrator._review("Build X", context=[], shared_state={})
        self.assertFalse(complete)
        self.assertIn("NO", feedback)

        self.mock_llm.chat_completion.side_effect = Exception("Review timeout")
        complete, feedback = await orchestrator._review("Build X", context=[], shared_state={})
        self.assertFalse(complete)
        self.assertIn("Review error", feedback)

    # -------------------------------------------------------------------------
    # 6. Full End-to-End Orchestration Scenarios
    # -------------------------------------------------------------------------
    async def test_e2e_full_lifecycle_single_iteration(self):
        """E2E Test: Single-iteration orchestration executing parallel subtasks and finalizing state."""
        orchestrator = MagenticOneOrchestratorV3(llm=self.mock_llm)

        plan_json = json.dumps([
            {"subtask": "Search algorithm specs", "worker": "WebSurfer"},
            {"subtask": "Implement algorithm in python", "worker": "Coder"},
            {"subtask": "Write test cases to disk", "worker": "FileSurfer"},
        ])
        web_res = "Found spec for A* search."
        code_res = "def a_star(): return 'path'"
        file_res = "test_a_star.py written."
        review_res = "YES The A* algorithm and tests have been successfully delivered."

        self.mock_llm.chat_completion.side_effect = [
            plan_json,
            web_res,
            code_res,
            file_res,
            review_res,
        ]

        final_state = await orchestrator.orchestrate("Implement A* algorithm", max_iterations=3)

        self.assertEqual(final_state["status"], "completed")
        self.assertEqual(final_state["iterations"], 1)
        self.assertIn("A* algorithm", final_state["summary"])
        self.assertEqual(len(final_state["history"]), 3)

        self.assertEqual(final_state["artifacts"]["WebSurfer"]["Search algorithm specs"], web_res)
        self.assertEqual(final_state["artifacts"]["Coder"]["Implement algorithm in python"], code_res)
        self.assertEqual(final_state["artifacts"]["FileSurfer"]["Write test cases to disk"], file_res)

    async def test_e2e_multi_iteration_workflow_with_feedback(self):
        """E2E Test: Multi-iteration workflow where iteration 1 yields feedback and iteration 2 completes."""
        orchestrator = MagenticOneOrchestratorV3(llm=self.mock_llm)

        plan_1 = json.dumps([{"subtask": "Initial draft", "worker": "Coder"}])
        code_1 = "Draft code v1"
        review_1 = "NO: Needs edge case handling for empty inputs."

        plan_2 = json.dumps([{"subtask": "Handle empty inputs", "worker": "Coder"}])
        code_2 = "Added empty input guard: if not arr: return None"
        review_2 = "YES: All edge cases covered and verified."

        self.mock_llm.chat_completion.side_effect = [
            plan_1, code_1, review_1,
            plan_2, code_2, review_2,
        ]

        final_state = await orchestrator.orchestrate("Write robust parser", max_iterations=3)

        self.assertEqual(final_state["status"], "completed")
        self.assertEqual(final_state["iterations"], 2)
        self.assertIn("YES", final_state["summary"])
        self.assertEqual(len(final_state["history"]), 2)
        self.assertEqual(final_state["history"][0]["result"], "Draft code v1")
        self.assertEqual(final_state["history"][1]["result"], "Added empty input guard: if not arr: return None")

    async def test_e2e_max_iterations_exhausted(self):
        """E2E Test: When review repeatedly returns NO, terminates cleanly at max_iterations."""
        orchestrator = MagenticOneOrchestratorV3(llm=self.mock_llm)

        plan = json.dumps([{"subtask": "Attempt step", "worker": "Coder"}])
        res = "Attempted step"
        review = "NO: Still incomplete."

        self.mock_llm.chat_completion.side_effect = [
            plan, res, review,
            plan, res, review,
        ]

        final_state = await orchestrator.orchestrate("Impossible task", max_iterations=2)

        self.assertEqual(final_state["status"], "max_iterations_reached")
        self.assertEqual(final_state["iterations"], 2)
        self.assertIn("Task incomplete after 2 iterations", final_state["summary"])
        self.assertEqual(len(final_state["history"]), 2)

    async def test_e2e_parallel_concurrency_and_error_resilience(self):
        """E2E Test: Parallel execution executes concurrently and is resilient against partial worker errors."""
        orchestrator = MagenticOneOrchestratorV3(llm=self.mock_llm, max_concurrency=2)

        def chat_side_effect(messages):
            prompt = messages[0]["content"]
            if "Parallel task 2" in prompt:
                raise ConnectionError("Worker connection dropped")
            return "Async worker response"

        self.mock_llm.chat_completion.side_effect = chat_side_effect

        subtasks = [
            {"subtask": "Parallel task 1", "worker": "Coder"},
            {"subtask": "Parallel task 2", "worker": "FailingWorker"},
            {"subtask": "Parallel task 3", "worker": "WebSurfer"},
        ]

        results = await orchestrator.route_and_execute_parallel(
            subtasks=subtasks,
            context=[],
            shared_state={},
        )

        self.assertEqual(len(results), 3)
        completed_results = [r for r in results if r["status"] == "completed"]
        error_results = [r for r in results if r["status"] == "error"]

        self.assertEqual(len(completed_results), 2)
        self.assertEqual(len(error_results), 1)
        self.assertIn("Worker connection dropped", error_results[0]["result"])


if __name__ == "__main__":
    unittest.main()
