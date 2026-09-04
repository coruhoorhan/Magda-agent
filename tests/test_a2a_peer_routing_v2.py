"""
Unit tests for A2A Peer Delegation Routing V2.
"""

import asyncio
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.integration.a2a_peer_routing_v2 import (
        A2APeerDelegationPayload,
        A2APeerDelegationRouterV2,
        ComplexityEvaluationResult,
        DelegationResult,
        PeerAgentCard,
        TaskComplexityEvaluator,
        TaskComplexityLevel,
        TaskSpec,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "integration"
        / "a2a_peer_routing_v2.py"
    )
    spec = importlib.util.spec_from_file_location("a2a_peer_routing_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    A2APeerDelegationPayload = module.A2APeerDelegationPayload
    A2APeerDelegationRouterV2 = module.A2APeerDelegationRouterV2
    ComplexityEvaluationResult = module.ComplexityEvaluationResult
    DelegationResult = module.DelegationResult
    PeerAgentCard = module.PeerAgentCard
    TaskComplexityEvaluator = module.TaskComplexityEvaluator
    TaskComplexityLevel = module.TaskComplexityLevel
    TaskSpec = module.TaskSpec


class TestA2APeerRoutingV2(unittest.TestCase):
    def setUp(self):
        self.router = A2APeerDelegationRouterV2(sender_id="magda_test")

        # Register specialized test peers
        self.coder_peer = PeerAgentCard(
            agent_id="peer_coder_01",
            name="Coder Subagent",
            description="Specialized code generator and refactor subagent",
            capabilities=["python", "refactoring", "unit_testing"],
            max_complexity=TaskComplexityLevel.HIGH,
            workload_score=0.2,
        )

        self.researcher_peer = PeerAgentCard(
            agent_id="peer_research_01",
            name="Research Subagent",
            description="Deep web search and paper synthesis subagent",
            capabilities=["web_search", "synthesis", "summarization"],
            max_complexity=TaskComplexityLevel.HIGH,
            workload_score=0.1,
        )

        self.router.register_peer(self.coder_peer)
        self.router.register_peer(self.researcher_peer)

    def test_evaluate_complexity(self):
        evaluator = TaskComplexityEvaluator()

        # Low complexity task
        low_task = TaskSpec(
            title="Simple greeting",
            description="Say hello to the user",
        )
        low_res = evaluator.evaluate(low_task)
        self.assertEqual(low_res.level, TaskComplexityLevel.LOW)

        # High complexity task
        high_task = TaskSpec(
            title="Refactor distributed consensus",
            description="Perform deep refactor and optimization of distributed consensus engine",
            required_capabilities=["python", "refactoring"],
            estimated_tokens=5000,
        )
        high_res = evaluator.evaluate(high_task)
        self.assertIn(high_res.level, (TaskComplexityLevel.HIGH, TaskComplexityLevel.CRITICAL))
        self.assertTrue(high_res.requires_specialized_peer)

    def test_match_peer_by_capabilities(self):
        task = TaskSpec(
            title="Write unit tests for parser",
            description="Implement unit test suite in python",
            required_capabilities=["python", "unit_testing"],
        )

        matched = self.router.match_peer(task)
        self.assertIsNotNone(matched)
        peer, caps = matched
        self.assertEqual(peer.agent_id, "peer_coder_01")
        self.assertIn("python", caps)
        self.assertIn("unit_testing", caps)

    def test_match_peer_no_match(self):
        task = TaskSpec(
            title="Kubernetes Cluster Setup",
            description="Deploy k8s cluster",
            required_capabilities=["kubernetes", "terraform"],
        )

        matched = self.router.match_peer(task)
        self.assertIsNone(matched)

    def test_delegate_task_sync_success(self):
        mock_transport = MagicMock(return_value={"status": "completed", "output": "Refactor finished"})

        task = TaskSpec(
            task_id="task_123",
            title="Refactor core module",
            description="Refactor module logic",
            required_capabilities=["refactoring"],
        )

        result = self.router.delegate_task(task, transport_mock=mock_transport)

        self.assertTrue(result.success)
        self.assertEqual(result.target_agent_id, "peer_coder_01")
        self.assertIsNotNone(result.payload)
        self.assertEqual(result.payload.sender_id, "magda_test")
        self.assertEqual(result.payload.target_agent_id, "peer_coder_01")
        self.assertEqual(result.response_data["status"], "completed")
        mock_transport.assert_called_once()

    def test_delegate_task_no_peer_failure(self):
        task = TaskSpec(
            title="Unknown task",
            required_capabilities=["quantum_computing"],
        )

        result = self.router.delegate_task(task)
        self.assertFalse(result.success)
        self.assertIn("No available peer matches required capabilities", result.error)

    def test_async_delegation(self):
        async def run_async_test():
            async def async_transport(payload_dict):
                await asyncio.sleep(0.01)
                return {"result": f"Done task {payload_dict['task']['task_id']}"}

            task = TaskSpec(
                task_id="task_async_1",
                title="Search literature",
                description="Literature synthesis",
                required_capabilities=["web_search", "synthesis"],
            )

            res = await self.router.delegate_task_async(task, transport_mock=async_transport)
            self.assertTrue(res.success)
            self.assertEqual(res.target_agent_id, "peer_research_01")
            self.assertEqual(res.response_data["result"], "Done task task_async_1")

        asyncio.run(run_async_test())

    def test_workload_preference(self):
        # Register second coder with higher workload
        busy_coder = PeerAgentCard(
            agent_id="peer_coder_busy",
            name="Busy Coder Subagent",
            description="Overworked coder",
            capabilities=["python", "refactoring", "unit_testing"],
            max_complexity=TaskComplexityLevel.HIGH,
            workload_score=0.9,
        )
        self.router.register_peer(busy_coder)

        task = TaskSpec(
            title="Write tests",
            required_capabilities=["python", "unit_testing"],
        )

        # Should prefer peer_coder_01 (workload 0.2) over peer_coder_busy (workload 0.9)
        matched = self.router.match_peer(task)
        self.assertIsNotNone(matched)
        peer, _ = matched
        self.assertEqual(peer.agent_id, "peer_coder_01")


if __name__ == "__main__":
    unittest.main()
