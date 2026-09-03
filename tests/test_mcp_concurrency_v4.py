"""
Tests for MCP Action Tool Concurrency v4.
"""

import asyncio
import time
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.skills.mcp_concurrency_v4 import (
        MCPActionToolConcurrencyV4,
        MCPToolCallSpec,
        MCPToolExecutionResult,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "skills" / "mcp_concurrency_v4.py"
    spec = importlib.util.spec_from_file_location("mcp_concurrency_v4", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    MCPActionToolConcurrencyV4 = module.MCPActionToolConcurrencyV4
    MCPToolCallSpec = module.MCPToolCallSpec
    MCPToolExecutionResult = module.MCPToolExecutionResult


class TestMCPConcurrencyV4(unittest.IsolatedAsyncioTestCase):
    """
    Test suite verifying concurrent MCP action tool execution,
    order preservation, timeout enforcement, and server rate-limiting.
    """

    def setUp(self):
        self.executor = MCPActionToolConcurrencyV4(
            max_global_concurrency=10,
            max_per_server_concurrency=3,
        )

    # -------------------------------------------------------------------------
    # 1. Order Preservation & Concurrency
    # -------------------------------------------------------------------------
    async def test_concurrent_execution_order_preserved(self):
        """Batch results must strictly match the original input call order."""
        async def fast_tool(idx: int) -> str:
            await asyncio.sleep(0.01)
            return f"Result {idx}"

        async def slow_tool(idx: int) -> str:
            await asyncio.sleep(0.03)
            return f"Result {idx}"

        calls = [
            MCPToolCallSpec(name="slow_0", arguments={"idx": 0}, tool_func=slow_tool),
            MCPToolCallSpec(name="fast_1", arguments={"idx": 1}, tool_func=fast_tool),
            MCPToolCallSpec(name="fast_2", arguments={"idx": 2}, tool_func=fast_tool),
            MCPToolCallSpec(name="slow_3", arguments={"idx": 3}, tool_func=slow_tool),
        ]

        results = await self.executor.execute_action_tools_concurrently(calls)

        self.assertEqual(len(results), 4)
        for i, res in enumerate(results):
            self.assertTrue(res.success)
            self.assertEqual(res.result, f"Result {i}")

    async def test_concurrent_execution_speedup(self):
        """Concurrent execution of 4 sleep(0.05s) calls should take much less than sequential time."""
        async def sleep_tool(val: int) -> int:
            await asyncio.sleep(0.05)
            return val

        calls = [
            {"name": f"server1_tool_{i}", "args": {"val": i}, "tool_func": sleep_tool}
            for i in range(4)
        ]

        start = time.perf_counter()
        results = await self.executor.execute_action_tools_concurrently(calls)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(results), 4)
        # 4 * 0.05s = 0.20s sequentially. Concurrently should be < 0.15s
        self.assertLess(elapsed, 0.15)
        self.assertEqual([r.result for r in results], [0, 1, 2, 3])

    # -------------------------------------------------------------------------
    # 2. Timeout & Partial Failure Tolerance
    # -------------------------------------------------------------------------
    async def test_tool_timeout_handling(self):
        """Tools exceeding timeout_seconds should fail with timed_out=True without failing the batch."""
        async def hanging_tool():
            await asyncio.sleep(1.0)
            return "Should not finish"

        async def ok_tool():
            await asyncio.sleep(0.01)
            return "OK"

        calls = [
            MCPToolCallSpec(name="hang", timeout_seconds=0.05, tool_func=hanging_tool),
            MCPToolCallSpec(name="quick", timeout_seconds=1.0, tool_func=ok_tool),
        ]

        results = await self.executor.execute_action_tools_concurrently(calls)

        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].success)
        self.assertTrue(results[0].timed_out)
        self.assertIn("timed out", results[0].error)

        self.assertTrue(results[1].success)
        self.assertEqual(results[1].result, "OK")

    async def test_partial_failure_tolerance(self):
        """Exceptions in individual tools must be isolated and captured in results."""
        def faulty_tool():
            raise ValueError("Invalid parameters passed to action tool")

        def good_tool():
            return "Success"

        calls = [
            {"name": "bad", "tool_func": faulty_tool},
            {"name": "good", "tool_func": good_tool},
        ]

        results = await self.executor.execute_action_tools_concurrently(calls)

        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].success)
        self.assertIn("Invalid parameters", results[0].error)

        self.assertTrue(results[1].success)
        self.assertEqual(results[1].result, "Success")

    # -------------------------------------------------------------------------
    # 3. Tool Resolver & Sync Wrapper
    # -------------------------------------------------------------------------
    async def test_custom_tool_resolver(self):
        """Tool resolver callback should dynamically map tool names to callable functions."""
        registry = {
            "github_create_issue": lambda title: f"Created issue: {title}",
            "slack_send_msg": lambda msg: f"Sent message: {msg}",
        }

        executor = MCPActionToolConcurrencyV4(tool_resolver=lambda name: registry.get(name))

        calls = [
            {"name": "github_create_issue", "args": {"title": "Fix bug #123"}},
            {"name": "slack_send_msg", "args": {"msg": "Deployment finished"}},
        ]

        results = await executor.execute_action_tools_concurrently(calls)
        self.assertEqual(results[0].result, "Created issue: Fix bug #123")
        self.assertEqual(results[1].result, "Sent message: Deployment finished")

    def test_sync_execution_wrapper(self):
        """execute_sync should run concurrent tool batches synchronously."""
        def sync_worker(val: int) -> int:
            return val * 10

        calls = [
            {"name": "calc", "args": {"val": 5}, "tool_func": sync_worker}
        ]

        results = self.executor.execute_sync(calls)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].result, 50)


if __name__ == "__main__":
    unittest.main()
