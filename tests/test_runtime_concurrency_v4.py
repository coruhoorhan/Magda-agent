"""
Unit tests for OpenAI SDK Runtime Function Tool Concurrency V1 (V4 Engine).
"""

import asyncio
import time
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.skills.runtime_concurrency_v4 import (
        FunctionToolCallResult,
        FunctionToolCallSpec,
        OpenAIRuntimeFunctionConcurrencyV4,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "skills"
        / "runtime_concurrency_v4.py"
    )
    spec = importlib.util.spec_from_file_location("runtime_concurrency_v4", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    FunctionToolCallResult = module.FunctionToolCallResult
    FunctionToolCallSpec = module.FunctionToolCallSpec
    OpenAIRuntimeFunctionConcurrencyV4 = module.OpenAIRuntimeFunctionConcurrencyV4


class TestRuntimeConcurrencyV4(unittest.TestCase):
    def setUp(self):
        self.engine = OpenAIRuntimeFunctionConcurrencyV4(max_concurrency=8)

    def test_parallel_execution_speedup_and_order_preservation(self):
        async def run_async():
            async def delayed_task(item_id: int, delay_s: float = 0.05) -> str:
                await asyncio.sleep(delay_s)
                return f"Result_{item_id}"

            self.engine.register_function("delayed_task", delayed_task)

            calls = [
                FunctionToolCallSpec(function_name="delayed_task", arguments={"item_id": i, "delay_s": 0.04}, call_id=f"c_{i}")
                for i in range(5)
            ]

            t0 = time.perf_counter()
            results = await self.engine.execute_concurrent_tools(calls)
            elapsed = time.perf_counter() - t0

            # 5 * 0.04s sequential would be ~0.20s; parallel should take < 0.12s
            self.assertLess(elapsed, 0.15)

            # Order preservation check
            self.assertEqual(len(results), 5)
            for i, res in enumerate(results):
                self.assertEqual(res.call_id, f"c_{i}")
                self.assertTrue(res.success)
                self.assertEqual(res.result, f"Result_{i}")

        asyncio.run(run_async())

    def test_timeout_handling(self):
        async def run_async():
            async def slow_func():
                await asyncio.sleep(0.5)
                return "finished"

            async def fast_func():
                return "quick"

            self.engine.register_function("slow_func", slow_func)
            self.engine.register_function("fast_func", fast_func)

            calls = [
                FunctionToolCallSpec(function_name="slow_func", timeout_seconds=0.02),
                FunctionToolCallSpec(function_name="fast_func", timeout_seconds=1.0),
            ]

            results = await self.engine.execute_concurrent_tools(calls)

            self.assertEqual(len(results), 2)
            # 1st call timed out
            self.assertFalse(results[0].success)
            self.assertTrue(results[0].timed_out)
            self.assertIn("exceeded timeout", results[0].error)

            # 2nd call succeeded
            self.assertTrue(results[1].success)
            self.assertEqual(results[1].result, "quick")

        asyncio.run(run_async())

    def test_partial_failure_isolation(self):
        async def run_async():
            def failing_tool():
                raise ValueError("Database connection failed")

            def healthy_tool():
                return 42

            self.engine.register_function("failing_tool", failing_tool)
            self.engine.register_function("healthy_tool", healthy_tool)

            calls = [
                {"name": "failing_tool"},
                {"name": "healthy_tool"},
            ]

            results = await self.engine.execute_concurrent_tools(calls)

            self.assertEqual(len(results), 2)
            self.assertFalse(results[0].success)
            self.assertIn("Database connection failed", results[0].error)

            self.assertTrue(results[1].success)
            self.assertEqual(results[1].result, 42)

        asyncio.run(run_async())

    def test_sync_execution_wrapper(self):
        self.engine.register_function("add", lambda a, b: a + b)
        results = self.engine.execute_concurrent_tools_sync([
            {"function_name": "add", "arguments": {"a": 10, "b": 20}}
        ])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].result, 30)


if __name__ == "__main__":
    unittest.main()
