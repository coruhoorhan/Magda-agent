"""
Unit tests for DynamicSkillOptimizerV2 (Hermes-inspired Dynamic Skill Optimizer V2).
"""

import asyncio
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.skills.dynamic_optimizer_v2 import (
        DynamicSkillOptimizerV2,
        OptimizedSkill,
        SkillExecutionRecord,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "skills"
        / "dynamic_optimizer_v2.py"
    )
    spec = importlib.util.spec_from_file_location("dynamic_optimizer_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    DynamicSkillOptimizerV2 = module.DynamicSkillOptimizerV2
    OptimizedSkill = module.OptimizedSkill
    SkillExecutionRecord = module.SkillExecutionRecord


class MockSkillRegistry:
    def __init__(self):
        self.skills = {}

    def register_skill(self, name, func, description):
        self.skills[name] = {"func": func, "description": description}


class TestDynamicSkillOptimizerV2(unittest.TestCase):
    def setUp(self):
        self.registry = MockSkillRegistry()
        self.optimizer = DynamicSkillOptimizerV2(
            consecutive_success_threshold=3,
            skill_registry=self.registry,
        )

    def test_record_execution_consecutive_success(self):
        # Initial state
        self.assertEqual(self.optimizer.get_consecutive_successes("fetch_weather"), 0)
        self.assertFalse(self.optimizer.is_optimized("fetch_weather"))

        # Success 1
        opt = self.optimizer.record_execution(
            skill_name="fetch_weather",
            input_args={"city": "London"},
            output={"temp": 18},
            success=True,
            latency_ms=120.0,
            token_usage=45,
        )
        self.assertEqual(self.optimizer.get_consecutive_successes("fetch_weather"), 1)
        self.assertIsNone(opt)
        self.assertFalse(self.optimizer.is_optimized("fetch_weather"))

        # Success 2
        opt = self.optimizer.record_execution(
            skill_name="fetch_weather",
            input_args={"city": "Paris"},
            output={"temp": 22},
            success=True,
            latency_ms=110.0,
            token_usage=40,
        )
        self.assertEqual(self.optimizer.get_consecutive_successes("fetch_weather"), 2)
        self.assertIsNone(opt)

        # Success 3 -> Threshold reached, triggers optimization
        opt = self.optimizer.record_execution(
            skill_name="fetch_weather",
            input_args={"city": "Berlin"},
            output={"temp": 20},
            success=True,
            latency_ms=115.0,
            token_usage=42,
        )
        self.assertIsNotNone(opt)
        self.assertTrue(self.optimizer.is_optimized("fetch_weather"))
        self.assertIn("fetch_weather", self.registry.skills)

    def test_failure_resets_consecutive_successes(self):
        self.optimizer.record_execution(
            skill_name="calc_sum",
            input_args={"a": 1, "b": 2},
            output=3,
            success=True,
        )
        self.assertEqual(self.optimizer.get_consecutive_successes("calc_sum"), 1)

        # Failure occurs
        self.optimizer.record_execution(
            skill_name="calc_sum",
            input_args={"a": "bad", "b": 2},
            output=None,
            success=False,
            error="TypeError",
        )
        self.assertEqual(self.optimizer.get_consecutive_successes("calc_sum"), 0)
        self.assertFalse(self.optimizer.is_optimized("calc_sum"))

    def test_execute_skill_mock_caching_and_savings(self):
        mock_func = MagicMock(side_effect=lambda query: f"Result for {query}")
        self.optimizer.register_original_skill("search_db", mock_func)

        # Execute 3 times to trigger optimization
        for i in range(3):
            res = self.optimizer.execute_skill("search_db", query=f"q_{i}")
            self.assertEqual(res, f"Result for q_{i}")

        self.assertTrue(self.optimizer.is_optimized("search_db"))
        self.assertEqual(mock_func.call_count, 3)

        # Now execute with cached query
        res1 = self.optimizer.execute_skill("search_db", query="q_100")
        self.assertEqual(res1, "Result for q_100")
        self.assertEqual(mock_func.call_count, 4)

        # Second call with same args should hit cache
        res2 = self.optimizer.execute_skill("search_db", query="q_100")
        self.assertEqual(res2, "Result for q_100")
        self.assertEqual(mock_func.call_count, 4)  # No extra call!

        stats = self.optimizer.get_optimization_stats("search_db")
        self.assertTrue(stats["is_optimized"])
        self.assertEqual(stats["hit_count"], 1)
        self.assertEqual(stats["miss_count"], 1)

    def test_async_skill_execution(self):
        async def run_test():
            async def mock_async_skill(x: int) -> int:
                await asyncio.sleep(0.01)
                return x * 2

            self.optimizer.register_original_skill("double_num", mock_async_skill)

            for i in range(3):
                r = await self.optimizer.execute_skill_async("double_num", x=i)
                self.assertEqual(r, i * 2)

            self.assertTrue(self.optimizer.is_optimized("double_num"))

            # Hit cache
            r1 = await self.optimizer.execute_skill_async("double_num", x=10)
            self.assertEqual(r1, 20)
            r2 = await self.optimizer.execute_skill_async("double_num", x=10)
            self.assertEqual(r2, 20)

            stats = self.optimizer.get_optimization_stats("double_num")
            self.assertEqual(stats["hit_count"], 1)

        asyncio.run(run_test())

    def test_clear_cache(self):
        mock_func = MagicMock(return_value="constant_val")
        self.optimizer.register_original_skill("const_skill", mock_func)

        for _ in range(3):
            self.optimizer.execute_skill("const_skill", param="a")

        self.optimizer.execute_skill("const_skill", param="b")
        opt = self.optimizer.get_optimized_skill("const_skill")
        self.assertIn('{"param": "b"}', opt.cache_store)

        self.optimizer.clear_cache("const_skill")
        self.assertEqual(len(opt.cache_store), 0)


if __name__ == "__main__":
    unittest.main()
