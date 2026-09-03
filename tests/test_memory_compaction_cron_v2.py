"""
Unit tests for OpenClaw Memory Compaction Cron V2.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.scheduler.memory_compaction_cron_v2 import (
        CompactionJobResult,
        OpenClawMemoryCompactionCronV2,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "scheduler"
        / "memory_compaction_cron_v2.py"
    )
    spec = importlib.util.spec_from_file_location("memory_compaction_cron_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    CompactionJobResult = module.CompactionJobResult
    OpenClawMemoryCompactionCronV2 = module.OpenClawMemoryCompactionCronV2


class MockMemoryEngine:
    def __init__(self, entries):
        self.entries = list(entries)

    def get_all(self):
        return list(self.entries)

    def replace_all(self, new_entries):
        self.entries = list(new_entries)


class TestMemoryCompactionCronV2(unittest.TestCase):
    def setUp(self):
        self.mock_scheduler = MagicMock()
        self.raw_entries = [
            {"id": f"e_{i}", "content": f"Episodic memory dialogue content item {i}", "tokens": 50}
            for i in range(20)
        ]
        self.mock_engine = MockMemoryEngine(self.raw_entries)
        self.cron = OpenClawMemoryCompactionCronV2(
            memory_engine=self.mock_engine,
            scheduler=self.mock_scheduler,
            cron_expression="0 4 * * *",
            compaction_threshold_tokens=500,
        )

    def test_schedule_compaction_job(self):
        job_id = self.cron.schedule_compaction_job("0 3 * * *")

        self.assertTrue(self.cron.is_scheduled())
        self.mock_scheduler.add_task.assert_called_once()
        args = self.mock_scheduler.add_task.call_args[0]
        self.assertEqual(args[1], "0 3 * * *")

    def test_run_compaction_step_success(self):
        result = self.cron.run_compaction_step(force=True)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.initial_entries_count, 20)
        self.assertEqual(result.initial_tokens, 1000)
        self.assertGreater(result.tokens_freed, 0)
        self.assertLess(result.final_tokens, result.initial_tokens)
        self.assertEqual(len(self.mock_engine.entries), 1)

    def test_skip_when_below_threshold_and_not_forced(self):
        small_engine = MockMemoryEngine([{"id": "1", "content": "short note", "tokens": 10}])
        cron_high_thresh = OpenClawMemoryCompactionCronV2(
            memory_engine=small_engine,
            compaction_threshold_tokens=2000,
        )

        res = cron_high_thresh.run_compaction_step(force=False)
        self.assertEqual(res.status, "skipped")
        self.assertEqual(res.tokens_freed, 0)
        self.assertEqual(len(small_engine.entries), 1)

    def test_async_compaction_execution(self):
        async def run_async():
            mock_compressor = MagicMock()
            mock_comp_result = MagicMock()
            mock_comp_result.compressed_fact_count = 3
            mock_comp_result.compressed_token_count = 150
            mock_comp_result.tokens_freed = 850
            mock_compressor.compress_episodic_to_semantic = AsyncMock(return_value=mock_comp_result)

            cron_with_comp = OpenClawMemoryCompactionCronV2(
                memory_engine=self.mock_engine,
                compressor=mock_compressor,
            )

            res = await cron_with_comp.run_compaction_step_async(force=True)
            self.assertEqual(res.status, "success")
            self.assertEqual(res.tokens_freed, 850)
            self.assertEqual(len(cron_with_comp.get_history()), 1)

        asyncio.run(run_async())


if __name__ == "__main__":
    unittest.main()
