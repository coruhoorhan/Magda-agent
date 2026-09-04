"""
Unit tests for Hermes Agent Scheduled Nightly Backups V3.
"""

import asyncio
import os
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.operations.cron_backups_v3 import (
        BackupRecord,
        EpisodicMemoryCronBackupManagerV3,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "operations"
        / "cron_backups_v3.py"
    )
    spec = importlib.util.spec_from_file_location("cron_backups_v3", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    BackupRecord = module.BackupRecord
    EpisodicMemoryCronBackupManagerV3 = module.EpisodicMemoryCronBackupManagerV3


class TestCronBackupsV3(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_scheduler = MagicMock()
        self.mock_memories = [
            {"id": "mem_1", "content": "User asked about task optimization"},
            {"id": "mem_2", "content": "Agent executed unit tests and verified code"},
        ]
        self.manager = EpisodicMemoryCronBackupManagerV3(
            memory_source=self.mock_memories,
            backup_dir=self.temp_dir,
            scheduler=self.mock_scheduler,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_schedule_nightly_backup(self):
        self.assertFalse(self.manager.is_scheduled())

        job_id = self.manager.schedule_nightly_backup("0 2 * * *")

        self.assertTrue(self.manager.is_scheduled())
        self.assertIsNotNone(job_id)
        self.mock_scheduler.schedule.assert_called_once()
        args = self.mock_scheduler.schedule.call_args
        self.assertEqual(args[0][0], "0 2 * * *")

    def test_perform_backup_and_restore(self):
        record = self.manager.perform_backup()

        self.assertEqual(record.status, "success")
        self.assertEqual(record.entries_count, 2)
        self.assertTrue(os.path.exists(record.backup_file))
        self.assertGreater(record.size_bytes, 0)

        # Restore
        restored = self.manager.restore_backup(record.backup_file)
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0]["id"], "mem_1")
        self.assertEqual(restored[1]["id"], "mem_2")

    def test_async_memory_source_backup(self):
        async def run_async():
            mock_source = MagicMock()
            mock_source.get_all = AsyncMock(return_value=[
                {"id": "async_mem_1", "content": "Async context chunk"},
            ])

            mgr = EpisodicMemoryCronBackupManagerV3(
                memory_source=mock_source,
                backup_dir=self.temp_dir,
            )

            rec = await mgr.perform_backup_async()
            self.assertEqual(rec.status, "success")
            self.assertEqual(rec.entries_count, 1)

            restored = mgr.restore_backup(rec.backup_file)
            self.assertEqual(restored[0]["id"], "async_mem_1")

        asyncio.run(run_async())


if __name__ == "__main__":
    unittest.main()
