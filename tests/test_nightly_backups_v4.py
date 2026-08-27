import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from magda_agent.operations.nightly_backups_v4 import NightlyBackupManagerV4
from magda_agent.scheduler.cron_scheduler_v2 import CronSchedulerV2


@pytest.fixture
def mock_scheduler() -> MagicMock:
    """Fixture for mocking the CronSchedulerV2."""
    scheduler = MagicMock(spec=CronSchedulerV2)
    return scheduler


def test_schedule_backup(mock_scheduler: MagicMock) -> None:
    """Test that schedule_backup correctly adds the task to the scheduler."""
    manager = NightlyBackupManagerV4(scheduler=mock_scheduler)
    manager.schedule_backup()

    mock_scheduler.add_task.assert_called_once_with(
        name="nightly_backup_v4",
        cron_expr="0 2 * * *",
        func=manager.perform_backup
    )


@pytest.mark.asyncio
async def test_perform_backup(mock_scheduler: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    """Test that perform_backup logs successfully."""
    manager = NightlyBackupManagerV4(scheduler=mock_scheduler)

    with caplog.at_level("INFO"):
        await manager.perform_backup()

    assert "Starting nightly backup v4..." in caplog.text
    assert "Completed nightly backup v4 successfully." in caplog.text
