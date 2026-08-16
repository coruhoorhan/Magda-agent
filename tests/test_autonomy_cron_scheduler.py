import asyncio
import pytest
from unittest.mock import MagicMock
from magda_agent.autonomy.cron_scheduler import CronScheduler
from apscheduler.job import Job

@pytest.fixture
def scheduler() -> CronScheduler:
    """Fixture to provide a fresh CronScheduler instance."""
    return CronScheduler()

@pytest.mark.asyncio
async def test_scheduler_start_stop(scheduler: CronScheduler) -> None:
    """Test that the scheduler can start and stop gracefully."""
    assert not scheduler.is_running
    scheduler.start()
    assert scheduler.is_running

    # Starting again should not raise errors
    scheduler.start()
    assert scheduler.is_running

    scheduler.stop()
    assert not scheduler.is_running

    # Stopping again should not raise errors
    scheduler.stop()
    assert not scheduler.is_running

@pytest.mark.asyncio
async def test_add_task(scheduler: CronScheduler) -> None:
    """Test adding a task to the scheduler."""
    mock_task = MagicMock()

    scheduler.add_task(
        func=mock_task,
        cron_expression="0 0 * * *",
        job_id="test_job_1",
        kwargs={"param": "value"}
    )

    job = scheduler.scheduler.get_job("test_job_1")
    assert job is not None
    assert job.id == "test_job_1"
    assert job.kwargs == {"param": "value"}
    assert job.func == mock_task

@pytest.mark.asyncio
async def test_invalid_cron_expression(scheduler: CronScheduler) -> None:
    """Test that adding a task with an invalid cron expression raises a ValueError."""
    mock_task = MagicMock()

    with pytest.raises(ValueError):
        scheduler.add_task(
            func=mock_task,
            cron_expression="invalid cron expression",
            job_id="test_job_invalid"
        )
