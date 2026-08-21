import asyncio
from datetime import datetime, timedelta
import pytest
from unittest.mock import patch, MagicMock
from magda_agent.scheduler.cron_scheduler_v2 import CronSchedulerV2

@pytest.fixture
def scheduler():
    """Returns a new instance of CronSchedulerV2."""
    return CronSchedulerV2()

@pytest.mark.asyncio
async def test_add_task_invalid_cron(scheduler):
    """Test adding a task with an invalid cron expression."""
    async def dummy_task():
        pass
    with pytest.raises(ValueError, match="Invalid cron expression"):
        scheduler.add_task("test_task", "invalid_cron", dummy_task)

@pytest.mark.asyncio
async def test_add_task_valid_cron(scheduler):
    """Test adding a task with a valid cron expression."""
    async def dummy_task():
        pass
    scheduler.add_task("test_task", "* * * * *", dummy_task)
    assert "test_task" in scheduler.tasks
    assert scheduler.tasks["test_task"]["cron_expr"] == "* * * * *"

@pytest.mark.asyncio
async def test_scheduler_start_stop(scheduler):
    """Test starting and stopping the scheduler loop."""
    assert not scheduler._running

    # Need to mock the loop behavior to not get stuck
    with patch.object(scheduler, '_loop', new_callable=MagicMock) as mock_loop:
        async def dummy_loop():
            pass
        mock_loop.side_effect = dummy_loop

        scheduler.start()
        assert scheduler._running
        assert scheduler._task is not None

        await scheduler.stop()
        assert not scheduler._running

@pytest.mark.asyncio
async def test_scheduler_executes_task_on_schedule(scheduler):
    """Test that the scheduler triggers a task when its schedule arrives."""
    task_executed = asyncio.Event()

    async def dummy_task():
        task_executed.set()

    scheduler.add_task("test_task", "* * * * *", dummy_task)

    past_time = datetime.now() - timedelta(minutes=1)
    scheduler.tasks["test_task"]["next_run"] = past_time

    with patch('magda_agent.scheduler.cron_scheduler_v2.croniter.get_next') as mock_get_next:
        mock_get_next.return_value = datetime.now() + timedelta(minutes=60)

        # Stop loop after one iteration
        with patch('magda_agent.scheduler.cron_scheduler_v2.asyncio.sleep') as mock_sleep:
            async def stop_loop(*args, **kwargs):
                scheduler._running = False
            mock_sleep.side_effect = stop_loop

            scheduler._running = True
            await scheduler._loop()

        assert task_executed.is_set()

@pytest.mark.asyncio
async def test_add_task_wakes_loop(scheduler):
    """Test that adding a new task wakes up the sleeping loop."""
    sleep_task = asyncio.create_task(asyncio.sleep(100))
    scheduler._sleep_task = sleep_task
    scheduler._running = True

    async def dummy_task():
        pass

    scheduler.add_task("test_task", "* * * * *", dummy_task)

    # cancel() schedules the cancellation but doesn't complete it instantly
    # yield to the event loop so the cancellation is processed
    await asyncio.sleep(0)

    assert scheduler._sleep_task.cancelled() or scheduler._sleep_task.done()

    scheduler._running = False
