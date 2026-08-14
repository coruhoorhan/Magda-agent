import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, patch

from magda_agent.scheduler.cron_tasks import CronTaskManager

@pytest.fixture
def task_manager():
    return CronTaskManager()

@pytest.mark.asyncio
async def test_register_and_get_tasks(task_manager):
    async def sample_func():
        return "ok"

    task_manager.register_task("sample_task", "0 0 * * *", sample_func)
    tasks = task_manager.get_tasks()

    assert len(tasks) == 1
    assert tasks[0]["name"] == "sample_task"
    assert tasks[0]["cron_expr"] == "0 0 * * *"
    assert isinstance(tasks[0]["next_run"], datetime)

@pytest.mark.asyncio
async def test_remove_task(task_manager):
    async def sample_func():
        return "ok"

    task_manager.register_task("task1", "0 0 * * *", sample_func)
    task_manager.register_task("task2", "0 1 * * *", sample_func)

    assert len(task_manager.get_tasks()) == 2
    removed = task_manager.remove_task("task1")
    assert removed is True
    assert len(task_manager.get_tasks()) == 1
    assert task_manager.get_tasks()[0]["name"] == "task2"

    removed_again = task_manager.remove_task("non_existent")
    assert removed_again is False

@pytest.mark.asyncio
async def test_tick_execution_without_real_delays(task_manager):
    mock_callback = AsyncMock()
    task_manager.scheduler.result_callback = mock_callback

    mock_func = AsyncMock(return_value="task_result")
    task_manager.register_task("test_job", "* * * * *", mock_func)

    now = datetime(2026, 6, 1, 12, 0, 0)
    with patch.object(task_manager.scheduler, '_get_now', return_value=now):
        # Re-register to compute next_run relative to patched now
        task_manager.scheduler.jobs = []
        task_manager.register_task("test_job", "* * * * *", mock_func)

        job = task_manager.scheduler.jobs[0]
        assert job["next_run"] == datetime(2026, 6, 1, 12, 1, 0)

        # Tick prior to scheduled time
        res_before = await task_manager.tick(current_time=datetime(2026, 6, 1, 12, 0, 30))
        assert len(res_before) == 0
        mock_func.assert_not_called()

        # Tick at or after scheduled time
        res_due = await task_manager.tick(current_time=datetime(2026, 6, 1, 12, 1, 0))
        assert len(res_due) == 1
        assert res_due[0]["name"] == "test_job"
        assert res_due[0]["result"] == "task_result"

        mock_func.assert_called_once()
        mock_callback.assert_called_once_with("task_result")

@pytest.mark.asyncio
async def test_register_log_cleanup(tmp_path):
    manager = CronTaskManager()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    old_log = log_dir / "old.log"
    new_log = log_dir / "new.log"
    txt_file = log_dir / "old.txt"

    old_log.write_text("old content")
    new_log.write_text("new content")
    txt_file.write_text("txt content")

    # Set mtime of old.log to 10 days ago
    ten_days_ago = time.time() - (10 * 86400)
    import os
    os.utime(str(old_log), (ten_days_ago, ten_days_ago))

    manager.register_log_cleanup("cleanup_job", log_dir=log_dir, max_age_days=7, cron_expr="* * * * *")

    job = manager.scheduler.jobs[0]
    result = await job["func"]()

    assert result["status"] == "success"
    assert result["cleaned_count"] == 1
    assert not old_log.exists()
    assert new_log.exists()
    assert txt_file.exists()

@pytest.mark.asyncio
async def test_register_daily_summary():
    manager = CronTaskManager()

    async def mock_summary_gen():
        return "Daily summary report text"

    manager.register_daily_summary("summary_job", mock_summary_gen)

    job = manager.scheduler.jobs[0]
    result = await job["func"]()

    assert result["status"] == "success"
    assert result["summary"] == "Daily summary report text"
    assert "timestamp" in result

@pytest.mark.asyncio
async def test_register_status_report():
    manager = CronTaskManager()

    async def mock_report_gen():
        return {"active_users": 10, "cpu_load": 0.25}

    manager.register_status_report("status_job", mock_report_gen)

    job = manager.scheduler.jobs[0]
    result = await job["func"]()

    assert result["status"] == "success"
    assert result["report"] == {"active_users": 10, "cpu_load": 0.25}

@pytest.mark.asyncio
async def test_start_and_stop(task_manager):
    await task_manager.start()
    assert task_manager.scheduler._running is True

    await task_manager.stop()
    assert task_manager.scheduler._running is False
