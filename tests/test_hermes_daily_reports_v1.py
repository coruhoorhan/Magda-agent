import pytest
import asyncio
from magda_agent.scheduler.cron_scheduler_v2 import CronSchedulerV2
from magda_agent.scheduler.hermes_daily_reports_v1 import HermesDailyReporterV1

@pytest.mark.asyncio
async def test_hermes_daily_reporter():
    """Test the daily reporter activity recording and report generation."""
    scheduler = CronSchedulerV2()
    reporter = HermesDailyReporterV1(scheduler)

    # Initially empty report
    empty_report = reporter.generate_report()
    assert "No activities recorded today." in empty_report

    # Record some activities
    reporter.record_activity("tool_call", "Executed web search.")
    reporter.record_activity("user_reply", "Answered user query.")

    # Generate populated report
    populated_report = reporter.generate_report()
    assert "Executed web search." in populated_report
    assert "Answered user query." in populated_report
    assert "tool_call" in populated_report

    # Ensure list clears after generating
    assert len(reporter.activities) == 0

@pytest.mark.asyncio
async def test_hermes_daily_reporter_cron_registration():
    """Test that the cron task is properly registered on the scheduler."""
    scheduler = CronSchedulerV2()
    reporter = HermesDailyReporterV1(scheduler)

    reporter.register_cron(cron_expr="0 0 * * *")

    assert "hermes_daily_report" in scheduler.tasks
    task_info = scheduler.tasks["hermes_daily_report"]
    assert task_info["cron_expr"] == "0 0 * * *"

    # Test the broadcast task without spinning up the actual cron loop
    await task_info["func"]()
