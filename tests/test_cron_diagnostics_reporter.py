import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from magda_agent.operations.cron_diagnostics_reporter import CronDiagnosticsReporter
from magda_agent.operations.cron_v3 import HermesCronSchedulerV3


def test_cron_diagnostics_reporter_init():
    """Test the initialization of the reporter with and without a scheduler."""
    reporter1 = CronDiagnosticsReporter()
    assert isinstance(reporter1.scheduler, HermesCronSchedulerV3)

    mock_scheduler = MagicMock(spec=HermesCronSchedulerV3)
    reporter2 = CronDiagnosticsReporter(scheduler=mock_scheduler)
    assert reporter2.scheduler == mock_scheduler


def test_gather_metrics():
    """Test that metrics are gathered and return a dictionary with the expected keys."""
    reporter = CronDiagnosticsReporter()
    metrics = reporter.gather_metrics()

    assert isinstance(metrics, dict)
    assert "error_density" in metrics
    assert "system_health" in metrics
    assert "memory_usage" in metrics
    assert "uptime_seconds" in metrics
    assert "active_tasks" in metrics


def test_schedule_report():
    """Test that scheduling the report correctly calls the underlying scheduler."""
    mock_scheduler = MagicMock(spec=HermesCronSchedulerV3)
    reporter = CronDiagnosticsReporter(scheduler=mock_scheduler)

    reporter.schedule_report(cron_expr="0 * * * *")

    mock_scheduler.schedule.assert_called_once_with(
        cron_expr="0 * * * *",
        func=reporter.generate_report,
        name="cron_diagnostics_reporter"
    )


def test_generate_report_success():
    """Test generating a successful report containing gathered metrics."""
    reporter = CronDiagnosticsReporter()

    # Mock gather_metrics to return deterministic data
    mock_metrics = {
        "error_density": 0.02,
        "system_health": "excellent",
        "memory_usage": "100MB",
        "uptime_seconds": 3600,
        "active_tasks": 5
    }

    with patch.object(reporter, 'gather_metrics', return_value=mock_metrics):
        # We use asyncio.run to execute the async method in a synchronous test
        report = asyncio.run(reporter.generate_report())

        assert "Diagnostic Report" in report
        assert "System Health: excellent" in report
        assert "Error Density: 0.02" in report
        assert "Memory Usage: 100MB" in report
        assert "Uptime: 3600 seconds" in report
        assert "Active Tasks: 5" in report


def test_generate_report_exception():
    """Test that exceptions during report generation are handled gracefully."""
    reporter = CronDiagnosticsReporter()

    # Force gather_metrics to raise an exception
    with patch.object(reporter, 'gather_metrics', side_effect=Exception("Simulated failure")):
        report = asyncio.run(reporter.generate_report())

        # It should catch the exception and return an empty string
        assert report == ""
