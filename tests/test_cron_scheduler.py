import pytest
import tempfile
import json
import yaml
from pathlib import Path
from typing import Any

from magda_agent.scheduler.cron_v5 import CronScheduler

@pytest.fixture
def scheduler():
    return CronScheduler()

@pytest.fixture
def mock_registry():
    async def dummy_backup(*args: Any, **kwargs: Any) -> str:
        return "backup_done"

    async def dummy_report(*args: Any, **kwargs: Any) -> str:
        return "report_done"

    return {
        "run_backup": dummy_backup,
        "run_report": dummy_report
    }

def test_load_config_json_success(scheduler, mock_registry, tmp_path):
    config_data = [
        {"name": "nightly_backup", "cron": "0 0 * * *", "action": "run_backup"},
        {"name": "daily_report", "cron": "0 9 * * *", "action": "run_report", "args": ["system"], "kwargs": {"verbose": True}}
    ]

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    scheduler.load_config(str(config_file), mock_registry)

    assert len(scheduler.jobs) == 2
    assert scheduler.jobs[0]["name"] == "nightly_backup"
    assert scheduler.jobs[0]["cron_expr"] == "0 0 * * *"
    assert scheduler.jobs[0]["func"] == mock_registry["run_backup"]

    assert scheduler.jobs[1]["name"] == "daily_report"
    assert scheduler.jobs[1]["cron_expr"] == "0 9 * * *"
    assert scheduler.jobs[1]["func"] == mock_registry["run_report"]
    assert scheduler.jobs[1]["args"] == ("system",)
    assert scheduler.jobs[1]["kwargs"] == {"verbose": True}

def test_load_config_yaml_success(scheduler, mock_registry, tmp_path):
    config_data = {
        "jobs": [
            {"name": "yaml_backup", "cron": "30 2 * * *", "action": "run_backup"}
        ]
    }

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))

    scheduler.load_config(str(config_file), mock_registry)

    assert len(scheduler.jobs) == 1
    assert scheduler.jobs[0]["name"] == "yaml_backup"
    assert scheduler.jobs[0]["cron_expr"] == "30 2 * * *"

def test_load_config_invalid_format(scheduler, mock_registry, tmp_path):
    config_file = tmp_path / "config.txt"
    config_file.write_text("some text")

    with pytest.raises(ValueError, match="Unsupported configuration file format: .txt"):
        scheduler.load_config(str(config_file), mock_registry)

def test_load_config_missing_action(scheduler, mock_registry, tmp_path):
    config_data = [
        {"name": "missing_action", "cron": "0 0 * * *", "action": "unknown_action"}
    ]

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    with pytest.raises(ValueError, match="Action 'unknown_action' for job 'missing_action' not found in function registry."):
        scheduler.load_config(str(config_file), mock_registry)

def test_load_config_file_not_found(scheduler, mock_registry):
    with pytest.raises(FileNotFoundError, match="Configuration file not found: non_existent.json"):
        scheduler.load_config("non_existent.json", mock_registry)

def test_load_config_skips_invalid_jobs(scheduler, mock_registry, tmp_path, caplog):
    config_data = [
        {"name": "valid_job", "cron": "0 0 * * *", "action": "run_backup"},
        {"name": "invalid_job", "cron": "0 0 * * *"} # Missing action
    ]

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    scheduler.load_config(str(config_file), mock_registry)

    assert len(scheduler.jobs) == 1
    assert scheduler.jobs[0]["name"] == "valid_job"
    assert "Skipping invalid job configuration" in caplog.text
