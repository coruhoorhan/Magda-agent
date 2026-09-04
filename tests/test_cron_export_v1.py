"""
Unit tests for Hermes Agent Scheduled Tasks Export Integration V1.
"""

import json
import unittest

try:
    from magda_agent.operations.cron_export_v1 import (
        ExportFormat,
        HermesCronTasksExporterV1,
        ScheduledTaskDefinition,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "operations"
        / "cron_export_v1.py"
    )
    spec = importlib.util.spec_from_file_location("cron_export_v1", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ExportFormat = module.ExportFormat
    HermesCronTasksExporterV1 = module.HermesCronTasksExporterV1
    ScheduledTaskDefinition = module.ScheduledTaskDefinition


class TestCronExportV1(unittest.TestCase):
    def setUp(self):
        self.exporter = HermesCronTasksExporterV1(platform_name="test_magda_ops")

        self.backup_task = ScheduledTaskDefinition(
            task_id="nightly_backup_job",
            name="Nightly Episodic Backup",
            schedule="0 3 * * *",
            command="python3",
            args=["-m", "magda_agent.operations.cron_backups_v3"],
            env={"MAGDA_ENV": "production"},
            timeout_seconds=600,
            description="Daily backup of episodic memory",
            tags=["backup", "ops"],
        )
        self.exporter.register_task(self.backup_task)

    def test_export_json_manifest_and_validation(self):
        json_manifest = self.exporter.export_to_json_manifest()
        parsed = json.loads(json_manifest)

        self.assertEqual(parsed["platform"], "test_magda_ops")
        self.assertEqual(parsed["total_tasks"], 1)
        self.assertEqual(parsed["tasks"][0]["task_id"], "nightly_backup_job")
        self.assertEqual(parsed["tasks"][0]["schedule"], "0 3 * * *")

        is_valid, errors = self.exporter.validate_manifest(parsed)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_export_crontab(self):
        crontab_content = self.exporter.export_to_crontab()

        self.assertIn("0 3 * * *", crontab_content)
        self.assertIn("MAGDA_ENV=production python3 -m magda_agent.operations.cron_backups_v3", crontab_content)
        self.assertIn("Nightly Episodic Backup", crontab_content)

    def test_export_systemd_timer(self):
        service, timer = self.exporter.export_to_systemd_timer("nightly_backup_job")

        self.assertIn("[Unit]", service)
        self.assertIn("ExecStart=python3 -m magda_agent.operations.cron_backups_v3", service)
        self.assertIn("Environment=\"MAGDA_ENV=production\"", service)

        self.assertIn("[Timer]", timer)
        self.assertIn("OnCalendar=*-*-* 03:00:00", timer)

    def test_export_kubernetes_cronjob(self):
        k8s_cron = self.exporter.export_to_kubernetes_cronjob(
            "nightly_backup_job",
            image="magda/agent-core:v1.2",
            namespace="magda-system",
        )

        self.assertEqual(k8s_cron["apiVersion"], "batch/v1")
        self.assertEqual(k8s_cron["kind"], "CronJob")
        self.assertEqual(k8s_cron["metadata"]["namespace"], "magda-system")
        self.assertEqual(k8s_cron["spec"]["schedule"], "0 3 * * *")

        container = k8s_cron["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(container["image"], "magda/agent-core:v1.2")
        self.assertEqual(container["command"], ["python3"])
        self.assertEqual(container["args"], ["-m", "magda_agent.operations.cron_backups_v3"])
        self.assertEqual(container["env"], [{"name": "MAGDA_ENV", "value": "production"}])

    def test_manifest_validation_failure(self):
        invalid_manifest = {
            "tasks": [
                {"task_id": "broken_task"}  # missing command and schedule
            ]
        }
        is_valid, errors = self.exporter.validate_manifest(invalid_manifest)
        self.assertFalse(is_valid)
        self.assertGreaterEqual(len(errors), 2)


if __name__ == "__main__":
    unittest.main()
