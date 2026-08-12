import json
import pytest
from magda_agent.skills.telemetry_export import SkillTelemetryTracker

def test_record_and_aggregate_metrics():
    tracker = SkillTelemetryTracker()

    # Record usage for skill_a
    tracker.record_usage("skill_a", success=True, execution_time_ms=100.0)
    tracker.record_usage("skill_a", success=False, execution_time_ms=50.0)
    tracker.record_usage("skill_a", success=True, execution_time_ms=150.0)

    # Record usage for skill_b
    tracker.record_usage("skill_b", success=True, execution_time_ms=200.0)

    aggregated = tracker.get_aggregated_metrics()

    assert "skill_a" in aggregated
    assert aggregated["skill_a"]["total_calls"] == 3
    assert aggregated["skill_a"]["success_rate"] == round(2/3, 4)
    assert aggregated["skill_a"]["average_execution_time_ms"] == 100.0

    assert "skill_b" in aggregated
    assert aggregated["skill_b"]["total_calls"] == 1
    assert aggregated["skill_b"]["success_rate"] == 1.0
    assert aggregated["skill_b"]["average_execution_time_ms"] == 200.0

def test_export_agentskills_format():
    tracker = SkillTelemetryTracker()

    tracker.record_usage("test_skill", success=True, execution_time_ms=120.5)

    export_json = tracker.export_agentskills_format()
    parsed = json.loads(export_json)

    assert parsed["version"] == "1.0"
    assert parsed["exporter"] == "magda_telemetry"
    assert "skills" in parsed

    skills = parsed["skills"]
    assert len(skills) == 1
    assert skills[0]["name"] == "test_skill"

    metrics = skills[0]["metrics"]
    assert metrics["total_calls"] == 1
    assert metrics["success_rate"] == 1.0
    assert metrics["average_execution_time_ms"] == 120.5

def test_empty_export():
    tracker = SkillTelemetryTracker()

    export_json = tracker.export_agentskills_format()
    parsed = json.loads(export_json)

    assert parsed["skills"] == []
    assert parsed["version"] == "1.0"
