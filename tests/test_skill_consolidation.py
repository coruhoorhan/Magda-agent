import time
import pytest
from unittest.mock import MagicMock

from magda_agent.safety.audit_trail import AuditTrail
from magda_agent.skills.registry import SkillRegistry
from magda_agent.learning.skill_consolidation import SkillConsolidator


def test_extract_successful_traces():
    audit_trail = AuditTrail(db_path=None)

    # Log mix of successful, blocked, error, and tool execution error logs
    audit_trail.log_call("search_web", {"q": "python"}, "user request", "search results", duration=0.5)
    audit_trail.log_call("fetch_page", {"url": "https://python.org"}, "user request", "blocked", duration=0.1)
    audit_trail.log_call("parse_html", {"html": "<p>hi</p>"}, "user request", "parsed data", duration=0.2)
    audit_trail.log_call("write_file", {"path": "out.txt"}, "user request", "error", duration=0.1)
    audit_trail.log_call("run_code", {"cmd": "ls"}, "user request", "Error: failed to run", duration=0.3)

    consolidator = SkillConsolidator(audit_trail=audit_trail)
    successful = consolidator.extract_successful_traces()

    assert len(successful) == 2
    tool_names = [e["tool_name"] for e in successful]
    assert "search_web" in tool_names
    assert "parse_html" in tool_names
    assert "fetch_page" not in tool_names
    assert "write_file" not in tool_names
    assert "run_code" not in tool_names


def test_detect_frequent_sequences():
    now = time.time()
    traces = [
        {"timestamp": now + 1, "tool_name": "toolA", "result": "ok"},
        {"timestamp": now + 2, "tool_name": "toolB", "result": "ok"},
        {"timestamp": now + 10, "tool_name": "toolA", "result": "ok"},
        {"timestamp": now + 11, "tool_name": "toolB", "result": "ok"},
        {"timestamp": now + 20, "tool_name": "toolC", "result": "ok"},
    ]

    consolidator = SkillConsolidator()
    frequent = consolidator.detect_frequent_sequences(traces, sequence_length=2, min_occurrences=2)

    assert len(frequent) == 1
    seq, instances = frequent[0]
    assert seq == ("toolA", "toolB")
    assert len(instances) == 2


def test_consolidate_skills_and_execute_macro():
    audit_trail = AuditTrail(db_path=None)
    skill_registry = SkillRegistry()

    # Register primitive tools in skill registry
    def mock_tool_a(val: str = "default_a"):
        return f"a_out_{val}"

    def mock_tool_b(val: str = "default_b"):
        return f"b_out_{val}"

    skill_registry.register_skill("toolA", mock_tool_a, "Tool A description")
    skill_registry.register_skill("toolB", mock_tool_b, "Tool B description")

    now = time.time()
    # Log two occurrences of toolA -> toolB sequence
    audit_trail.log_call("toolA", {"val": "1"}, "step 1", "a_out_1", duration=0.1)
    audit_trail.log_call("toolB", {"val": "2"}, "step 2", "b_out_2", duration=0.1)

    audit_trail.log_call("toolA", {"val": "3"}, "step 1", "a_out_3", duration=0.1)
    audit_trail.log_call("toolB", {"val": "4"}, "step 2", "b_out_4", duration=0.1)

    procedural_memory = MagicMock()

    consolidator = SkillConsolidator(
        audit_trail=audit_trail,
        skill_registry=skill_registry,
        procedural_memory=procedural_memory,
    )

    macros = consolidator.consolidate_skills(sequence_length=2, min_occurrences=2)

    expected_macro_name = "macro_toolA_then_toolB"
    assert expected_macro_name in macros
    assert skill_registry.has_skill(expected_macro_name)

    # Test executing the consolidated macro via SkillRegistry
    res = skill_registry.execute_skill(
        expected_macro_name,
        toolA_val="test_val_a",
        toolB_val="test_val_b",
    )

    assert res["macro_name"] == expected_macro_name
    assert res["success"] is True
    assert len(res["step_results"]) == 2
    assert res["step_results"][0]["result"] == "a_out_test_val_a"
    assert res["step_results"][1]["result"] == "b_out_test_val_b"

    # Verify procedural memory stored procedure
    procedural_memory.store_procedure.assert_called_once()
    call_args = procedural_memory.store_procedure.call_args
    assert call_args.kwargs["name"] == expected_macro_name
