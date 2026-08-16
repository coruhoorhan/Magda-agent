import logging
import pytest
from magda_agent.safety.acs_guard import ACSGuard, SecurityViolationError
from magda_agent.safety.audit_trail import AuditTrail
from magda_agent.safety.acs_logging import ACSStructuredLogger

def test_acs_structured_logger_all_pass():
    audit_trail = AuditTrail(db_path=None)
    guard = ACSGuard()
    structured_logger = ACSStructuredLogger(acs_guard=guard, audit_trail=audit_trail)

    valid_workflow = {
        "action": "execute",
        "tool": "system_tool",
        "current_state": "idle",
        "next_state": "planning",
        "output": "Clean output string"
    }

    passed, logs = structured_logger.evaluate_checkpoints(valid_workflow)
    assert passed is True
    assert len(logs) == 5

    stages = [log["stage"] for log in logs]
    assert stages == ["input", "input", "execution", "execution", "output"]

    statuses = [log["status"] for log in logs]
    assert all(s == "passed" for s in statuses)

    audit_logs = audit_trail.get_all()
    assert len(audit_logs) == 5
    assert all(entry["result"] == "allowed" for entry in audit_logs)

def test_acs_structured_logger_failure():
    audit_trail = AuditTrail(db_path=None)
    guard = ACSGuard()
    structured_logger = ACSStructuredLogger(acs_guard=guard, audit_trail=audit_trail)

    invalid_workflow = {
        "action": "unauthorized_action",
        "tool": "forbidden_tool",
        "output": "secret_key=12345"
    }

    passed, logs = structured_logger.evaluate_checkpoints(invalid_workflow)
    assert passed is False
    assert len(logs) == 5

    status_dict = {log["checkpoint_id"]: log["status"] for log in logs}
    assert status_dict[1] == "passed"
    assert status_dict[2] == "failed"
    assert status_dict[3] == "failed"
    assert status_dict[4] == "passed"
    assert status_dict[5] == "failed"

    audit_logs = audit_trail.get_all()
    assert len(audit_logs) == 5
    blocked_logs = [e for e in audit_logs if e["result"] == "blocked"]
    assert len(blocked_logs) == 3

def test_intercept_and_log_pass_and_fail():
    audit_trail = AuditTrail(db_path=None)
    guard = ACSGuard()
    structured_logger = ACSStructuredLogger(acs_guard=guard, audit_trail=audit_trail)

    valid_workflow = {
        "action": "read",
        "tool": "reader_tool",
        "current_state": "idle",
        "next_state": "reading",
        "output": "normal data"
    }

    res = structured_logger.intercept_and_log(valid_workflow)
    assert res == valid_workflow

    invalid_workflow = {
        "action": "unauthorized_action",
        "tool": "forbidden_tool"
    }

    with pytest.raises(SecurityViolationError) as exc_info:
        structured_logger.intercept_and_log(invalid_workflow)

    assert "Action blocked by ACS checkpoints" in str(exc_info.value)
