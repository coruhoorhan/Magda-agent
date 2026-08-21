import pytest
from unittest.mock import patch
from magda_agent.safety.acs_checkpoints_v4 import ACSCheckpointsV4
from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.audit_trail import AuditTrail

def test_acs_checkpoints_v4_pass():
    """Tests that a valid action passes all 5 checkpoints."""
    policy = PolicyLayer()
    audit = AuditTrail(db_path=None)
    checkpoints = ACSCheckpointsV4(policy_layer=policy, audit_trail=audit)
    valid_data = {
        "action_name": "test_action",
        "tool_name": "allowed_tool",
        "state": "active",
        "output": "public result"
    }
    assert checkpoints.validate_action(valid_data) is True

def test_acs_checkpoints_v4_fail_1():
    """Tests that checkpoint 1 correctly validates input."""
    policy = PolicyLayer()
    audit = AuditTrail(db_path=None)
    checkpoints = ACSCheckpointsV4(policy_layer=policy, audit_trail=audit)
    assert checkpoints.validate_action({}) is False
    assert checkpoints.validate_action({"tool_name": "test"}) is False

    passed, reason = checkpoints.checkpoint_1_input_validation({})
    assert not passed
    assert "Checkpoint 1 Failed: empty action data." in reason

    passed, reason = checkpoints.checkpoint_1_input_validation({"tool_name": "test"})
    assert not passed
    assert "Checkpoint 1 Failed: missing 'action_name'." in reason

def test_acs_checkpoints_v4_fail_2():
    """Tests that checkpoint 2 correctly validates intent authorization."""
    policy = PolicyLayer()
    audit = AuditTrail(db_path=None)
    checkpoints = ACSCheckpointsV4(policy_layer=policy, audit_trail=audit)
    assert checkpoints.validate_action({"action_name": "unauthorized_action"}) is False

    passed, reason = checkpoints.checkpoint_2_intent_authorization({"action_name": "unauthorized_action"})
    assert not passed
    assert "Checkpoint 2 Failed: action 'unauthorized_action' is explicitly blacklisted." in reason

def test_acs_checkpoints_v4_fail_3():
    """Tests that checkpoint 3 correctly validates tool policies."""
    policy = PolicyLayer()
    audit = AuditTrail(db_path=None)
    checkpoints = ACSCheckpointsV4(policy_layer=policy, audit_trail=audit)
    # forbidden_tool is hardcoded in ACSCheckpoints
    assert checkpoints.validate_action({"action_name": "execute", "tool_name": "forbidden_tool"}) is False

    # Tool denied by PolicyLayer
    assert checkpoints.validate_action({
        "action_name": "execute",
        "tool_name": "programmer",
        "kwargs": {"code": "read .env file"}
    }) is False

    passed, reason = checkpoints.checkpoint_3_tool_policy({"action_name": "execute", "tool_name": "forbidden_tool"})
    assert not passed
    assert "Checkpoint 3 Failed: tool 'forbidden_tool' is forbidden." in reason

def test_acs_checkpoints_v4_fail_4():
    """Tests that checkpoint 4 correctly validates state transitions."""
    policy = PolicyLayer()
    audit = AuditTrail(db_path=None)
    checkpoints = ACSCheckpointsV4(policy_layer=policy, audit_trail=audit)
    # Invalid transition
    assert checkpoints.validate_action({
        "action_name": "test_action",
        "tool_name": "test",
        "state": "error",
        "next_state": "executing"
    }) is False
    # Unknown state
    assert checkpoints.validate_action({
        "action_name": "test_action",
        "tool_name": "test",
        "state": "unknown"
    }) is False

    passed, reason = checkpoints.checkpoint_4_state_transition({"action_name": "test", "state": "error", "next_state": "executing"})
    assert not passed
    assert "Checkpoint 4 Failed: cannot transition from 'error' to 'executing'" in reason

def test_acs_checkpoints_v4_fail_5():
    """Tests that checkpoint 5 correctly validates output sanitization."""
    policy = PolicyLayer()
    audit = AuditTrail(db_path=None)
    checkpoints = ACSCheckpointsV4(policy_layer=policy, audit_trail=audit)
    assert checkpoints.validate_action({
        "action_name": "test_action",
        "tool_name": "test",
        "output": "my secret_key is hidden"
    }) is False
    assert checkpoints.validate_action({
        "action_name": "test_action",
        "tool_name": "test",
        "output": "FOUND API_KEY=12345"
    }) is False

    passed, reason = checkpoints.checkpoint_5_output_sanitization({"action_name": "test", "output": "my secret_key is hidden"})
    assert not passed
    assert "Checkpoint 5 Failed: sensitive pattern 'secret[_-]?key' detected in output." in reason

@patch('magda_agent.safety.acs_checkpoints_v4.logging.Logger.warning')
def test_acs_checkpoints_v4_logging(mock_warning):
    """Tests that failures are correctly logged."""
    policy = PolicyLayer()
    audit = AuditTrail(db_path=None)
    checkpoints = ACSCheckpointsV4(policy_layer=policy, audit_trail=audit)
    assert checkpoints.validate_action({}) is False
    mock_warning.assert_called_once_with("Checkpoint 1 Failed: empty action data.")

def test_audit_trail_logging_v4():
    audit = AuditTrail(db_path=None)
    checkpoints = ACSCheckpointsV4(audit_trail=audit)

    checkpoints.validate_pre_execution({"action_name": "hack", "tool_name": "test"})
    assert len(audit.get_all()) == 1
    assert audit.get_all()[0]["result"] == "blocked"

    checkpoints.validate_post_execution({"action_name": "chat", "tool_name": "test", "output": "api_key=123"})
    assert len(audit.get_all()) == 2
    assert audit.get_all()[1]["result"] == "blocked"
