import pytest
from unittest.mock import MagicMock
import asyncio

from magda_agent.safety.acs_controls_v5 import ACSControlsV5
from magda_agent.safety.guardrails import SecurityViolationError
from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.audit_trail import AuditTrail

@pytest.fixture
def policy_layer_mock():
    mock = MagicMock(spec=PolicyLayer)
    mock.evaluate.return_value = (True, "Policy allowed.")
    return mock

@pytest.fixture
def audit_trail_mock():
    return MagicMock(spec=AuditTrail)

@pytest.fixture
def acs_controls(policy_layer_mock, audit_trail_mock):
    return ACSControlsV5(policy_layer=policy_layer_mock, audit_trail=audit_trail_mock)

@pytest.fixture
def valid_action_data():
    return {
        "action_name": "execute",
        "tool_name": "safe_tool",
        "current_state": "idle",
        "next_state": "executing",
        "kwargs": {"arg1": "value1"}
    }

def test_checkpoint_1_input_validation(acs_controls):
    # Valid
    assert acs_controls.checkpoint_1_input_validation({"action_name": "x", "tool_name": "y"})[0] is True

    # Invalid type
    assert acs_controls.checkpoint_1_input_validation([])[0] is False
    assert acs_controls.checkpoint_1_input_validation(None)[0] is False

    # Missing fields
    assert acs_controls.checkpoint_1_input_validation({"action_name": "x"})[0] is False
    assert acs_controls.checkpoint_1_input_validation({"tool_name": "y"})[0] is False

def test_checkpoint_2_intent_authorization(acs_controls):
    # Allowed intent
    assert acs_controls.checkpoint_2_intent_authorization({"action_name": "execute"})[0] is True

    # Blacklisted
    assert acs_controls.checkpoint_2_intent_authorization({"action_name": "unauthorized_action"})[0] is False

    # Unknown intent
    assert acs_controls.checkpoint_2_intent_authorization({"action_name": "launch_missiles"})[0] is False

def test_checkpoint_3_tool_policy(acs_controls, policy_layer_mock):
    # Allowed by default mock
    assert acs_controls.checkpoint_3_tool_policy({"tool_name": "safe_tool", "kwargs": {}})[0] is True

    # Explicitly forbidden
    assert acs_controls.checkpoint_3_tool_policy({"tool_name": "forbidden_tool"})[0] is False

    # Denied by policy layer
    policy_layer_mock.evaluate.return_value = (False, "Blocked by policy.")
    assert acs_controls.checkpoint_3_tool_policy({"tool_name": "unsafe_tool"})[0] is False

def test_checkpoint_4_state_transition(acs_controls):
    # Valid transition
    assert acs_controls.checkpoint_4_state_transition({"current_state": "idle", "next_state": "executing"})[0] is True

    # Invalid transition
    assert acs_controls.checkpoint_4_state_transition({"current_state": "idle", "next_state": "evaluating"})[0] is False

    # Unknown state
    assert acs_controls.checkpoint_4_state_transition({"current_state": "unknown"})[0] is False

def test_checkpoint_5_output_sanitization(acs_controls):
    # Clean output
    assert acs_controls.checkpoint_5_output_sanitization("Hello world")[0] is True

    # Sensitive output (SSN mock)
    assert acs_controls.checkpoint_5_output_sanitization("My number is 123-45-6789.")[0] is False

    # Sensitive output (CC mock)
    assert acs_controls.checkpoint_5_output_sanitization("Charge card 1234-5678-9012-3456.")[0] is False

def test_execute_with_checkpoints_success_sync(acs_controls, valid_action_data, audit_trail_mock):
    def dummy_tool(arg1):
        return f"Result: {arg1}"

    result = acs_controls.execute_with_checkpoints(dummy_tool, valid_action_data)

    assert result == "Result: value1"
    audit_trail_mock.log_call.assert_called_with(
        tool_name="safe_tool",
        kwargs={"arg1": "value1"},
        why="All 5 ACS checkpoints passed.",
        result="allowed",
        duration=0.0
    )

@pytest.mark.asyncio
async def test_execute_with_checkpoints_success_async(acs_controls, valid_action_data, audit_trail_mock):
    async def dummy_tool_async(arg1):
        return f"Result: {arg1}"

    result = await acs_controls.execute_with_checkpoints(dummy_tool_async, valid_action_data)

    assert result == "Result: value1"
    audit_trail_mock.log_call.assert_called()

def test_execute_pre_execution_failure_blocks(acs_controls, valid_action_data, audit_trail_mock):
    def dummy_tool(arg1):
        return "Should not be called"

    # Make checkpoint 2 fail
    valid_action_data["action_name"] = "unauthorized_action"

    with pytest.raises(SecurityViolationError, match="Checkpoint 2 Failed"):
        acs_controls.execute_with_checkpoints(dummy_tool, valid_action_data)

    audit_trail_mock.log_call.assert_called_with(
        tool_name="safe_tool",
        kwargs={"arg1": "value1"},
        why="Checkpoint 2 Failed: action 'unauthorized_action' is explicitly blacklisted.",
        result="blocked",
        duration=0.0
    )

def test_execute_post_execution_failure_blocks(acs_controls, valid_action_data, audit_trail_mock):
    def dummy_tool(arg1):
        # Tool returns sensitive data
        return "Secret: 123-45-6789"

    with pytest.raises(SecurityViolationError, match="Checkpoint 5 Failed"):
        acs_controls.execute_with_checkpoints(dummy_tool, valid_action_data)

    audit_trail_mock.log_call.assert_called_with(
        tool_name="safe_tool",
        kwargs={"arg1": "value1"},
        why=r"Checkpoint 5 Failed: sensitive pattern '\b\d{3}-\d{2}-\d{4}\b' detected in output.",
        result="blocked",
        duration=0.0
    )
