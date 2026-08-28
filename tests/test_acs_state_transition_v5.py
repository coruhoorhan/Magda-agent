import pytest
from magda_agent.safety.acs_state_transition_v5 import ACSStateTransitionV5

def test_checkpoint_4_state_transition_no_next_state():
    """Test that transition is allowed when next_state is not provided."""
    guardrail = ACSStateTransitionV5()
    workflow_data = {"current_state": "idle"}
    passed, reason = guardrail.checkpoint_4_state_transition(workflow_data)
    assert passed is True
    assert "not provided" in reason

def test_checkpoint_4_state_transition_valid_transition():
    """Test that a valid transition is allowed."""
    guardrail = ACSStateTransitionV5()
    workflow_data = {"current_state": "idle", "next_state": "planning"}
    passed, reason = guardrail.checkpoint_4_state_transition(workflow_data)
    assert passed is True
    assert reason == "State transition passed."

def test_checkpoint_4_state_transition_valid_error_transition():
    """Test that transition to error state is allowed from any known state."""
    guardrail = ACSStateTransitionV5()
    workflow_data = {"current_state": "planning", "next_state": "error"}
    passed, reason = guardrail.checkpoint_4_state_transition(workflow_data)
    assert passed is True
    assert reason == "State transition passed."

def test_checkpoint_4_state_transition_invalid_transition():
    """Test that an invalid transition is blocked."""
    guardrail = ACSStateTransitionV5()
    workflow_data = {"current_state": "reflecting", "next_state": "executing"}
    passed, reason = guardrail.checkpoint_4_state_transition(workflow_data)
    assert passed is False
    assert "cannot transition" in reason

def test_checkpoint_4_state_transition_unknown_current_state():
    """Test that an unknown current_state is blocked."""
    guardrail = ACSStateTransitionV5()
    workflow_data = {"current_state": "unknown_state", "next_state": "idle"}
    passed, reason = guardrail.checkpoint_4_state_transition(workflow_data)
    assert passed is False
    assert "unknown current_state" in reason

def test_checkpoint_4_state_transition_default_current_state():
    """Test that default current_state 'idle' is correctly handled."""
    guardrail = ACSStateTransitionV5()
    # Missing current_state implies "idle"
    workflow_data = {"next_state": "planning"}
    passed, reason = guardrail.checkpoint_4_state_transition(workflow_data)
    assert passed is True
    assert reason == "State transition passed."
