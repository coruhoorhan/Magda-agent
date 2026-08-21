import pytest
from unittest.mock import MagicMock
from magda_agent.safety.acs_adaptive_v1 import AdaptiveGuardrail
from magda_agent.safety.acs_checkpoints_v4 import ACSCheckpointsV4

@pytest.fixture
def mock_acs_checkpoints() -> MagicMock:
    """Fixture that provides a mock ACSCheckpoints instance."""
    mock = MagicMock(spec=ACSCheckpointsV4)
    # Default mock behavior: all stages pass
    mock.checkpoint_1_input_validation.return_value = (True, 'Passed')
    mock.checkpoint_2_intent_authorization.return_value = (True, 'Passed')
    mock.checkpoint_3_tool_policy.return_value = (True, 'Passed')
    mock.checkpoint_4_state_transition.return_value = (True, 'Passed')
    mock.checkpoint_5_output_sanitization.return_value = (True, 'Passed')
    return mock

@pytest.fixture
def adaptive_guardrail(mock_acs_checkpoints: MagicMock) -> AdaptiveGuardrail:
    """Fixture that provides an AdaptiveGuardrail instance configured with the mock."""
    return AdaptiveGuardrail(acs_checkpoints=mock_acs_checkpoints)

def test_low_risk_bypasses_checkpoints(adaptive_guardrail: AdaptiveGuardrail, mock_acs_checkpoints: MagicMock) -> None:
    """Tests that low risk workflows only execute the INPUT stage."""
    workflow_data = {"action_name": "chat", "tool_name": "test_tool"}

    passed, msg = adaptive_guardrail.evaluate(workflow_data, risk_score="low")

    assert passed is True
    assert "Adaptive guardrails passed" in msg
    mock_acs_checkpoints.checkpoint_1_input_validation.assert_called_once_with(workflow_data)
    mock_acs_checkpoints.checkpoint_2_intent_authorization.assert_called_once_with(workflow_data)

def test_medium_risk_runs_execution(adaptive_guardrail: AdaptiveGuardrail, mock_acs_checkpoints: MagicMock) -> None:
    """Tests that medium risk workflows execute both INPUT and EXECUTION stages, handling failures."""
    workflow_data = {"action_name": "chat", "tool_name": "forbidden_tool"}

    mock_acs_checkpoints.checkpoint_3_tool_policy.return_value = (False, 'Checkpoint 3 Failed: Tool forbidden')

    passed, msg = adaptive_guardrail.evaluate(workflow_data, risk_score="medium")

    assert passed is False
    assert "Checkpoint 3 Failed: Tool forbidden" in msg


def test_high_risk_runs_all(adaptive_guardrail: AdaptiveGuardrail, mock_acs_checkpoints: MagicMock) -> None:
    """Tests that high risk workflows execute all stages, and properly catches output sanitization failures."""
    workflow_data = {"action_name": "chat", "tool_name": "valid_tool", "output": "secret"}

    mock_acs_checkpoints.checkpoint_5_output_sanitization.return_value = (False, 'Checkpoint 5 Failed: Sensitive data')

    passed, msg = adaptive_guardrail.evaluate(workflow_data, risk_score="high")

    assert passed is False
    assert "Checkpoint 5 Failed: Sensitive data" in msg
