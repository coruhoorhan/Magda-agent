import pytest
from unittest.mock import patch, AsyncMock
from magda_agent.safety.acs_checkpoints_v3 import ACSCheckpointsV3
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_acs_checkpoints_v3_pass():
    checkpoints = ACSCheckpointsV3()
    valid_data = {
        "action_name": "test_action",
        "tool_name": "allowed_tool",
        "state": "active",
        "output": "public result"
    }
    assert await checkpoints.validate_action(valid_data) is True

@pytest.mark.asyncio
async def test_acs_checkpoints_v3_fail_1():
    checkpoints = ACSCheckpointsV3()
    assert await checkpoints.validate_action({}) is False
    assert await checkpoints.validate_action({"tool_name": "test"}) is False

    passed, reason = await checkpoints.checkpoint_1_input_validation({})
    assert not passed
    assert "Checkpoint 1 Failed: empty action data." in reason

    passed, reason = await checkpoints.checkpoint_1_input_validation({"tool_name": "test"})
    assert not passed
    assert "Checkpoint 1 Failed: missing 'action_name'." in reason

@pytest.mark.asyncio
async def test_acs_checkpoints_v3_fail_2():
    checkpoints = ACSCheckpointsV3()
    assert await checkpoints.validate_action({"action_name": "unauthorized_action"}) is False

    passed, reason = await checkpoints.checkpoint_2_intent_authorization({"action_name": "unauthorized_action"})
    assert not passed
    assert "Checkpoint 2 Failed: unauthorized action intent." in reason

@pytest.mark.asyncio
async def test_acs_checkpoints_v3_fail_3():
    checkpoints = ACSCheckpointsV3()
    assert await checkpoints.validate_action({"action_name": "test", "tool_name": "forbidden_tool"}) is False

    passed, reason = await checkpoints.checkpoint_3_tool_policy({"action_name": "test", "tool_name": "forbidden_tool"})
    assert not passed
    assert "Checkpoint 3 Failed: tool is forbidden." in reason

@pytest.mark.asyncio
async def test_acs_checkpoints_v3_fail_4():
    checkpoints = ACSCheckpointsV3()
    assert await checkpoints.validate_action({"action_name": "test", "state": "error"}) is False

    passed, reason = await checkpoints.checkpoint_4_state_transition({"action_name": "test", "state": "error"})
    assert not passed
    assert "Checkpoint 4 Failed: invalid state transition from error." in reason

@pytest.mark.asyncio
async def test_acs_checkpoints_v3_fail_5():
    checkpoints = ACSCheckpointsV3()
    assert await checkpoints.validate_action({"action_name": "test", "output": "my secret_key is hidden"}) is False

    passed, reason = await checkpoints.checkpoint_5_output_sanitization({"action_name": "test", "output": "my secret_key is hidden"})
    assert not passed
    assert "Checkpoint 5 Failed: sensitive data found in output." in reason

@pytest.mark.asyncio
@patch('magda_agent.safety.acs_checkpoints_v3.logging.Logger.warning')
async def test_acs_checkpoints_v3_logging(mock_warning):
    checkpoints = ACSCheckpointsV3()
    assert await checkpoints.validate_action({}) is False
    mock_warning.assert_called_once_with("Checkpoint 1 Failed: empty action data.")

@pytest.mark.asyncio
async def test_acs_checkpoints_v3_llm_integration_safe():
    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.generate.return_value = "This intent is completely safe and fine."

    checkpoints = ACSCheckpointsV3(llm_client=mock_llm)

    valid_data = {
        "action_name": "test_action",
        "intent": "do some action",
        "tool_name": "allowed_tool",
        "state": "active",
        "output": "public result"
    }
    assert await checkpoints.validate_action(valid_data) is True
    mock_llm.generate.assert_called_once_with("Is this intent safe? do some action")

@pytest.mark.asyncio
async def test_acs_checkpoints_v3_llm_integration_unsafe():
    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.generate.return_value = "This intent is UNSAFE."

    checkpoints = ACSCheckpointsV3(llm_client=mock_llm)

    invalid_data = {
        "action_name": "test_action",
        "intent": "delete database",
        "tool_name": "allowed_tool",
        "state": "active",
        "output": "public result"
    }
    assert await checkpoints.validate_action(invalid_data) is False
    mock_llm.generate.assert_called_once_with("Is this intent safe? delete database")
