"""Tests for ACS Guardrails Taint Tracking V3 (magda_agent/safety/acs_taint_v3.py)."""

import pytest
from unittest.mock import MagicMock
from magda_agent.safety.acs_taint_v3 import (
    ACSTaintGuardrailV3,
    ACSTaintViolationError,
)
from magda_agent.safety.policy import PolicyLayer


def test_taint_and_origin_tracking():
    guard = ACSTaintGuardrailV3()
    untainted_text = "hello world"
    assert not guard.is_tainted(untainted_text)

    tainted_text = guard.taint_data(untainted_text, origin="untrusted_api")
    assert guard.is_tainted(tainted_text)
    assert guard.get_origins(tainted_text) == {"untrusted_api"}

    sanitized = guard.sanitize(tainted_text)
    assert not guard.is_tainted(sanitized)
    assert sanitized == "hello world"


def test_validate_tool_execution_untainted_passes():
    mock_policy = MagicMock(spec=PolicyLayer)
    mock_policy.evaluate.return_value = (True, "Allowed")
    guard = ACSTaintGuardrailV3(policy_layer=mock_policy)

    # Executing tool with untainted args should succeed
    def sample_tool(command: str):
        return f"Executed {command}"

    res = guard.execute_tool(sample_tool, "shell_cmd", command="ls")
    assert res == "Executed ls"
    mock_policy.evaluate.assert_called_once_with("shell_cmd", command="ls")


def test_validate_tool_execution_tainted_input_blocked():
    mock_policy = MagicMock(spec=PolicyLayer)
    mock_policy.evaluate.return_value = (True, "Allowed")
    guard = ACSTaintGuardrailV3(policy_layer=mock_policy)

    tainted_arg = guard.taint_data("rm -rf /", origin="external_user_input")

    def sample_tool(command: str):
        return f"Executed {command}"

    with pytest.raises(ACSTaintViolationError) as exc_info:
        guard.execute_tool(sample_tool, "shell_cmd", command=tainted_arg)

    assert "Tainted payload blocked from re-execution" in str(exc_info.value)
    assert "external_user_input" in exc_info.value.details["origins"]


def test_validate_sensitive_tool_blocked():
    guard = ACSTaintGuardrailV3(sensitive_tools={"execute_code"})
    tainted_code = guard.taint_data("import os; os.system('env')", origin="chat_input")

    def execute_code(code: str):
        return "done"

    with pytest.raises(ACSTaintViolationError):
        guard.execute_tool(execute_code, "execute_code", code=tainted_code)


def test_policy_layer_blocking():
    mock_policy = MagicMock(spec=PolicyLayer)
    mock_policy.evaluate.return_value = (False, "Command forbidden by security policy")
    guard = ACSTaintGuardrailV3(policy_layer=mock_policy)

    def sample_tool(command: str):
        return "ok"

    with pytest.raises(ACSTaintViolationError) as exc_info:
        guard.execute_tool(sample_tool, "restricted_tool", command="do_something")

    assert "blocked by policy layer" in str(exc_info.value)


def test_a2a_broadcast_validation():
    guard = ACSTaintGuardrailV3(block_tainted_a2a_broadcast=True)

    untainted_payload = {"status": "ok", "result": 42}
    broadcast_mock = MagicMock(return_value={"sent": True})

    res = guard.broadcast_a2a_payload(broadcast_mock, untainted_payload, target_agent_id="agent_123")
    assert res == {"sent": True}
    broadcast_mock.assert_called_once_with(untainted_payload, target_agent_id="agent_123")

    # Tainted payload should be blocked
    tainted_payload = {"status": "ok", "secret": guard.taint_data("api_key_123", origin="vault")}

    with pytest.raises(ACSTaintViolationError) as exc_info:
        guard.broadcast_a2a_payload(broadcast_mock, tainted_payload, target_agent_id="agent_123")

    assert "blocked from A2A broadcast" in str(exc_info.value)
    assert exc_info.value.details["target_agent_id"] == "agent_123"


@pytest.mark.asyncio
async def test_async_tool_execution():
    guard = ACSTaintGuardrailV3()

    async def async_tool(param: str):
        return f"async result: {param}"

    res_coro = guard.execute_tool(async_tool, "async_tool", param="test")
    res = await res_coro
    assert res == "async result: test"
