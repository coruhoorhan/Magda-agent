import pytest
import asyncio
from unittest.mock import MagicMock

from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.realtime_guardrail_v2 import RealtimeGuardrailFallbackV2


@pytest.mark.asyncio
async def test_mcp_realtime_guardrail_v2_success() -> None:
    """
    Tests that a successful tool execution goes through without violations
    and returns (True, result).
    """
    policy = MagicMock(spec=PolicyLayer)
    policy.evaluate.return_value = (True, "Allowed")

    async def mock_tool(arg1: str) -> str:
        return f"Executed with {arg1}"

    fallback = RealtimeGuardrailFallbackV2(policy_layer=policy)
    success, result = await fallback.execute_with_reprompt_fallback(
        tool_func=mock_tool,
        tool_name="test_tool",
        kwargs={"arg1": "hello"}
    )

    assert success is True
    assert result == "Executed with hello"
    policy.evaluate.assert_called_once_with("test_tool", arg1="hello")


@pytest.mark.asyncio
async def test_mcp_realtime_guardrail_v2_policy_violation() -> None:
    """
    Tests that a policy violation is intercepted and returns a safety fallback reprompt.
    """
    policy = MagicMock(spec=PolicyLayer)
    policy.evaluate.return_value = (False, "Command injection pattern detected")

    async def mock_tool(arg1: str) -> str:
        return f"Executed with {arg1}"

    fallback = RealtimeGuardrailFallbackV2(policy_layer=policy)
    success, result = await fallback.execute_with_reprompt_fallback(
        tool_func=mock_tool,
        tool_name="restricted_tool",
        kwargs={"arg1": "unsafe_input"}
    )

    assert success is False
    assert "SAFETY GUARDRAIL TRIGGERED:" in result
    assert "blocked due to a policy violation: Command injection pattern detected" in result
    policy.evaluate.assert_called_once_with("restricted_tool", arg1="unsafe_input")


@pytest.mark.asyncio
async def test_mcp_realtime_guardrail_v2_execution_failure() -> None:
    """
    Tests that a tool execution failure is caught and returns an execution failure fallback reprompt.
    """
    policy = MagicMock(spec=PolicyLayer)
    policy.evaluate.return_value = (True, "Allowed")

    async def mock_tool_failing(arg1: str) -> str:
        raise ConnectionError("Timeout connecting to server")

    fallback = RealtimeGuardrailFallbackV2(policy_layer=policy)
    success, result = await fallback.execute_with_reprompt_fallback(
        tool_func=mock_tool_failing,
        tool_name="unstable_tool",
        kwargs={"arg1": "retry_this"}
    )

    assert success is False
    assert "EXECUTION FALLBACK TRIGGERED:" in result
    assert "Timeout connecting to server" in result
    policy.evaluate.assert_called_once_with("unstable_tool", arg1="retry_this")

@pytest.mark.asyncio
async def test_mcp_realtime_guardrail_v2_sync_tool_success() -> None:
    """
    Tests that a synchronous successful tool execution goes through without violations.
    """
    policy = MagicMock(spec=PolicyLayer)
    policy.evaluate.return_value = (True, "Allowed")

    def mock_sync_tool(arg1: str) -> str:
        return f"Sync executed with {arg1}"

    fallback = RealtimeGuardrailFallbackV2(policy_layer=policy)
    success, result = await fallback.execute_with_reprompt_fallback(
        tool_func=mock_sync_tool,
        tool_name="sync_tool",
        kwargs={"arg1": "world"}
    )

    assert success is True
    assert result == "Sync executed with world"
    policy.evaluate.assert_called_once_with("sync_tool", arg1="world")
