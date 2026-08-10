import asyncio
import pytest
from unittest.mock import MagicMock
from magda_agent.safety.realtime_interceptor import RealtimeGuardrailInterceptor, PolicyViolationError


class MockPolicyLayer:
    def __init__(self, allow=True, explanation="Allowed"):
        self.allow = allow
        self.explanation = explanation

    def evaluate(self, tool_name, **kwargs):
        return self.allow, self.explanation


def test_interceptor_pre_execution_block():
    mock_policy = MockPolicyLayer(allow=False, explanation="Denied due to test.")
    interceptor = RealtimeGuardrailInterceptor(policy_layer=mock_policy)

    async def mock_tool():
        return "Success"

    result, message = asyncio.run(interceptor.intercept_and_execute(mock_tool, "test_tool", {}))
    assert result is False
    assert "SAFETY ALERT" in message
    assert "Denied due to test" in message


def test_interceptor_mid_execution_policy_violation():
    mock_policy = MockPolicyLayer(allow=True)
    interceptor = RealtimeGuardrailInterceptor(policy_layer=mock_policy)

    async def mock_tool():
        raise PolicyViolationError("Detected unsafe output generated mid-execution")

    result, message = asyncio.run(interceptor.intercept_and_execute(mock_tool, "test_tool", {}))
    assert result is False
    assert "MID-EXECUTION SAFETY ALERT" in message
    assert "Detected unsafe output generated mid-execution" in message


def test_interceptor_execution_failure():
    mock_policy = MockPolicyLayer(allow=True)
    interceptor = RealtimeGuardrailInterceptor(policy_layer=mock_policy)

    def sync_mock_tool():
        raise ValueError("Generic error")

    result, message = asyncio.run(interceptor.intercept_and_execute(sync_mock_tool, "sync_tool", {}))
    assert result is False
    assert "EXECUTION FAILURE" in message
    assert "Generic error" in message


def test_interceptor_success_async():
    mock_policy = MockPolicyLayer(allow=True)
    interceptor = RealtimeGuardrailInterceptor(policy_layer=mock_policy)

    async def async_mock_tool(arg1):
        return f"Async Success: {arg1}"

    result, message = asyncio.run(interceptor.intercept_and_execute(async_mock_tool, "async_tool", {"arg1": "test"}))
    assert result is True
    assert message == "Async Success: test"


def test_interceptor_success_sync():
    mock_policy = MockPolicyLayer(allow=True)
    interceptor = RealtimeGuardrailInterceptor(policy_layer=mock_policy)

    def sync_mock_tool(arg1):
        return f"Sync Success: {arg1}"

    result, message = asyncio.run(interceptor.intercept_and_execute(sync_mock_tool, "sync_tool", {"arg1": "test"}))
    assert result is True
    assert message == "Sync Success: test"
