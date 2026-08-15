import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.safety.guardrail_fallback import GuardrailFallbackExecutor
from magda_agent.safety.realtime_interceptor import RealtimeGuardrailInterceptor

@pytest.mark.asyncio
async def test_execute_with_fallback_success():
    """Test that when the primary tool succeeds, the fallback is not called."""
    # Arrange
    interceptor_mock = AsyncMock(spec=RealtimeGuardrailInterceptor)
    interceptor_mock.intercept_and_execute.return_value = (True, "primary_success_result")

    primary_tool = AsyncMock()
    fallback_tool = AsyncMock()

    executor = GuardrailFallbackExecutor()

    # Act
    success, result = await executor.execute_with_fallback(
        interceptor=interceptor_mock,
        tool_func=primary_tool,
        tool_name="test_tool",
        kwargs={"arg1": "value1"},
        fallback_func=fallback_tool,
        fallback_kwargs={"fallback_arg": "value2"}
    )

    # Assert
    assert success is True
    assert result == "primary_success_result"
    interceptor_mock.intercept_and_execute.assert_called_once_with(
        primary_tool, "test_tool", {"arg1": "value1"}
    )
    fallback_tool.assert_not_called()

@pytest.mark.asyncio
async def test_execute_with_fallback_blocked():
    """Test that when the primary tool is blocked (returns False), the fallback is called."""
    # Arrange
    interceptor_mock = AsyncMock(spec=RealtimeGuardrailInterceptor)
    interceptor_mock.intercept_and_execute.return_value = (False, "fallback prompt string")

    primary_tool = AsyncMock()
    fallback_tool = AsyncMock()
    fallback_tool.return_value = "fallback_success_result"

    executor = GuardrailFallbackExecutor()

    # Act
    success, result = await executor.execute_with_fallback(
        interceptor=interceptor_mock,
        tool_func=primary_tool,
        tool_name="test_tool",
        kwargs={"arg1": "value1"},
        fallback_func=fallback_tool,
        fallback_kwargs={"fallback_arg": "value2"}
    )

    # Assert
    assert success is True
    assert result == "fallback_success_result"
    interceptor_mock.intercept_and_execute.assert_called_once_with(
        primary_tool, "test_tool", {"arg1": "value1"}
    )
    fallback_tool.assert_called_once_with(fallback_arg="value2")

@pytest.mark.asyncio
async def test_execute_with_fallback_sync_fallback():
    """Test that a synchronous fallback function works properly."""
    # Arrange
    interceptor_mock = AsyncMock(spec=RealtimeGuardrailInterceptor)
    interceptor_mock.intercept_and_execute.return_value = (False, "fallback prompt string")

    primary_tool = AsyncMock()

    def sync_fallback(fallback_arg):
        return f"sync_fallback_result: {fallback_arg}"

    executor = GuardrailFallbackExecutor()

    # Act
    success, result = await executor.execute_with_fallback(
        interceptor=interceptor_mock,
        tool_func=primary_tool,
        tool_name="test_tool",
        kwargs={"arg1": "value1"},
        fallback_func=sync_fallback,
        fallback_kwargs={"fallback_arg": "value2"}
    )

    # Assert
    assert success is True
    assert result == "sync_fallback_result: value2"

@pytest.mark.asyncio
async def test_execute_both_fail():
    """Test that when both primary and fallback fail, it returns False."""
    # Arrange
    interceptor_mock = AsyncMock(spec=RealtimeGuardrailInterceptor)
    interceptor_mock.intercept_and_execute.return_value = (False, "primary failed")

    primary_tool = AsyncMock()
    fallback_tool = AsyncMock()
    fallback_tool.side_effect = Exception("fallback error")

    executor = GuardrailFallbackExecutor()

    # Act
    success, result = await executor.execute_with_fallback(
        interceptor=interceptor_mock,
        tool_func=primary_tool,
        tool_name="test_tool",
        kwargs={"arg1": "value1"},
        fallback_func=fallback_tool,
        fallback_kwargs={"fallback_arg": "value2"}
    )

    # Assert
    assert success is False
    assert "fallback error" in result
