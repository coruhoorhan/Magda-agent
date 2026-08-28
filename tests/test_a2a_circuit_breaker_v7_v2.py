import pytest
import asyncio
import time
from magda_agent.integration.a2a_circuit_breaker_v7_v2 import A2ACircuitBreaker, CircuitBreakerState, CircuitBreakerOpenException
from unittest.mock import AsyncMock, patch

@pytest.fixture
def circuit_breaker() -> A2ACircuitBreaker:
    return A2ACircuitBreaker(failure_threshold=3, reset_timeout=1.0)

@pytest.mark.asyncio
async def test_successful_call(circuit_breaker: A2ACircuitBreaker) -> None:
    mock_func = AsyncMock(return_value="success")
    result = await circuit_breaker.call(mock_func)
    assert result == "success"
    assert circuit_breaker.state == CircuitBreakerState.CLOSED
    assert circuit_breaker.failure_count == 0

@pytest.mark.asyncio
async def test_failure_records_and_trips(circuit_breaker: A2ACircuitBreaker) -> None:
    mock_func = AsyncMock(side_effect=RuntimeError("API Error"))

    # Fail 1
    with pytest.raises(RuntimeError):
        await circuit_breaker.call(mock_func)
    assert circuit_breaker.state == CircuitBreakerState.CLOSED
    assert circuit_breaker.failure_count == 1

    # Fail 2
    with pytest.raises(RuntimeError):
        await circuit_breaker.call(mock_func)
    assert circuit_breaker.state == CircuitBreakerState.CLOSED
    assert circuit_breaker.failure_count == 2

    # Fail 3 - should trip the breaker
    with pytest.raises(RuntimeError):
        await circuit_breaker.call(mock_func)
    assert circuit_breaker.state == CircuitBreakerState.OPEN

@pytest.mark.asyncio
async def test_open_circuit_raises_immediately(circuit_breaker: A2ACircuitBreaker) -> None:
    mock_func = AsyncMock(side_effect=RuntimeError("API Error"))

    # Trip the breaker
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await circuit_breaker.call(mock_func)

    assert circuit_breaker.state == CircuitBreakerState.OPEN

    # Next call should raise CircuitBreakerOpenException immediately without calling mock_func
    mock_func.reset_mock()
    with pytest.raises(CircuitBreakerOpenException, match="Circuit breaker is OPEN"):
        await circuit_breaker.call(mock_func)

    mock_func.assert_not_called()

@pytest.mark.asyncio
async def test_recovery_half_open_to_closed(circuit_breaker: A2ACircuitBreaker) -> None:
    mock_func = AsyncMock(side_effect=RuntimeError("API Error"))

    # Trip the breaker
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await circuit_breaker.call(mock_func)

    assert circuit_breaker.state == CircuitBreakerState.OPEN

    # Mock time to simulate waiting for reset timeout
    with patch('time.time', return_value=time.time() + 2.0):
        # Now we are past reset_timeout
        # Create a new mock func that succeeds
        success_func = AsyncMock(return_value="success")

        # This call should succeed and transition to CLOSED
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

@pytest.mark.asyncio
async def test_half_open_to_open_on_failure(circuit_breaker: A2ACircuitBreaker) -> None:
    mock_func = AsyncMock(side_effect=RuntimeError("API Error"))

    # Trip the breaker
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await circuit_breaker.call(mock_func)

    assert circuit_breaker.state == CircuitBreakerState.OPEN

    # Mock time to simulate waiting for reset timeout
    with patch('time.time', return_value=time.time() + 2.0):
        # This call will fail again while in HALF_OPEN
        with pytest.raises(RuntimeError):
            await circuit_breaker.call(mock_func)

        assert circuit_breaker.state == CircuitBreakerState.OPEN
