import pytest
import time
from unittest.mock import Mock, patch
from magda_agent.safety.prempti_logger_v1 import PremptiLoggerV1
from magda_agent.safety.audit_trail import AuditTrail

@pytest.fixture
def audit_trail():
    return AuditTrail(db_path=None)

@pytest.fixture
def logger(audit_trail):
    return PremptiLoggerV1(audit_trail=audit_trail)

def test_sync_intercept_success(logger, audit_trail):
    @logger.intercept(tool_name="test_tool", why="testing sync")
    def sync_func(a, b=2):
        return a + b

    result = sync_func(3)
    assert result == 5

    logs = audit_trail.query(tool_name="test_tool")
    assert len(logs) == 2
    assert logs[0]["result"] == "PENDING"
    assert logs[0]["why"] == "testing sync"
    assert logs[0]["kwargs"] == {"a": 3, "b": 2}

    assert logs[1]["result"] == 5
    assert logs[1]["why"] == "testing sync (completed)"
    assert logs[1]["kwargs"] == {"a": 3, "b": 2}

def test_sync_intercept_failure(logger, audit_trail):
    @logger.intercept(tool_name="test_tool_fail", why="testing fail")
    def sync_func_fail():
        raise ValueError("Oops")

    with pytest.raises(ValueError):
        sync_func_fail()

    logs = audit_trail.query(tool_name="test_tool_fail")
    assert len(logs) == 2
    assert logs[0]["result"] == "PENDING"
    assert logs[1]["result"] == "Oops"
    assert logs[1]["why"] == "testing fail (failed)"

@pytest.mark.asyncio
async def test_async_intercept_success(logger, audit_trail):
    @logger.intercept(tool_name="test_async", why="testing async")
    async def async_func(val):
        return val * 2

    result = await async_func(4)
    assert result == 8

    logs = audit_trail.query(tool_name="test_async")
    assert len(logs) == 2
    assert logs[0]["result"] == "PENDING"
    assert logs[0]["kwargs"] == {"val": 4}
    assert logs[1]["result"] == 8

@pytest.mark.asyncio
async def test_async_intercept_failure(logger, audit_trail):
    @logger.intercept(tool_name="test_async_fail")
    async def async_func_fail():
        raise RuntimeError("Async error")

    with pytest.raises(RuntimeError):
        await async_func_fail()

    logs = audit_trail.query(tool_name="test_async_fail")
    assert len(logs) == 2
    assert logs[0]["result"] == "PENDING"
    assert logs[1]["result"] == "Async error"
