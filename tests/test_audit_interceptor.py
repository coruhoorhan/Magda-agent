import os
import pytest
import asyncio
import tempfile
from magda_agent.safety.audit_interceptor import PreemptiveAuditInterceptor

@pytest.fixture
def temp_db() -> str:
    """Fixture to provide a temporary database file path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)

def test_sync_preemptive_interception(temp_db) -> None:
    """Test preemptive logging and execution for sync functions."""
    interceptor = PreemptiveAuditInterceptor(db_path=temp_db)

    @interceptor.intercept(tool_name="test_sync_tool", why="checking sync execution")
    def my_tool(x: int, password: str = "supersecret") -> int:
        # Check that the preemptive log was ALREADY written with status "preemptive_start"
        # and with sanitized arguments BEFORE the tool finishes executing!
        in_progress_logs = interceptor.query(tool_name="test_sync_tool", status="preemptive_start")
        assert len(in_progress_logs) == 1
        assert in_progress_logs[0]["kwargs"]["x"] == x
        assert in_progress_logs[0]["kwargs"]["password"] == "***"
        return x * 10

    result = my_tool(42)
    assert result == 420

    # Query completed status
    completed_logs = interceptor.query(tool_name="test_sync_tool", status="success")
    assert len(completed_logs) == 1
    assert completed_logs[0]["result"] == 420
    assert completed_logs[0]["duration"] >= 0

    # Verify SQLite database query matches
    db_logs = interceptor.query_db(tool_name="test_sync_tool")
    assert len(db_logs) == 2  # one preemptive_start, one success
    start_entry = [e for e in db_logs if e["status"] == "preemptive_start"][0]
    success_entry = [e for e in db_logs if e["status"] == "success"][0]
    assert start_entry["kwargs"]["x"] == 42
    assert start_entry["kwargs"]["password"] == "***"
    assert success_entry["result"] == 420

def test_sync_failure_interception(temp_db) -> None:
    """Test sync failures are caught and logged, preserving the preemptive start record."""
    interceptor = PreemptiveAuditInterceptor(db_path=temp_db)

    @interceptor.intercept(tool_name="failing_sync_tool", why="trigger exception")
    def bad_tool(val: str) -> None:
        raise ValueError(f"Bad value: {val}")

    with pytest.raises(ValueError, match="Bad value: oops"):
        bad_tool("oops")

    # In-memory check
    logs = interceptor.query(tool_name="failing_sync_tool")
    assert len(logs) == 2
    assert logs[0]["status"] == "preemptive_start"
    assert logs[1]["status"] == "failed"
    assert "Bad value: oops" in logs[1]["result"]

    # Database check
    db_logs = interceptor.query_db(tool_name="failing_sync_tool")
    assert len(db_logs) == 2
    assert db_logs[0]["status"] == "preemptive_start"
    assert db_logs[1]["status"] == "failed"
    assert "Bad value: oops" in db_logs[1]["result"]

@pytest.mark.asyncio
async def test_async_preemptive_interception(temp_db) -> None:
    """Test preemptive logging and execution for async functions."""
    interceptor = PreemptiveAuditInterceptor(db_path=temp_db)

    @interceptor.intercept(tool_name="test_async_tool", why="checking async execution")
    async def my_async_tool(api_key: str, data: list) -> str:
        # Check that the preemptive log exists before this coroutine returns
        in_progress_logs = interceptor.query(tool_name="test_async_tool", status="preemptive_start")
        assert len(in_progress_logs) == 1
        assert in_progress_logs[0]["kwargs"]["api_key"] == "***"
        assert in_progress_logs[0]["kwargs"]["data"] == data
        await asyncio.sleep(0.01)
        return "async_done"

    result = await my_async_tool("secret-123", [1, 2, 3])
    assert result == "async_done"

    completed_logs = interceptor.query(tool_name="test_async_tool", status="success")
    assert len(completed_logs) == 1
    assert completed_logs[0]["result"] == "async_done"

@pytest.mark.asyncio
async def test_async_failure_interception(temp_db) -> None:
    """Test async failures are correctly captured in logs."""
    interceptor = PreemptiveAuditInterceptor(db_path=temp_db)

    @interceptor.intercept(tool_name="failing_async_tool")
    async def bad_async_tool() -> None:
        await asyncio.sleep(0.01)
        raise RuntimeError("Async crash")

    with pytest.raises(RuntimeError, match="Async crash"):
        await bad_async_tool()

    logs = interceptor.query(tool_name="failing_async_tool")
    assert len(logs) == 2
    assert logs[0]["status"] == "preemptive_start"
    assert logs[1]["status"] == "failed"
    assert "Async crash" in logs[1]["result"]

@pytest.mark.asyncio
async def test_explicit_execution_wrappers(temp_db) -> None:
    """Test execute_sync and execute_async explicit wrappers."""
    interceptor = PreemptiveAuditInterceptor(db_path=temp_db)

    def simple_add(a: int, b: int) -> int:
        return a + b

    async def simple_sub(a: int, b: int) -> int:
        return a - b

    # Test execute_sync
    res_sync = interceptor.execute_sync(simple_add, "explicit_sync", "testing sync wrap", 10, b=20)
    assert res_sync == 30

    sync_logs = interceptor.query(tool_name="explicit_sync")
    assert len(sync_logs) == 2
    assert sync_logs[0]["status"] == "preemptive_start"
    assert sync_logs[0]["kwargs"]["a"] == 10
    assert sync_logs[0]["kwargs"]["b"] == 20
    assert sync_logs[1]["status"] == "success"
    assert sync_logs[1]["result"] == 30

    # Test execute_async
    res_async = await interceptor.execute_async(simple_sub, "explicit_async", "testing async wrap", 50, 15)
    assert res_async == 35

    async_logs = interceptor.query(tool_name="explicit_async")
    assert len(async_logs) == 2
    assert async_logs[0]["status"] == "preemptive_start"
    assert async_logs[0]["kwargs"]["a"] == 50
    assert async_logs[0]["kwargs"]["b"] == 15
    assert async_logs[1]["status"] == "success"
    assert async_logs[1]["result"] == 35

def test_recursive_sanitization() -> None:
    """Test deep parameter and result recursive sanitization of credentials."""
    interceptor = PreemptiveAuditInterceptor(db_path=None)

    complex_kwargs = {
        "user": "Alice",
        "nested": {
            "token_id": "sensitive_token",
            "normal_field": "public_data",
            "credentials": [
                {"private_key": "abcde"},
                {"public_id": "12345"}
            ]
        }
    }

    sanitized = interceptor._sanitize(complex_kwargs)
    assert sanitized["user"] == "Alice"
    assert sanitized["nested"]["token_id"] == "***"
    assert sanitized["nested"]["normal_field"] == "public_data"
    assert sanitized["nested"]["credentials"][0]["private_key"] == "***"
    assert sanitized["nested"]["credentials"][1]["public_id"] == "12345"
