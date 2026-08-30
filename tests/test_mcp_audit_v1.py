import pytest
import sqlite3
import os
from magda_agent.safety.mcp_audit_v1 import MCPAuditTrailV1

@pytest.fixture
def temp_db_path(tmp_path):
    db_file = tmp_path / "test_mcp_audit.db"
    yield str(db_file)
    if db_file.exists():
        db_file.unlink()

@pytest.fixture
def mcp_audit(temp_db_path):
    audit = MCPAuditTrailV1(db_path=temp_db_path)
    yield audit
    audit.clear()

@pytest.mark.asyncio
async def test_log_and_get_mcp_logs(mcp_audit):
    # Log some invocations
    await mcp_audit.log_mcp_invocation(
        server_name="test_server_1",
        tool_name="test_tool_A",
        arguments={"arg1": "val1"},
        result={"res": "ok"},
        duration=1.2,
        status="success"
    )

    await mcp_audit.log_mcp_invocation(
        server_name="test_server_1",
        tool_name="test_tool_B",
        arguments={"arg2": "val2"},
        result={"error": "failed"},
        duration=0.5,
        status="error"
    )

    await mcp_audit.log_mcp_invocation(
        server_name="test_server_2",
        tool_name="test_tool_C",
        arguments={"arg3": "val3"},
        result="success_str",
        duration=2.1,
        status="success"
    )

    # Test get all logs
    logs = mcp_audit.get_mcp_logs()
    assert len(logs) == 3

    # Test filtering by server_name
    logs_server_1 = mcp_audit.get_mcp_logs(server_name="test_server_1")
    assert len(logs_server_1) == 2
    assert all(log["server_name"] == "test_server_1" for log in logs_server_1)

    # Test filtering by tool_name
    logs_tool_B = mcp_audit.get_mcp_logs(tool_name="test_tool_B")
    assert len(logs_tool_B) == 1
    assert logs_tool_B[0]["tool_name"] == "test_tool_B"
    assert logs_tool_B[0]["status"] == "error"

    # Test filtering by status
    logs_success = mcp_audit.get_mcp_logs(status="success")
    assert len(logs_success) == 2
    assert all(log["status"] == "success" for log in logs_success)

    # Check data integrity
    log_a = mcp_audit.get_mcp_logs(tool_name="test_tool_A")[0]
    assert log_a["arguments"] == {"arg1": "val1"}
    assert log_a["result"] == {"res": "ok"}
    assert log_a["duration"] == 1.2

@pytest.mark.asyncio
async def test_mcp_audit_in_memory_only():
    audit = MCPAuditTrailV1(db_path=None)

    await audit.log_mcp_invocation(
        server_name="mem_server",
        tool_name="mem_tool",
        arguments={"k": "v"},
        result="ok",
        duration=0.1,
        status="success"
    )

    logs = audit.get_mcp_logs(server_name="mem_server")
    assert len(logs) == 1
    assert logs[0]["tool_name"] == "mem_tool"

    audit.clear()
    assert len(audit.get_mcp_logs()) == 0

@pytest.mark.asyncio
async def test_mcp_audit_persistence(temp_db_path):
    audit1 = MCPAuditTrailV1(db_path=temp_db_path)
    await audit1.log_mcp_invocation(
        server_name="persistent_server",
        tool_name="persistent_tool",
        arguments={"k": "v"},
        result="ok",
        duration=0.1,
        status="success"
    )

    # Re-initialize with same path to test persistence
    audit2 = MCPAuditTrailV1(db_path=temp_db_path)
    logs = audit2.get_mcp_logs()

    assert len(logs) == 1
    assert logs[0]["server_name"] == "persistent_server"
    assert logs[0]["tool_name"] == "persistent_tool"
