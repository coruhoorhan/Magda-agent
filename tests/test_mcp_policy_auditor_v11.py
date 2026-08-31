import pytest
import asyncio
from unittest.mock import MagicMock
from magda_agent.safety.mcp_policy_auditor_v11 import MCPPolicyAuditorV11
from magda_agent.safety.audit_trail import AuditTrail
from magda_agent.safety.mcp_audit_v1 import MCPAuditTrailV1

@pytest.fixture
def memory_audit_trail():
    # Use db_path=None to keep it entirely in-memory
    trail = AuditTrail(db_path=None)
    trail.log_call("tool_allow", {"arg": 1}, "just because", "Success", 0.1)
    trail.log_call("tool_block", {"arg": 2}, "policy violation", "Blocked", 0.2)
    trail.log_call("tool_block", {"arg": 3}, "testing", {"status": "error", "message": "denied"}, 0.1)
    return trail

@pytest.fixture
def memory_mcp_audit_trail():
    trail = MCPAuditTrailV1(db_path=None)
    return trail

def test_auditor_with_audit_trail(memory_audit_trail):
    auditor = MCPPolicyAuditorV11(audit_trail=memory_audit_trail)
    report = auditor.audit_blocked_calls()

    assert "**Total Blocked Calls:** 2" in report
    assert "- **tool_block**: 2 blocked attempt(s)" in report
    assert "tool_allow" not in report

@pytest.mark.asyncio
async def test_auditor_with_mcp_audit_trail(memory_mcp_audit_trail):
    await memory_mcp_audit_trail.log_mcp_invocation("server1", "mcp_tool_1", {}, "ok", 0.1, "success")
    await memory_mcp_audit_trail.log_mcp_invocation("server1", "mcp_tool_2", {}, "denied", 0.2, "error")
    await memory_mcp_audit_trail.log_mcp_invocation("server2", "mcp_tool_2", {}, "failed", 0.3, "blocked")

    auditor = MCPPolicyAuditorV11(audit_trail=memory_mcp_audit_trail)
    report = auditor.audit_blocked_calls()

    assert "**Total Blocked Calls:** 2" in report
    assert "- **mcp_tool_2**: 2 blocked attempt(s)" in report
    assert "mcp_tool_1" not in report

def test_auditor_no_blocked_calls():
    trail = AuditTrail(db_path=None)
    trail.log_call("tool_allow", {"arg": 1}, "just because", "Success", 0.1)

    auditor = MCPPolicyAuditorV11(audit_trail=trail)
    report = auditor.audit_blocked_calls()

    assert "No blocked calls found." in report
    assert "Total Blocked Calls" not in report

def test_auditor_unsupported_trail():
    # Use a dummy object that lacks both get_all and get_mcp_logs
    dummy_trail = MagicMock()
    del dummy_trail.get_all
    del dummy_trail.get_mcp_logs

    auditor = MCPPolicyAuditorV11(audit_trail=dummy_trail)
    with pytest.raises(ValueError, match="Unsupported audit trail type"):
        auditor.audit_blocked_calls()
