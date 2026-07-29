"""Tests for MCP Tool Runtime Permission Scoping."""

import pytest
from unittest.mock import MagicMock, patch
import asyncio

from magda_agent.safety.mcp_permission_scoping import (
    PermissionScope,
    MCPPermissionScopingPolicy,
)
from magda_agent.safety.realtime_guardrail import MCPRealtimeGuardrailFallback


def test_permission_scope_enum() -> None:
    """Test that the PermissionScope enum has the expected values."""
    assert PermissionScope.SAFE.value == "safe"
    assert PermissionScope.SENSITIVE.value == "sensitive"
    assert PermissionScope.RESTRICTED.value == "restricted"


def test_mcp_permission_scoping_policy_safe_tool() -> None:
    """Test that a tool mapped to SAFE scope is allowed without confirmation."""
    policy = MCPPermissionScopingPolicy()
    policy.set_tool_scope("safe_tool", PermissionScope.SAFE)

    # Allow without confirmed parameter
    allow, explanation = policy.evaluate("safe_tool", arg1="test")
    assert allow is True
    assert "Allowed" in explanation or "allowed" in explanation


def test_mcp_permission_scoping_policy_unmapped_tool_defaults_to_safe() -> None:
    """Test that an unmapped tool defaults to SAFE scope."""
    policy = MCPPermissionScopingPolicy()

    allow, explanation = policy.evaluate("unknown_tool", arg1="test")
    assert allow is True
    assert "Allowed" in explanation or "allowed" in explanation


def test_mcp_permission_scoping_policy_sensitive_tool_without_confirmation() -> None:
    """Test that a SENSITIVE tool without explicit confirmation is denied."""
    policy = MCPPermissionScopingPolicy()
    policy.set_tool_scope("sensitive_tool", PermissionScope.SENSITIVE)

    allow, explanation = policy.evaluate("sensitive_tool", arg1="test")
    assert allow is False
    assert "requires explicit confirmation" in explanation
    assert "confirmed=True" in explanation


def test_mcp_permission_scoping_policy_sensitive_tool_with_false_confirmation() -> None:
    """Test that a SENSITIVE tool with confirmed=False is denied."""
    policy = MCPPermissionScopingPolicy()
    policy.set_tool_scope("sensitive_tool", PermissionScope.SENSITIVE)

    allow, explanation = policy.evaluate("sensitive_tool", arg1="test", confirmed=False)
    assert allow is False
    assert "requires explicit confirmation" in explanation


def test_mcp_permission_scoping_policy_restricted_tool_without_confirmation() -> None:
    """Test that a RESTRICTED tool without explicit confirmation is denied."""
    policy = MCPPermissionScopingPolicy()
    policy.set_tool_scope("restricted_tool", PermissionScope.RESTRICTED)

    allow, explanation = policy.evaluate("restricted_tool", arg1="test")
    assert allow is False
    assert "requires explicit confirmation" in explanation


def test_mcp_permission_scoping_policy_sensitive_tool_with_confirmation() -> None:
    """Test that a SENSITIVE tool with explicit confirmation is allowed."""
    policy = MCPPermissionScopingPolicy()
    policy.set_tool_scope("sensitive_tool", PermissionScope.SENSITIVE)

    allow, explanation = policy.evaluate("sensitive_tool", arg1="test", confirmed=True)
    assert allow is True
    assert "Allowed" in explanation or "allowed" in explanation


@pytest.mark.asyncio
async def test_integration_with_realtime_guardrail_fallback_denied() -> None:
    """Test that RealtimeGuardrail intercepts denied tools and returns a dynamic prompt."""
    policy = MCPPermissionScopingPolicy()
    policy.set_tool_scope("dangerous_action", PermissionScope.RESTRICTED)

    guardrail = MCPRealtimeGuardrailFallback(policy_layer=policy)

    async def mock_tool(arg1: str) -> str:
        return "Executed"

    success, result = await guardrail.execute_with_reprompt_fallback(
        tool_func=mock_tool,
        tool_name="dangerous_action",
        kwargs={"arg1": "test"}
    )

    assert success is False
    assert "SAFETY ALERT:" in result
    assert "blocked due to a policy violation" in result
    assert "requires explicit confirmation" in result


@pytest.mark.asyncio
async def test_integration_with_realtime_guardrail_fallback_allowed() -> None:
    """Test that RealtimeGuardrail allows confirmed tools."""
    policy = MCPPermissionScopingPolicy()
    policy.set_tool_scope("dangerous_action", PermissionScope.RESTRICTED)

    guardrail = MCPRealtimeGuardrailFallback(policy_layer=policy)

    async def mock_tool(arg1: str, confirmed: bool) -> str:
        return "Executed"

    success, result = await guardrail.execute_with_reprompt_fallback(
        tool_func=mock_tool,
        tool_name="dangerous_action",
        kwargs={"arg1": "test", "confirmed": True}
    )

    assert success is True
    assert result == "Executed"
