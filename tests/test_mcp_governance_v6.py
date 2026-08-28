import pytest
from typing import Any
from unittest.mock import AsyncMock, patch

from magda_agent.operations.mcp_governance_v6 import MCPGovernanceV6


def test_governance_allows_valid_tool() -> None:
    """Tests that the governance layer allows explicitly permitted tools."""
    governance = MCPGovernanceV6(allowed_tools=["allowed_tool"])
    # Should not raise an exception
    governance.intercept_tool_execution("allowed_tool", arg1="value")


def test_governance_blocks_invalid_tool() -> None:
    """Tests that the governance layer blocks unpermitted tools."""
    governance = MCPGovernanceV6(allowed_tools=["allowed_tool"])
    with pytest.raises(RuntimeError, match="Tool blocked_tool is blocked by governance policy."):
        governance.intercept_tool_execution("blocked_tool", arg1="value")


def test_governance_blocks_denied_tool() -> None:
    """Tests that the governance layer blocks explicitly denied tools."""
    governance = MCPGovernanceV6(denied_tools=["bad_tool"])
    with pytest.raises(RuntimeError, match="Tool bad_tool is blocked by governance policy."):
        governance.intercept_tool_execution("bad_tool", arg1="value")


@pytest.mark.asyncio
async def test_mcp_client_interception() -> None:
    """Tests that the governance layer intercepts MCPClient calls."""
    from magda_agent.skills.mcp_client import MCPClient
    from magda_agent.operations.mcp_governance_v6 import MCPGovernanceV6

    client = MCPClient()
    client.governance = MCPGovernanceV6(allowed_tools=["safe_tool"])

    # Allowed tool shouldn't return a governance error.
    # It will fail with 'not found' because the remote skill isn't registered, which means it passed governance.
    result_safe = await client.execute_tool("safe_tool", arg="test")
    assert "not found" in result_safe

    # Blocked tool should return the governance error string directly from the execute_tool method
    result_unsafe = await client.execute_tool("unsafe_tool", arg="test")
    assert "Error: Tool unsafe_tool is blocked by governance policy" in result_unsafe
