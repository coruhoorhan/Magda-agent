import asyncio
import pytest
from unittest.mock import MagicMock

from magda_agent.safety.governance_layer import GovernanceLayer, UnauthorizedActionError


@pytest.fixture
def governance_layer():
    return GovernanceLayer()


@pytest.mark.asyncio
async def test_governance_layer_allows_action(governance_layer):
    # Register a policy rule that always allows
    def allow_rule(tool_name, kwargs):
        return True, ""
    governance_layer.register_policy_rule(allow_rule)

    # Mock tool
    mock_tool = MagicMock(return_value="success")

    result = await governance_layer.intercept_tool_call(mock_tool, "test_tool", arg1="value1")

    assert result == "success"
    mock_tool.assert_called_once_with(arg1="value1")


@pytest.mark.asyncio
async def test_governance_layer_blocks_action(governance_layer):
    # Register a policy rule that always blocks
    def block_rule(tool_name, kwargs):
        return False, "Blocked by policy"
    governance_layer.register_policy_rule(block_rule)

    # Mock tool
    mock_tool = MagicMock()

    with pytest.raises(UnauthorizedActionError, match="Action 'test_tool' blocked: Blocked by policy"):
        await governance_layer.intercept_tool_call(mock_tool, "test_tool", arg1="value1")

    # Ensure the tool was not called
    mock_tool.assert_not_called()


@pytest.mark.asyncio
async def test_governance_layer_multiple_rules(governance_layer):
    def allow_rule(tool_name, kwargs):
        return True, ""

    def block_rule(tool_name, kwargs):
        if kwargs.get("dangerous") is True:
            return False, "Dangerous action not allowed"
        return True, ""

    governance_layer.register_policy_rule(allow_rule)
    governance_layer.register_policy_rule(block_rule)

    mock_tool = MagicMock(return_value="safe_result")

    # Safe call
    result = await governance_layer.intercept_tool_call(mock_tool, "test_tool", dangerous=False)
    assert result == "safe_result"
    mock_tool.assert_called_once_with(dangerous=False)

    mock_tool.reset_mock()

    # Dangerous call
    with pytest.raises(UnauthorizedActionError, match="Dangerous action not allowed"):
        await governance_layer.intercept_tool_call(mock_tool, "test_tool", dangerous=True)

    mock_tool.assert_not_called()


@pytest.mark.asyncio
async def test_governance_layer_async_tool(governance_layer):
    def allow_rule(tool_name, kwargs):
        return True, ""
    governance_layer.register_policy_rule(allow_rule)

    async def async_tool(**kwargs):
        await asyncio.sleep(0)
        return "async_success"

    result = await governance_layer.intercept_tool_call(async_tool, "async_tool")
    assert result == "async_success"
