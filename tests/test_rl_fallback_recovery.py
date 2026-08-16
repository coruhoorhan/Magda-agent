import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.learning.rl_fallback_recovery import RLFallbackRecoveryHandler

@pytest.mark.asyncio
async def test_handle_execution_error_with_integrator() -> None:
    """
    Test that RLFallbackRecoveryHandler correctly extracts contextual features
    and calls the online_rl_integrator to process negative feedback.
    """
    mock_integrator = AsyncMock()
    handler = RLFallbackRecoveryHandler(online_rl_integrator=mock_integrator)

    exception = ValueError("Invalid parameter provided")
    action_context = "Executing some skill with invalid args"
    skill_used = "dummy_skill"
    user_id = 123

    await handler.handle_execution_error(
        exception=exception,
        action_context=action_context,
        skill_used=skill_used,
        user_id=user_id
    )

    mock_integrator.process_feedback.assert_called_once_with(
        user_reply="Execution failed with ValueError: Invalid parameter provided",
        action_context=action_context,
        user_id=user_id,
        explicit_score=0.0,
        tool_success=False,
        skill_used=skill_used
    )

@pytest.mark.asyncio
async def test_handle_execution_error_without_integrator() -> None:
    """
    Test that RLFallbackRecoveryHandler does not raise an error
    if online_rl_integrator is None.
    """
    handler = RLFallbackRecoveryHandler(online_rl_integrator=None)

    exception = KeyError("Missing key")
    action_context = "Executing some skill missing a key"
    skill_used = "another_skill"

    # This should not raise any exceptions
    await handler.handle_execution_error(
        exception=exception,
        action_context=action_context,
        skill_used=skill_used,
        user_id=None
    )
