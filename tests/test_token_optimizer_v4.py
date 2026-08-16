"""
Tests for TokenOptimizerV4.
"""

from unittest.mock import MagicMock
from magda_agent.agents.token_optimizer_v4 import TokenOptimizerV4


def test_token_optimizer_v4_under_limit() -> None:
    """
    Test that when the context is under the token limit, it is returned unchanged.
    """
    mock_summarizer = MagicMock(return_value="Summarized text")
    optimizer = TokenOptimizerV4(summarizer=mock_summarizer)

    context = "This is a short context."
    token_limit = 100

    result = optimizer.optimize(context, token_limit)

    assert result == context
    mock_summarizer.assert_not_called()


def test_token_optimizer_v4_over_limit() -> None:
    """
    Test that when the context is over the token limit, it is compressed using the summarizer.
    """
    mock_summarizer = MagicMock(return_value="Summarized text")
    optimizer = TokenOptimizerV4(summarizer=mock_summarizer)

    # 40 characters ~= 10 tokens
    context = "This is a slightly longer context string."
    token_limit = 5

    result = optimizer.optimize(context, token_limit)

    assert result == "Summarized text"
    mock_summarizer.assert_called_once_with(context)
