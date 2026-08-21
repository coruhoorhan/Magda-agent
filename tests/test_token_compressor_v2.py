import pytest
from unittest.mock import MagicMock
from magda_agent.agents.token_compressor_v2 import TokenCompressorV2

def test_compress_under_threshold():
    """Test that context is returned as is when under the token threshold."""
    mock_summarizer = MagicMock()
    compressor = TokenCompressorV2(summarizer=mock_summarizer, token_threshold=10)

    # "Hello" is 5 chars => 1 token
    context = "Hello"
    result = compressor.compress(context)

    assert result == "Hello"
    mock_summarizer.assert_not_called()

def test_compress_over_threshold():
    """Test that summarizer is called when context exceeds token threshold."""
    mock_summarizer = MagicMock(return_value="Summarized context.")
    compressor = TokenCompressorV2(summarizer=mock_summarizer, token_threshold=2)

    # "A very long context string" is 26 chars => 6 tokens (> 2)
    context = "A very long context string"
    result = compressor.compress(context)

    assert result == "Summarized context."
    mock_summarizer.assert_called_once_with(context)

def test_compress_list_of_strings():
    """Test that a list of strings is properly joined and compressed."""
    mock_summarizer = MagicMock(return_value="Summarized list.")
    compressor = TokenCompressorV2(summarizer=mock_summarizer, token_threshold=5)

    # Each word is 4 chars, joined by newlines => 24 chars => 6 tokens
    context_list = ["one!", "two!", "thre", "four", "five"]
    result = compressor.compress(context_list)

    expected_join = "one!\ntwo!\nthre\nfour\nfive"
    assert result == "Summarized list."
    mock_summarizer.assert_called_once_with(expected_join)

def test_verify_reduced_token_count_logic():
    """Verify the token count logic triggers compression correctly before spawning subagents."""
    mock_summarizer = MagicMock(return_value="Short summary")
    compressor = TokenCompressorV2(summarizer=mock_summarizer, token_threshold=1000)

    # Context that is exactly 4000 chars => 1000 tokens
    exact_limit_context = "a" * 4000
    res1 = compressor.compress(exact_limit_context)
    assert res1 == exact_limit_context
    mock_summarizer.assert_not_called()

    # Context that is 4004 chars => 1001 tokens (over limit)
    over_limit_context = "a" * 4004
    res2 = compressor.compress(over_limit_context)
    assert res2 == "Short summary"
    mock_summarizer.assert_called_once_with(over_limit_context)
