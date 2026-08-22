import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.token_optimizer_v2 import ClaudeTokenContextOptimizerV2

@pytest.fixture
def mock_sync_summarizer():
    return MagicMock(return_value="summarized")

@pytest.fixture
def mock_async_summarizer():
    return AsyncMock(return_value="summarized_async")

@pytest.mark.asyncio
async def test_empty_context():
    optimizer = ClaudeTokenContextOptimizerV2(summarizer=lambda x: "summary", max_tokens=100)
    result = await optimizer.optimize_context([])
    assert result == []

@pytest.mark.asyncio
async def test_under_token_limit(mock_sync_summarizer):
    optimizer = ClaudeTokenContextOptimizerV2(summarizer=mock_sync_summarizer, max_tokens=100)
    context = ["a" * 100, "b" * 100]  # 25 + 25 = 50 tokens
    result = await optimizer.optimize_context(context)

    assert result == context
    mock_sync_summarizer.assert_not_called()

@pytest.mark.asyncio
async def test_over_token_limit_sync_summarizer(mock_sync_summarizer):
    optimizer = ClaudeTokenContextOptimizerV2(summarizer=mock_sync_summarizer, max_tokens=50)
    # Token sizes approx: 100/4 = 25. Three items = 75 tokens > 50 max_tokens
    context = ["a" * 100, "b" * 100, "c" * 100]
    result = await optimizer.optimize_context(context)

    assert mock_sync_summarizer.called
    assert result[0] == "[Compressed Summary] summarized"
    # Compression makes the first item 29 chars / 4 = 7 tokens.
    # Total becomes 7 + 25 + 25 = 57 > 50, so second item compressed too.
    assert result[1] == "[Compressed Summary] summarized"
    assert result[2] == "c" * 100

@pytest.mark.asyncio
async def test_over_token_limit_async_summarizer(mock_async_summarizer):
    optimizer = ClaudeTokenContextOptimizerV2(summarizer=mock_async_summarizer, max_tokens=50)
    # Token sizes approx: 100/4 = 25. Three items = 75 tokens > 50 max_tokens
    context = ["a" * 100, "b" * 100, "c" * 100]
    result = await optimizer.optimize_context(context)

    assert mock_async_summarizer.called
    assert result[0] == "[Compressed Summary] summarized_async"
    assert result[1] == "[Compressed Summary] summarized_async"
    assert result[2] == "c" * 100
