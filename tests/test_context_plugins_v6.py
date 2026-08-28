import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.context_plugins_v6 import SemanticCompressionLifecyclePluginV6
from magda_agent.memory.working import MemoryEntry
from magda_agent.memory.context_engine import ContextEngine

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="Compressed summary.")
    return llm

@pytest.fixture
def mock_entries():
    return [
        MemoryEntry(content=f"Item {i}", importance=0.5, tags=["tag"], emotional_state="neutral", user_id=1)
        for i in range(5)
    ]

@pytest.mark.asyncio
async def test_bootstrap():
    plugin = SemanticCompressionLifecyclePluginV6()
    await plugin.bootstrap({"key": "value"})
    assert plugin.config == {"key": "value"}

@pytest.mark.asyncio
async def test_compact_under_limit(mock_llm, mock_entries):
    plugin = SemanticCompressionLifecyclePluginV6(llm=mock_llm)
    # limit is 10, entries len is 5
    result = await plugin.compact(mock_entries, {"limit": 10})
    assert len(result) == 5
    assert result == mock_entries
    mock_llm.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_compact_over_limit_no_llm(mock_entries):
    plugin = SemanticCompressionLifecyclePluginV6(llm=None)
    # limit is 3, entries len is 5
    result = await plugin.compact(mock_entries, {"limit": 3})
    assert len(result) == 3
    assert result == mock_entries[-3:]

@pytest.mark.asyncio
async def test_compact_over_limit_with_llm(mock_llm, mock_entries):
    plugin = SemanticCompressionLifecyclePluginV6(llm=mock_llm)
    # limit is 3, entries len is 5, will compress first 5 - 3 + 1 = 3 items
    result = await plugin.compact(mock_entries, {"limit": 3})

    # Remaining 2 items + 1 summary item = 3 total items
    assert len(result) == 3

    mock_llm.chat_completion.assert_called_once()
    assert result[0].content == "Compressed summary."
    assert result[0].tags == ["tag"]
    assert result[0].user_id == 1

    assert result[1] == mock_entries[3]
    assert result[2] == mock_entries[4]

@pytest.mark.asyncio
async def test_compact_llm_exception(mock_llm, mock_entries):
    mock_llm.chat_completion = AsyncMock(side_effect=Exception("API Error"))
    plugin = SemanticCompressionLifecyclePluginV6(llm=mock_llm)

    result = await plugin.compact(mock_entries, {"limit": 3})

    # Fallback drops oldest, keeps last 3
    assert len(result) == 3
    assert result == mock_entries[-3:]
    mock_llm.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_engine_integration():
    plugin = SemanticCompressionLifecyclePluginV6()
    engine = ContextEngine(plugins=[plugin])

    assert plugin in engine._plugins
    assert 'bootstrap' in engine.hook_registry._hooks
    assert 'compact' in engine.hook_registry._hooks

    assert plugin.before_retrieval("query", 1) == "query"
    assert plugin.after_retrieval(["context"], "query", 1) == ["context"]
