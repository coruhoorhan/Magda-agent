import pytest
from typing import List, Any
from unittest.mock import MagicMock, AsyncMock
from magda_agent.memory.context_engine_v6 import ContextEngineV6, ContextPluginV6

class MockContextPluginV6:
    """Mock implementation of ContextPluginV6 for testing."""
    def __init__(self) -> None:
        """Initialize mock states."""
        self.before_eviction_called = False
        self.after_eviction_called = False
        self.evicted_items: List[Any] = []

    def before_eviction(self, context_items: List[Any], limit: int) -> List[Any]:
        """Mock implementation of before_eviction."""
        self.before_eviction_called = True
        if len(context_items) > limit:
            return context_items[:len(context_items) - limit]
        return []

    def after_eviction(self, evicted_items: List[Any]) -> None:
        """Mock implementation of after_eviction."""
        self.after_eviction_called = True
        self.evicted_items = evicted_items

def test_eviction_hooks_triggered() -> None:
    """Test that eviction hooks are correctly triggered when limits are exceeded."""
    plugin = MockContextPluginV6()
    engine = ContextEngineV6(plugins=[plugin])

    context_items = ["item1", "item2", "item3"]
    limit = 2

    evicted = engine.evict(context_items, limit)

    assert plugin.before_eviction_called is True
    assert plugin.after_eviction_called is True
    assert evicted == ["item1"]
    assert plugin.evicted_items == ["item1"]

def test_eviction_no_eviction_needed() -> None:
    """Test that after_eviction is not called when no items are evicted."""
    plugin = MockContextPluginV6()
    engine = ContextEngineV6(plugins=[plugin])

    context_items = ["item1", "item2"]
    limit = 5

    evicted = engine.evict(context_items, limit)

    assert plugin.before_eviction_called is True
    assert plugin.after_eviction_called is False
    assert evicted == []

def test_eviction_default_behavior() -> None:
    """Test the default eviction behavior when no plugin intercepts."""
    engine = ContextEngineV6(plugins=[])

    context_items = ["item1", "item2", "item3"]
    limit = 1

    evicted = engine.evict(context_items, limit)

    assert evicted == ["item1", "item2"]

def test_eviction_default_behavior_no_limit_exceeded() -> None:
    """Test the default eviction behavior when limit is not exceeded."""
    engine = ContextEngineV6(plugins=[])

    context_items = ["item1", "item2"]
    limit = 5

    evicted = engine.evict(context_items, limit)

    assert evicted == []

class MockItem:
    """Mock item for testing compaction fallback."""
    def __init__(self, content: str) -> None:
        """Initialize mock item with content."""
        self.content = content

@pytest.mark.asyncio
async def test_fallback_compaction() -> None:
    """Test the fallback compaction logic using an LLM mock."""
    engine = ContextEngineV6(plugins=[], llm=MagicMock())
    engine.llm.chat_completion = AsyncMock(return_value="Summarized fallback content")

    context_items = [MockItem("item1"), MockItem("item2"), MockItem("item3"), MockItem("item4")]
    limit = 2

    metadata = {"limit": limit}
    evicted = await engine.compact(context_items, metadata)
    assert len(evicted) == 3
    assert evicted[0].content == "Summarized fallback content"

@pytest.mark.asyncio
async def test_fallback_compaction_no_llm() -> None:
    """Test the fallback compaction logic when no LLM is provided."""
    engine = ContextEngineV6(plugins=[])
    context_items = ["item1", "item2", "item3", "item4"]
    limit = 2

    metadata = {"limit": limit}
    evicted = await engine.compact(context_items, metadata)
    assert len(evicted) == 3
    assert evicted[0] == "item2"
