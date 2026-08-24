import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from magda_agent.memory.context_hooks_v9 import SemanticMemoryFilterPlugin
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.architecture.context_hooks_v5 import HookRegistry

@pytest.fixture
def current_time():
    return time.time()

@pytest.fixture
def plugin():
    return SemanticMemoryFilterPlugin(max_age_seconds=100)

def test_plugin_initialization(plugin):
    assert plugin.max_age_seconds == 100

@pytest.mark.asyncio
async def test_plugin_bootstrap(plugin):
    config = {"max_age_seconds": 200}
    await plugin.bootstrap(config)
    assert plugin.max_age_seconds == 200

def test_after_retrieval_filters_old_memories(plugin, current_time):
    # Set current time using a mock
    with patch('time.time', return_value=current_time):
        context = [
            # Recent memory
            {"id": "1", "text": "fact 1", "metadata": {"timestamp": current_time - 50}},
            # Old memory (should be filtered)
            {"id": "2", "text": "fact 2", "metadata": {"timestamp": current_time - 150}},
            # Boundary memory (exact max age - kept, since age > max_age_seconds is the filter condition)
            {"id": "3", "text": "fact 3", "metadata": {"timestamp": current_time - 100}},
            # Very recent memory
            {"id": "4", "text": "fact 4", "metadata": {"timestamp": current_time - 1}},
        ]

        filtered = plugin.after_retrieval(context, "query", 1)

        assert len(filtered) == 3
        ids = [item["id"] for item in filtered]
        assert "1" in ids
        assert "2" not in ids
        assert "3" in ids
        assert "4" in ids

def test_after_retrieval_keeps_non_dict_items(plugin, current_time):
    with patch('time.time', return_value=current_time):
        context = [
            "Just a string",
            {"id": "1", "text": "No metadata"},
            {"id": "2", "text": "Empty metadata", "metadata": {}},
            {"id": "3", "text": "Old memory", "metadata": {"timestamp": current_time - 150}},
            # Using an object instead of dict
            type("MemoryObj", (object,), {"id": "4", "text": "Not a dict"})()
        ]

        filtered = plugin.after_retrieval(context, "query", 1)

        assert len(filtered) == 4
        # The old dictionary memory should be filtered out
        assert not any(isinstance(i, dict) and i.get("id") == "3" for i in filtered)

def test_after_retrieval_handles_invalid_timestamps(plugin, current_time):
    with patch('time.time', return_value=current_time):
        context = [
            {"id": "1", "metadata": {"timestamp": "invalid"}},
            {"id": "2", "metadata": {"timestamp": None}},
        ]

        filtered = plugin.after_retrieval(context, "query", 1)

        # Invalid timestamps should be kept (or ignored by filter)
        assert len(filtered) == 2

def test_context_engine_integration(plugin, current_time):
    engine = ContextEngine(plugins=[plugin])

    # Mock retrieval function
    def mock_retrieve(query, user_id):
        return [
            {"id": "recent", "metadata": {"timestamp": current_time - 10}},
            {"id": "old", "metadata": {"timestamp": current_time - 200}}
        ]

    with patch('time.time', return_value=current_time):
        result = engine.retrieve_context("test query", 1, mock_retrieve)

        assert len(result) == 1
        assert result[0]["id"] == "recent"
