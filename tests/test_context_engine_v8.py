import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict, List

from magda_agent.memory.context_engine_v8 import ContextEngineV8, ContextPluginV8
from magda_agent.architecture.context_hooks_v5 import HookRegistry

class MockPlugin:
    def __init__(self):
        self.bootstrap = AsyncMock()
        self.pre_process = AsyncMock(side_effect=lambda c, m: f"pre_{c}")
        self.ingest = AsyncMock(side_effect=lambda c, m: f"ingest_{c}")
        self.assemble = AsyncMock(side_effect=lambda items, m: "assembled_mock")
        self.compact = AsyncMock(side_effect=lambda items, m: items[:m.get('limit', 10)])
        self.post_process = AsyncMock(side_effect=lambda r, m: f"post_{r}")

@pytest.fixture
def engine():
    return ContextEngineV8()

@pytest.mark.asyncio
async def test_engine_initialization(engine):
    assert engine.hook_registry is not None
    assert isinstance(engine.hook_registry, HookRegistry)

@pytest.mark.asyncio
async def test_register_plugin_and_bootstrap(engine):
    plugin = MockPlugin()
    engine.register_plugin(plugin)

    await engine.bootstrap_all({"test": "config"})
    plugin.bootstrap.assert_called_once()
    args, kwargs = plugin.bootstrap.call_args
    assert args[0]["test"] == "config"
    assert "hook_registry" in args[0]

@pytest.mark.asyncio
async def test_pre_process(engine):
    plugin = MockPlugin()
    engine.register_plugin(plugin)

    result = await engine.pre_process("data", {"meta": "data"})
    assert result == "pre_data"
    plugin.pre_process.assert_called_once_with("data", {"meta": "data"})

@pytest.mark.asyncio
async def test_ingest(engine):
    plugin = MockPlugin()
    engine.register_plugin(plugin)

    result = await engine.ingest("data", {})
    assert result == "ingest_data"
    plugin.ingest.assert_called_once_with("data", {})

@pytest.mark.asyncio
async def test_assemble_without_plugin(engine):
    items = ["item1", "item2"]
    result = await engine.assemble(items, {})
    assert result == "item1\nitem2"

@pytest.mark.asyncio
async def test_assemble_with_plugin(engine):
    plugin = MockPlugin()
    engine.register_plugin(plugin)

    items = ["item1", "item2"]
    result = await engine.assemble(items, {})
    assert result == "assembled_mock"
    plugin.assemble.assert_called_once_with(items, {})

@pytest.mark.asyncio
async def test_compact_under_limit(engine):
    plugin = MockPlugin()
    engine.register_plugin(plugin)

    items = [1, 2, 3]
    result = await engine.compact(items, {"limit": 5})
    assert result == [1, 2, 3]
    plugin.compact.assert_called_once_with(items, {"limit": 5})

@pytest.mark.asyncio
async def test_compact_over_limit_fallback(engine):
    plugin = MockPlugin()
    # Mocking compact to return same items to trigger fallback
    plugin.compact = AsyncMock(side_effect=lambda items, m: items)
    engine.register_plugin(plugin)

    items = [1, 2, 3, 4, 5]
    result = await engine.compact(items, {"limit": 3})
    assert len(result) == 3
    assert result == [3, 4, 5]  # Fallback keeps last `limit` items
    plugin.compact.assert_called_once_with(items, {"limit": 3})

@pytest.mark.asyncio
async def test_post_process(engine):
    plugin = MockPlugin()
    engine.register_plugin(plugin)

    result = await engine.post_process("response", {})
    assert result == "post_response"
    plugin.post_process.assert_called_once_with("response", {})
