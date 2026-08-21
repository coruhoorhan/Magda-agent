import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from magda_agent.memory.compression_hook_v2 import CompressionHookPluginV2

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="Compressed summary of past entries.")
    return llm

@pytest.fixture
def mock_memory_system():
    mock_sys = MagicMock()

    mock_working_memory = MagicMock()
    mock_working_memory.limit = 5
    mock_working_memory.add = AsyncMock()

    mock_sys.working_memory = mock_working_memory
    return mock_sys

@pytest.mark.asyncio
async def test_compression_hook_v2_bootstrap():
    plugin = CompressionHookPluginV2()
    mock_sys = MagicMock()
    mock_llm = MagicMock()

    config = {
        "memory_system": mock_sys,
        "llm": mock_llm,
        "limit": 3
    }
    await plugin.bootstrap(config)

    assert plugin.memory_system == mock_sys
    assert plugin.llm == mock_llm
    assert plugin.limit == 3

def test_compression_hook_v2_fallback_remove(mock_memory_system):
    mock_working_memory = mock_memory_system.working_memory
    # explicitly mock the limit to ensure it takes precedence or is used
    mock_working_memory.limit = 3

    # We do not provide an LLM, so it will fallback
    plugin = CompressionHookPluginV2(memory_system=mock_memory_system, limit=3)

    # Create 4 entries
    entry1 = MagicMock(id="id1", importance=0.8, user_id=1)
    entry2 = MagicMock(id="id2", importance=0.1, user_id=1) # least important
    entry3 = MagicMock(id="id3", importance=0.5, user_id=1)
    entry4 = MagicMock(id="id4", importance=0.2, user_id=1) # second least important

    mock_working_memory.get_entries.return_value = [entry1, entry2, entry3, entry4]

    # Limit is 3, length is 4. It needs to remove length - limit + 1 = 4 - 3 + 1 = 2 entries
    result_query = plugin.before_retrieval("test query", 1)

    assert result_query == "test query"

    # We expect remove to be called 2 times
    assert mock_working_memory.remove.call_count == 2
    mock_working_memory.remove.assert_any_call("id2", 1)
    mock_working_memory.remove.assert_any_call("id4", 1)

@patch('asyncio.get_running_loop')
def test_compression_hook_v2_with_llm(mock_get_running_loop, mock_memory_system, mock_llm):
    # Setup loop mock
    mock_loop = MagicMock()
    mock_get_running_loop.return_value = mock_loop

    mock_working_memory = mock_memory_system.working_memory
    mock_working_memory.limit = 3

    plugin = CompressionHookPluginV2(memory_system=mock_memory_system, llm=mock_llm, limit=3)

    # mock _async_compress to prevent coroutine never awaited warnings
    with patch.object(plugin, '_async_compress', new_callable=MagicMock) as mock_compress:
        entry1 = MagicMock(id="id1", importance=0.8, user_id=1)
        entry2 = MagicMock(id="id2", importance=0.5, user_id=1)
        entry3 = MagicMock(id="id3", importance=0.7, user_id=1)
        entry4 = MagicMock(id="id4", importance=0.9, user_id=1)

        mock_working_memory.get_entries.return_value = [entry1, entry2, entry3, entry4]

        plugin.before_retrieval("test query", 1)

        # It should schedule a task in the running loop
        mock_loop.create_task.assert_called_once_with(mock_compress.return_value)

@pytest.mark.asyncio
async def test_compression_hook_v2_async_compress(mock_memory_system, mock_llm):
    plugin = CompressionHookPluginV2(memory_system=mock_memory_system, llm=mock_llm, limit=3)
    mock_working_memory = mock_memory_system.working_memory

    entry1 = MagicMock(id="id1", importance=0.8, user_id=1)
    entry2 = MagicMock(id="id2", importance=0.5, user_id=1)
    entry1.content = "content 1"
    entry2.content = "content 2"

    to_compress = [entry1, entry2]

    await plugin._async_compress(to_compress, mock_working_memory, 1)

    mock_llm.chat_completion.assert_called_once()

    assert mock_working_memory.remove.call_count == 2
    mock_working_memory.add.assert_called_once()

    # Verify the new summary entry
    summary_entry = mock_working_memory.add.call_args[0][0]
    assert summary_entry.content == "Compressed summary of past entries."
