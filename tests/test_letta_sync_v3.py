import pytest
import asyncio
import json
from unittest.mock import patch, mock_open, AsyncMock
from magda_agent.memory.letta_sync_v3 import LettaVirtualContextSyncV3

@pytest.fixture
def sync_plugin():
    return LettaVirtualContextSyncV3(sync_path="dummy_path.jsonl")

@pytest.mark.asyncio
async def test_bootstrap(sync_plugin):
    config = {"sync_path": "new_dummy_path.jsonl"}
    await sync_plugin.bootstrap(config)
    assert sync_plugin.sync_path == "new_dummy_path.jsonl"

def test_on_context_update_no_procedural_memory(sync_plugin):
    with patch("magda_agent.memory.letta_sync_v3.aiofiles.open", new_callable=AsyncMock) as mock_file:
        sync_plugin.on_context_update({"other_memory": "value"}, user_id=1)
        mock_file.assert_not_called()

def test_on_context_update_not_dict(sync_plugin):
    with patch("magda_agent.memory.letta_sync_v3.aiofiles.open", new_callable=AsyncMock) as mock_file:
        sync_plugin.on_context_update("just a string", user_id=1)
        mock_file.assert_not_called()

@pytest.mark.asyncio
async def test_sync_to_disk(sync_plugin):
    procedural_data = {"key": "value", "skill": "coding"}
    user_id = 42

    mocked_file = AsyncMock()

    with patch("magda_agent.memory.letta_sync_v3.aiofiles.open", return_value=mocked_file) as mock_open_func:
        mocked_file.__aenter__.return_value = mocked_file

        await sync_plugin._sync_to_disk(procedural_data, user_id)

        mock_open_func.assert_called_once_with("dummy_path.jsonl", mode="a", encoding="utf-8")
        mocked_file.write.assert_called_once()

        write_call_args = mocked_file.write.call_args[0][0]
        written_data = json.loads(write_call_args.strip())

        assert written_data["user_id"] == user_id
        assert written_data["data"] == procedural_data
        assert "timestamp" in written_data

@pytest.mark.asyncio
async def test_protocol_methods(sync_plugin):
    # Just checking they don't crash and return expected pass-through
    assert await sync_plugin.ingest("content", {}) == "content"
    assert await sync_plugin.assemble([1, 2], {}) == "1\n2"
    assert await sync_plugin.compact([1, 2], {}) == [1, 2]
    assert sync_plugin.before_retrieval("query", 1) == "query"
    assert sync_plugin.after_retrieval([1], "query", 1) == [1]
    assert sync_plugin.before_write("ctx", 1) == "ctx"
    assert sync_plugin.after_write("ctx", 1) is None
    assert await sync_plugin.pre_process("content", {}) == "content"
    assert await sync_plugin.post_process("response", {}) == "response"
