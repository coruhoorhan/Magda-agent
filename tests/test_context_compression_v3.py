import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.agents.context_compression_v3 import SubagentContextCompressorV3
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_v3_compress_context_short():
    """Test that short context is not compressed and returned as is."""
    llm_mock = MagicMock(spec=LLMClient)
    compressor = SubagentContextCompressorV3(llm=llm_mock)

    context = "Short context under limits."
    task = "Do some task."
    result = await compressor.compress_context(context, task, max_length=100)

    assert result == context
    llm_mock.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_v3_compress_context_long():
    """Test that long context is compressed by calling the LLM."""
    llm_mock = MagicMock(spec=LLMClient)
    llm_mock.chat_completion = AsyncMock(return_value="Successfully compressed context.")
    compressor = SubagentContextCompressorV3(llm=llm_mock)

    context = "A" * 3000
    task = "Find elements."
    result = await compressor.compress_context(context, task, max_length=2000)

    assert result == "Successfully compressed context."
    llm_mock.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_v3_compress_context_preserves_constraints():
    """Test that critical constraints are strictly preserved or re-injected."""
    llm_mock = MagicMock(spec=LLMClient)
    # Mock return value lacks the constraint
    llm_mock.chat_completion = AsyncMock(return_value="Compressed summary but forgot critical constraint.")
    compressor = SubagentContextCompressorV3(llm=llm_mock)

    context = "A" * 2500 + "\nYou must never expose confidential keys.\n" + "B" * 500
    task = "Complete secure upload."
    result = await compressor.compress_context(context, task, max_length=2000)

    assert "Compressed summary but forgot critical constraint." in result
    assert "You must never expose confidential keys." in result
    llm_mock.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_v3_compress_payload():
    """Test that payload dictionary context/messages are compressed correctly."""
    llm_mock = MagicMock(spec=LLMClient)
    llm_mock.chat_completion = AsyncMock(return_value="Compressed summary.")
    compressor = SubagentContextCompressorV3(llm=llm_mock)

    payload = {
        "task": "Perform diagnostic check",
        "context": "A" * 3000
    }
    result = await compressor.compress_payload(payload, max_length=2000)

    assert result["context"] == "Compressed summary."
    assert result["task"] == "Perform diagnostic check"
    llm_mock.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_v3_compress_messages_short():
    """Test that message list within limit is not compressed."""
    llm_mock = MagicMock(spec=LLMClient)
    compressor = SubagentContextCompressorV3(llm=llm_mock)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]
    result = await compressor.compress_messages(messages, task="Greet", max_messages=5)

    assert len(result) == 3
    assert result == messages
    llm_mock.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_v3_compress_messages_long():
    """Test that long message lists are summarized down to target limits."""
    llm_mock = MagicMock(spec=LLMClient)
    llm_mock.chat_completion = AsyncMock(return_value="Summary of older messages.")
    compressor = SubagentContextCompressorV3(llm=llm_mock)

    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Response 1"},
        {"role": "user", "content": "Message 2"},
        {"role": "assistant", "content": "Response 2"},
        {"role": "user", "content": "Message 3"},
        {"role": "assistant", "content": "Response 3"}
    ]

    result = await compressor.compress_messages(messages, task="Keep tracking", max_messages=2)

    # Expected: system message + 1 summary message + 2 retained user/assistant messages = 4 messages
    assert len(result) == 4
    assert result[0]["content"] == "System prompt."
    assert "[SYSTEM: Compressed Summary of Old History: Summary of older messages.]" in result[1]["content"]
    assert result[2]["content"] == "Message 3"
    assert result[3]["content"] == "Response 3"
    llm_mock.chat_completion.assert_called_once()
