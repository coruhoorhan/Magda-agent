import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.agents.context_compression import ClaudeSubagentContextWrapper, RPCPayloadCompressor


@pytest.mark.asyncio
async def test_wrapper_skips_compression_when_within_limit():
    llm_mock = MagicMock()
    llm_mock.chat_completion = AsyncMock(return_value="Compressed context output")

    wrapper = ClaudeSubagentContextWrapper(llm=llm_mock, max_length=500)

    short_context = "This is a short context well under five hundred characters."
    payload = {"context": short_context, "task": "Analyze logs"}

    result = await wrapper.wrap_payload(payload)

    assert result["context"] == short_context
    llm_mock.chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_wrapper_compresses_large_payload_context():
    llm_mock = MagicMock()
    llm_mock.chat_completion = AsyncMock(return_value="Condensed context focusing on task. MUST follow safety rules.")

    wrapper = ClaudeSubagentContextWrapper(llm=llm_mock, max_length=50)

    long_context = "A" * 100 + " MUST follow safety rules."
    payload = {"context": long_context, "task": "Execute step"}

    result = await wrapper.wrap_payload(payload)

    assert "Condensed context" in result["context"]
    assert llm_mock.chat_completion.called


@pytest.mark.asyncio
async def test_wrapper_wrap_context_string():
    llm_mock = MagicMock()
    llm_mock.chat_completion = AsyncMock(return_value="Summarized context for task")

    wrapper = ClaudeSubagentContextWrapper(llm=llm_mock, max_length=50)

    long_context = "B" * 120
    res_str = await wrapper.wrap_context(long_context, task="Subtask 1")

    assert res_str == "Summarized context for task"


@pytest.mark.asyncio
async def test_wrapper_fallback_truncation_without_compressor():
    wrapper = ClaudeSubagentContextWrapper(llm=None, compressor=None, max_length=30)

    long_context = "1234567890" * 5
    payload = {"context": long_context, "task": "No LLM task"}

    result = await wrapper.wrap_payload(payload)

    assert len(result["context"]) == 30
    assert result["context"] == long_context[:30]


@pytest.mark.asyncio
async def test_critical_constraint_reinjection():
    llm_mock = MagicMock()
    # LLM summary accidentally omits the mandatory requirement keyword
    llm_mock.chat_completion = AsyncMock(return_value="Summarized text without rules")

    compressor = RPCPayloadCompressor(llm=llm_mock)
    wrapper = ClaudeSubagentContextWrapper(compressor=compressor, max_length=20)

    context_with_rule = "User info long text...\nYou MUST NOT delete production database."
    payload = {"context": context_with_rule, "task": "Database maintenance"}

    result = await wrapper.wrap_payload(payload)

    assert "Summarized text without rules" in result["context"]
    assert "PRESERVED CRITICAL CONSTRAINTS" in result["context"]
    assert "You MUST NOT delete production database." in result["context"]
