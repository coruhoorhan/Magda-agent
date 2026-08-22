import pytest
from typing import List, Any
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.memory.context_engine_lifecycle import LifecyclePlugin

def dummy_retrieval_func(query: str, user_id: int) -> List[Any]:
    return [f"retrieved for: {query}"]

@pytest.mark.asyncio
async def test_context_engine_lifecycle_hooks():
    plugin = LifecyclePlugin()
    engine = ContextEngine(plugins=[plugin])

    result = engine.retrieve_context("test query", 1, dummy_retrieval_func)

    assert len(result) == 2
    assert "test query (modified by pre_retrieval hook)" in result[0]
    assert "metadata: post_retrieval hook executed for 1" in result[1]

    # Test before_write, update_context, after_write
    # We will use write_context which calls all three
    engine.write_context(["initial context"], 2)
    # The plugin does not save context globally, but we can verify it doesn't crash
    # and the logic in ContextEngine hooks runs successfully.

    pre_result = await engine.pre_process("raw content", {})
    assert pre_result == "raw content (pre_processed)"

    post_result = await engine.post_process("raw response", {})
    assert post_result == "raw response (post_processed)"
