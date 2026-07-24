import pytest
import asyncio
from magda_agent.visualization.canvas_logger_v3 import CanvasLoggerPluginV3
from magda_agent.tracing.tracer import ThoughtChainTracer

@pytest.mark.asyncio
async def test_canvas_logger_v3_lifecycle_hooks():
    plugin = CanvasLoggerPluginV3()

    # 1. Test bootstrap
    config = {"app_setting": "v3_enabled"}
    await plugin.bootstrap(config)
    assert len(plugin.logs) == 1
    assert plugin.logs[0]["event"] == "bootstrap"
    assert plugin.logs[0]["category"] == "bootstrap"
    assert plugin.logs[0]["config"] == config

    # 2. Test ingest
    content = "user query input"
    metadata = {"session": "abc"}
    ingested_res = await plugin.ingest(content, metadata)
    assert ingested_res == content
    assert len(plugin.logs) == 2
    assert plugin.logs[1]["event"] == "ingest"
    assert plugin.logs[1]["content"] == content

    # 3. Test assemble
    items = ["context_item_1", "context_item_2"]
    assembled_res = await plugin.assemble(items, metadata)
    assert assembled_res == "context_item_1\ncontext_item_2"
    assert len(plugin.logs) == 3
    assert plugin.logs[2]["event"] == "assemble"
    assert plugin.logs[2]["item_count"] == 2

    # 4. Test compact
    compacted_res = await plugin.compact(items, metadata)
    assert compacted_res == items
    assert len(plugin.logs) == 4
    assert plugin.logs[3]["event"] == "compact"
    assert plugin.logs[3]["item_count_before"] == 2


def test_canvas_logger_v3_retrieval_and_write_hooks():
    plugin = CanvasLoggerPluginV3()

    # 1. before_retrieval
    query = "search query"
    user_id = 99
    res_query = plugin.before_retrieval(query, user_id)
    assert res_query == query
    assert len(plugin.logs) == 1
    assert plugin.logs[0]["event"] == "before_retrieval"
    assert plugin.logs[0]["query"] == query
    assert plugin.logs[0]["user_id"] == user_id

    # 2. after_retrieval
    context = ["doc1", "doc2"]
    res_context = plugin.after_retrieval(context, query, user_id)
    assert res_context == context
    assert len(plugin.logs) == 2
    assert plugin.logs[1]["event"] == "after_retrieval"
    assert plugin.logs[1]["context_length"] == 2

    # 3. before_write
    write_context = "memory block"
    res_write = plugin.before_write(write_context, user_id)
    assert res_write == write_context
    assert len(plugin.logs) == 3
    assert plugin.logs[2]["event"] == "before_write"

    # 4. after_write
    plugin.after_write(write_context, user_id)
    assert len(plugin.logs) == 4
    assert plugin.logs[3]["event"] == "after_write"

    # 5. on_context_update
    plugin.on_context_update("new update", user_id)
    assert len(plugin.logs) == 5
    assert plugin.logs[4]["event"] == "on_context_update"


def test_canvas_logger_v3_manual_thoughts_and_tracer():
    tracer = ThoughtChainTracer()
    plugin = CanvasLoggerPluginV3(tracer=tracer)

    # Manual step logging
    plugin.log_thought_step("Step 1: Planning", data={"plan": "details"}, category="planning", user_id="user_123")
    assert len(plugin.logs) == 1
    assert plugin.logs[0]["event"] == "thought_step"
    assert plugin.logs[0]["step"] == "Step 1: Planning"
    assert plugin.logs[0]["category"] == "planning"
    assert plugin.logs[0]["user_id"] == "user_123"

    # Verify synchronization with tracer
    tracer_steps = tracer.get_trace()
    assert len(tracer_steps) == 1
    assert tracer_steps[0]["step"] == "Step 1: Planning"

    # Test get_logs merging with tracer
    tracer.add_step("Step 2: External action", data={"action": "details"})
    logs = plugin.get_logs(include_tracer=True)
    assert len(logs) == 3  # 1 from plugin, 2 from tracer (one synced from plugin, one added only to tracer)

    # Test formatting of the trace
    formatted = plugin.get_formatted_thought_trace()
    assert formatted["schema_version"] == "openclaw_canvas_v3"
    assert formatted["trace_length"] == 3
    assert len(formatted["steps"]) == 3

    # Test clear_logs
    plugin.clear_logs(clear_tracer=True)
    assert len(plugin.logs) == 0
    assert len(tracer.get_trace()) == 0
