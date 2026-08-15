import pytest
from typing import Any, List, Dict
from unittest.mock import MagicMock
from magda_agent.memory.metric_plugin_v2 import MetricPluginV2
from magda_agent.safety.audit_trail import AuditTrail
from magda_agent.memory.context_engine import ContextEngine
import time

@pytest.mark.asyncio
async def test_metric_plugin_v2_assemble_metrics() -> None:
    """Test that MetricPluginV2 accurately captures token length and latency during the assemble phase."""
    mock_audit_trail = MagicMock(spec=AuditTrail)
    plugin = MetricPluginV2(audit_trail=mock_audit_trail)

    # We test the plugin directly for its assemble hook behavior
    context_items = ["item1", "item2 with some more text"]
    metadata = {"user_id": 123, "some_key": "some_value"}

    assembled_str = await plugin.assemble(context_items, metadata)

    assert assembled_str == "item1\nitem2 with some more text"

    assert mock_audit_trail.log_call.called

    call_kwargs = mock_audit_trail.log_call.call_args.kwargs
    assert call_kwargs["tool_name"] == "context_assemble"
    assert call_kwargs["why"] == "ContextEngine assemble metric logging"
    assert call_kwargs["result"] == "Assembled 2 items"

    assert "duration" in call_kwargs
    assert call_kwargs["duration"] >= 0.0

    assert "kwargs" in call_kwargs
    assert "metadata" in call_kwargs["kwargs"]
    assert call_kwargs["kwargs"]["metadata"] == metadata

    # tokens logic: "item1\nitem2 with some more text" has 6 words.
    # int(6 / 0.75) = 8
    assert call_kwargs["kwargs"]["estimated_tokens"] == 8

def test_metric_plugin_v2_estimate_tokens() -> None:
    """Test token estimation logic for MetricPluginV2."""
    plugin = MetricPluginV2()
    assert plugin._estimate_tokens("word one two three") == int(4 / 0.75) # 5

@pytest.mark.asyncio
async def test_metric_plugin_v2_empty_assemble() -> None:
    """Test the assemble phase metrics when context items list is empty."""
    mock_audit_trail = MagicMock(spec=AuditTrail)
    plugin = MetricPluginV2(audit_trail=mock_audit_trail)

    assembled_str = await plugin.assemble([], {})

    assert assembled_str == ""

    assert mock_audit_trail.log_call.called
    call_kwargs = mock_audit_trail.log_call.call_args.kwargs
    assert call_kwargs["kwargs"]["estimated_tokens"] == 0
    assert call_kwargs["result"] == "Assembled 0 items"
