import time
from typing import Any, List
from unittest.mock import MagicMock
from magda_agent.memory.metric_plugin import MetricPlugin
from magda_agent.safety.audit_trail import AuditTrail
from magda_agent.memory.context_engine import ContextEngine

def test_metric_plugin_lifecycle() -> None:
    """Test that MetricPlugin hooks into ContextEngine correctly."""
    mock_audit_trail = MagicMock(spec=AuditTrail)
    plugin = MetricPlugin(audit_trail=mock_audit_trail)
    engine = ContextEngine(plugins=[plugin])

    # Base retrieval function mock
    def base_retrieval_func(query: str, user_id: int) -> List[Any]:
        # Simulate some retrieval delay
        time.sleep(0.01)
        return ["item1", "item2 with more words"]

    # Retrieve context
    retrieved = engine.retrieve_context("test query", 123, base_retrieval_func)

    assert retrieved == ["item1", "item2 with more words"]

    # Verify log_call was invoked with expected arguments
    assert mock_audit_trail.log_call.called

    # Get the arguments passed to log_call
    call_kwargs = mock_audit_trail.log_call.call_args.kwargs
    assert call_kwargs["tool_name"] == "context_retrieval"
    assert call_kwargs["why"] == "ContextEngine retrieval metric logging"
    assert call_kwargs["result"] == "Retrieved 2 items"

    # Check latency
    assert call_kwargs["duration"] >= 0.01

    # Check tokens (total 5 words -> int(5/0.75) = int(6.66) = 6)
    assert call_kwargs["kwargs"]["query"] == "test query"
    assert call_kwargs["kwargs"]["user_id"] == 123
    assert call_kwargs["kwargs"]["estimated_tokens"] == 6

def test_metric_plugin_estimate_tokens() -> None:
    """Test the token estimation logic."""
    plugin = MetricPlugin()
    assert plugin._estimate_tokens("hello world") == int(2 / 0.75)  # 2

def test_metric_plugin_empty_context() -> None:
    """Test behavior when context retrieval is empty."""
    mock_audit_trail = MagicMock(spec=AuditTrail)
    plugin = MetricPlugin(audit_trail=mock_audit_trail)

    # Simulate hook calls directly for unit testing
    user_id = 456
    query = "empty query"

    plugin.before_retrieval(query, user_id)
    plugin.after_retrieval([], query, user_id)

    call_kwargs = mock_audit_trail.log_call.call_args.kwargs
    assert call_kwargs["result"] == "Retrieved 0 items"
    assert call_kwargs["kwargs"]["estimated_tokens"] == 0
