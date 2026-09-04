"""
Unit tests for Agent Teams Distributed Tracing V9.
"""

import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.architecture.agent_teams_tracing_v9 import (
        AgentTeamsDistributedTracerV9,
        SpanRecord,
        TraceContext,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "architecture"
        / "agent_teams_tracing_v9.py"
    )
    spec = importlib.util.spec_from_file_location("agent_teams_tracing_v9", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    AgentTeamsDistributedTracerV9 = module.AgentTeamsDistributedTracerV9
    SpanRecord = module.SpanRecord
    TraceContext = module.TraceContext


class TestAgentTeamsDistributedTracingV9(unittest.TestCase):
    def setUp(self):
        self.tracer = AgentTeamsDistributedTracerV9(service_name="test_agent_team")

    def test_w3c_traceparent_serialization(self):
        ctx = TraceContext(
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            span_id="00f067aa0ba902b7",
            trace_flags="01",
        )

        traceparent = ctx.to_w3c_traceparent()
        self.assertEqual(traceparent, "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")

        parsed = TraceContext.from_w3c_traceparent(traceparent)
        self.assertEqual(parsed.trace_id, ctx.trace_id)
        self.assertEqual(parsed.span_id, ctx.span_id)
        self.assertEqual(parsed.trace_flags, "01")

    def test_worktree_env_injection_and_extraction(self):
        ctx = TraceContext(
            trace_id="12345678901234567890123456789012",
            span_id="1234567890123456",
            baggage={"team": "alpha"},
        )

        env = ctx.inject_env({"BASE_VAR": "val"})
        self.assertIn("TRACEPARENT", env)
        self.assertEqual(env["MAGDA_TRACE_ID"], ctx.trace_id)
        self.assertEqual(env["MAGDA_SPAN_ID"], ctx.span_id)

        # Extract on subagent side
        extracted = TraceContext.extract_from_env(env)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted.trace_id, ctx.trace_id)
        self.assertEqual(extracted.span_id, ctx.span_id)
        self.assertEqual(extracted.baggage.get("team"), "alpha")

    def test_trace_propagation_across_subagents(self):
        # 1. Root orchestrator span
        root_ctx, root_span = self.tracer.start_root_trace("team_coordinator")

        # 2. Spawn subagent 1 (coder)
        sub1_ctx, sub1_span = self.tracer.create_subagent_span(
            parent_context=root_ctx,
            agent_id="agent_coder_01",
            span_name="code_generation",
            worktree_path="/tmp/wt_coder",
        )

        # 3. Spawn subagent 2 (tester)
        sub2_ctx, sub2_span = self.tracer.create_subagent_span(
            parent_context=root_ctx,
            agent_id="agent_tester_01",
            span_name="run_tests",
            worktree_path="/tmp/wt_tester",
        )

        # Verify trace IDs match
        self.assertEqual(root_ctx.trace_id, sub1_ctx.trace_id)
        self.assertEqual(root_ctx.trace_id, sub2_ctx.trace_id)
        self.assertEqual(sub1_span.trace_id, root_span.trace_id)
        self.assertEqual(sub2_span.trace_id, root_span.trace_id)

        # Verify parent-child linkage
        self.assertEqual(sub1_span.parent_span_id, root_span.span_id)
        self.assertEqual(sub2_span.parent_span_id, root_span.span_id)

        # Close spans
        self.tracer.end_span(sub1_span.span_id, status="OK")
        self.tracer.end_span(sub2_span.span_id, status="OK")
        self.tracer.end_span(root_span.span_id, status="OK")

        # Build and verify tree
        tree = self.tracer.build_trace_tree(root_ctx.trace_id)
        self.assertEqual(tree["trace_id"], root_ctx.trace_id)
        self.assertEqual(tree["total_spans"], 3)
        self.assertEqual(len(tree["root_spans"]), 1)
        self.assertEqual(len(tree["root_spans"][0]["children"]), 2)

    def test_span_events_and_duration(self):
        _, span = self.tracer.start_root_trace("test_events")
        span.add_event("worktree_created", {"path": "/tmp/wt"})
        self.assertEqual(len(span.events), 1)

        self.tracer.end_span(span.span_id, status="OK")
        self.assertGreaterEqual(span.duration_ms, 0.0)
        self.assertEqual(span.status, "OK")


if __name__ == "__main__":
    unittest.main()
