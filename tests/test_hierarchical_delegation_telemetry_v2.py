"""
Tests for Hierarchical Delegation Telemetry Exporter v2.
"""

import time
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.architecture.hierarchical_delegation_telemetry_v2 import (
        HierarchicalDelegationTelemetryExporterV2,
        AgentMessageType,
        AgentTelemetryEvent,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "architecture" / "hierarchical_delegation_telemetry_v2.py"
    spec = importlib.util.spec_from_file_location("hierarchical_delegation_telemetry_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    HierarchicalDelegationTelemetryExporterV2 = module.HierarchicalDelegationTelemetryExporterV2
    AgentMessageType = module.AgentMessageType
    AgentTelemetryEvent = module.AgentTelemetryEvent


class TestHierarchicalDelegationTelemetryExporterV2(unittest.TestCase):
    """
    Test suite verifying telemetry capture of sub-agent communication,
    topology graph generation, and live streaming for visualization tools.
    """

    def setUp(self):
        self.exporter = HierarchicalDelegationTelemetryExporterV2(session_id="test_sess_01")

    # -------------------------------------------------------------------------
    # 1. Message Recording & Raw Export
    # -------------------------------------------------------------------------
    def test_record_message_and_export(self):
        """Messages passed between agents should be stored and exported with metadata."""
        evt = self.exporter.record_message(
            source_agent_id="Orchestrator",
            target_agent_id="Coder",
            message_type=AgentMessageType.TASK_DELEGATION,
            payload={"task": "Implement feature X"},
            duration_ms=15.5,
            status="completed",
        )

        self.assertTrue(evt.event_id.startswith("evt_"))
        self.assertEqual(evt.source_agent_id, "Orchestrator")
        self.assertEqual(evt.target_agent_id, "Coder")
        self.assertEqual(evt.duration_ms, 15.5)

        exported = self.exporter.export_events()
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]["event_id"], evt.event_id)
        self.assertIn("Implement feature X", exported[0]["payload_summary"])

    # -------------------------------------------------------------------------
    # 2. Delegation Lifecycle Tracking
    # -------------------------------------------------------------------------
    def test_delegation_lifecycle_tracking(self):
        """Starting and completing a delegation task should create paired events."""
        del_id = self.exporter.record_delegation_start(
            parent_agent_id="LeadAgent",
            subagent_id="WorkerAgent1",
            task="Scrape website docs",
        )
        self.assertTrue(del_id.startswith("del_"))

        # Complete delegation
        complete_evt = self.exporter.record_delegation_complete(
            delegation_id=del_id,
            parent_agent_id="LeadAgent",
            subagent_id="WorkerAgent1",
            result="Scraped 50 pages successfully",
            duration_ms=120.0,
            status="completed",
        )

        self.assertIsNotNone(complete_evt)
        self.assertEqual(complete_evt.source_agent_id, "WorkerAgent1")
        self.assertEqual(complete_evt.target_agent_id, "LeadAgent")
        self.assertEqual(complete_evt.message_type, AgentMessageType.SUBTASK_RESULT.value)

        events = self.exporter.export_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["message_type"], AgentMessageType.TASK_DELEGATION.value)
        self.assertEqual(events[1]["message_type"], AgentMessageType.SUBTASK_RESULT.value)

    # -------------------------------------------------------------------------
    # 3. Topology Graph Export
    # -------------------------------------------------------------------------
    def test_topology_graph_export(self):
        """Topology export must contain nodes and weighted links representing communication flow."""
        self.exporter.register_agent("LeadAgent", role="orchestrator")
        self.exporter.register_agent("Coder", role="worker", parent_id="LeadAgent")
        self.exporter.register_agent("Reviewer", role="reviewer", parent_id="LeadAgent")

        # Simulate message traffic
        self.exporter.record_message("LeadAgent", "Coder", AgentMessageType.TASK_DELEGATION, "Build API")
        self.exporter.record_message("Coder", "Reviewer", AgentMessageType.INTER_AGENT_MESSAGE, "Review code")
        self.exporter.record_message("Reviewer", "LeadAgent", AgentMessageType.SUBTASK_RESULT, "Approved")

        graph = self.exporter.export_topology_graph()
        self.assertEqual(graph["session_id"], "test_sess_01")
        self.assertGreaterEqual(graph["total_nodes"], 3)
        self.assertEqual(graph["total_links"], 3)

        # Verify specific link
        coder_to_reviewer = next(
            (link for link in graph["links"] if link["source"] == "Coder" and link["target"] == "Reviewer"),
            None,
        )
        self.assertIsNotNone(coder_to_reviewer)
        self.assertEqual(coder_to_reviewer["message_count"], 1)

    # -------------------------------------------------------------------------
    # 4. Metrics Summary Aggregation
    # -------------------------------------------------------------------------
    def test_metrics_summary_aggregation(self):
        """Summary should compute totals, byte transfers, error counts, and distribution."""
        self.exporter.record_message("A", "B", AgentMessageType.TASK_DELEGATION, "Task 1", duration_ms=10.0)
        self.exporter.record_message("B", "A", AgentMessageType.SUBTASK_RESULT, "Result 1", duration_ms=20.0)
        self.exporter.record_message("A", "C", AgentMessageType.ERROR_REPORT, "Failed", duration_ms=5.0, status="error")

        summary = self.exporter.export_metrics_summary()
        self.assertEqual(summary["total_messages"], 3)
        self.assertGreater(summary["total_bytes"], 0)
        self.assertAlmostEqual(summary["average_duration_ms"], (10.0 + 20.0 + 5.0) / 3, delta=0.01)
        self.assertEqual(summary["error_count"], 1)
        self.assertAlmostEqual(summary["error_rate"], 1 / 3, delta=0.01)
        self.assertEqual(summary["message_distribution"][AgentMessageType.TASK_DELEGATION.value], 1)

    # -------------------------------------------------------------------------
    # 5. Live Streaming Subscribers
    # -------------------------------------------------------------------------
    def test_live_streaming_listeners(self):
        """Subscribers must receive real-time event updates."""
        mock_listener = MagicMock()
        self.exporter.subscribe(mock_listener)

        self.exporter.record_message("AgentX", "AgentY", AgentMessageType.STATUS_UPDATE, "Heartbeat")

        mock_listener.assert_called_once()
        received_dict = mock_listener.call_args[0][0]
        self.assertEqual(received_dict["source_agent_id"], "AgentX")
        self.assertEqual(received_dict["target_agent_id"], "AgentY")

    # -------------------------------------------------------------------------
    # 6. Bounded Queue Capacity & Filtering
    # -------------------------------------------------------------------------
    def test_bounded_event_capacity(self):
        """Exporter must enforce max_events buffer size to prevent memory leaks."""
        small_exporter = HierarchicalDelegationTelemetryExporterV2(max_events=3)
        for i in range(10):
            small_exporter.record_message("A", "B", AgentMessageType.STATE_SYNC, f"Sync {i}")

        events = small_exporter.export_events()
        self.assertEqual(len(events), 3)
        self.assertIn("Sync 9", events[-1]["payload_summary"])

    def test_export_pagination_and_time_filter(self):
        """Export limit and timestamp filtering should accurately restrict output."""
        t0 = time.time()
        for i in range(5):
            self.exporter.record_message("A", "B", AgentMessageType.STATE_SYNC, f"Msg {i}")

        limited = self.exporter.export_events(limit=2)
        self.assertEqual(len(limited), 2)

        since_t0 = self.exporter.export_events(since_timestamp=t0 - 1.0)
        self.assertEqual(len(since_t0), 5)


if __name__ == "__main__":
    unittest.main()
