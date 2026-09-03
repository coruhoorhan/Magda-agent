"""
Tests for A2A Distributed Telemetry V8 module.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.telemetry.a2a_distributed_v8 import A2ADistributedTelemetryV8
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "telemetry" / "a2a_distributed_v8.py"
    spec = importlib.util.spec_from_file_location("a2a_distributed_v8", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    A2ADistributedTelemetryV8 = module.A2ADistributedTelemetryV8


class TestA2ADistributedTelemetryV8(unittest.IsolatedAsyncioTestCase):
    """
    Test suite verifying distributed telemetry collection, event queuing,
    and A2A mesh network broadcasting.
    """

    async def asyncSetUp(self):
        self.mock_broadcaster = AsyncMock()
        self.telemetry = A2ADistributedTelemetryV8(broadcaster_fn=self.mock_broadcaster)

    # -------------------------------------------------------------------------
    # 1. Event Tracking & Queuing
    # -------------------------------------------------------------------------
    def test_track_event_enqueues_structured_payload(self):
        """Track event should append a structured event dictionary to the queue."""
        evt = self.telemetry.track_event(
            subagent_id="subagent_alpha",
            event_name="task_started",
            payload={"task_id": "T123", "step": "parsing"},
        )

        self.assertEqual(evt["subagent_id"], "subagent_alpha")
        self.assertEqual(evt["event_name"], "task_started")
        self.assertEqual(evt["payload"]["step"], "parsing")
        self.assertTrue(evt["event_id"].startswith("evt_"))

        queued = self.telemetry.get_queued_events()
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0]["event_id"], evt["event_id"])

    # -------------------------------------------------------------------------
    # 2. Broadcasting & Queue Clearing
    # -------------------------------------------------------------------------
    async def test_broadcast_events_dispatches_and_clears(self):
        """Broadcasting should package envelope, trigger broadcaster, and clear queue."""
        self.telemetry.track_event("agent_1", "start", {"data": 1})
        self.telemetry.track_event("agent_2", "finish", {"data": 2})

        self.assertEqual(len(self.telemetry.get_queued_events()), 2)

        envelope = await self.telemetry.broadcast_events()

        self.assertIsNotNone(envelope)
        self.assertEqual(envelope["type"], "telemetry_broadcast")
        self.assertEqual(envelope["version"], "v8")
        self.assertEqual(envelope["events_count"], 2)
        self.assertEqual(len(envelope["events"]), 2)

        self.mock_broadcaster.assert_called_once_with(envelope)
        self.assertEqual(len(self.telemetry.get_queued_events()), 0)
        self.assertEqual(len(self.telemetry.broadcast_history), 1)

    async def test_broadcast_empty_queue_is_noop(self):
        """Broadcasting an empty queue should do nothing and return None."""
        res = await self.telemetry.broadcast_events()
        self.assertIsNone(res)
        self.mock_broadcaster.assert_not_called()

    # -------------------------------------------------------------------------
    # 3. Specialized Event Broadcast Helpers
    # -------------------------------------------------------------------------
    async def test_broadcast_pad_shift(self):
        """PAD shift helper should record and broadcast emotional shifts."""
        pad_data = {"p_shift": 0.5, "a_shift": 0.2, "d_shift": 0.1}
        evt = await self.telemetry.broadcast_pad_shift(
            subagent_id="agent_empathy",
            pad_shift=pad_data,
        )

        self.assertEqual(evt["event_name"], "pad_shift")
        self.assertEqual(evt["payload"]["pad_shift"], pad_data)
        self.mock_broadcaster.assert_called_once()

    async def test_broadcast_rl_reward(self):
        """RL reward helper should record and broadcast reinforcement learning signals."""
        evt = await self.telemetry.broadcast_rl_reward(
            subagent_id="agent_rl",
            reward_signal=0.85,
            details={"source": "implicit_feedback"},
        )

        self.assertEqual(evt["event_name"], "rl_reward")
        self.assertEqual(evt["payload"]["reward_signal"], 0.85)
        self.assertEqual(evt["payload"]["details"]["source"], "implicit_feedback")
        self.mock_broadcaster.assert_called_once()

    async def test_broadcast_tool_execution(self):
        """Tool execution helper should record execution timing and status."""
        evt = await self.telemetry.broadcast_tool_execution(
            subagent_id="agent_tool",
            tool_name="bash_executor",
            arguments={"cmd": "ls"},
            result="file1.py\nfile2.py",
            success=True,
            duration_ms=45.2,
        )

        self.assertEqual(evt["event_name"], "tool_execution")
        self.assertEqual(evt["payload"]["tool_name"], "bash_executor")
        self.assertTrue(evt["payload"]["success"])
        self.assertEqual(evt["payload"]["duration_ms"], 45.2)
        self.mock_broadcaster.assert_called_once()

    def test_clear_events(self):
        """clear_events should drop pending events without broadcasting."""
        self.telemetry.track_event("a", "b", {})
        self.assertEqual(len(self.telemetry.get_queued_events()), 1)
        self.telemetry.clear_events()
        self.assertEqual(len(self.telemetry.get_queued_events()), 0)


if __name__ == "__main__":
    unittest.main()
