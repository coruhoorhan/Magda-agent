"""
Unit tests for OpenClaw Canvas Live Reflection Tracing V1.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.telemetry.canvas_reflection_v1 import (
        CanvasReflectionEventPayload,
        CanvasReflectionTelemetryBroadcasterV1,
        ReflectionEventType,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "telemetry"
        / "canvas_reflection_v1.py"
    )
    spec = importlib.util.spec_from_file_location("canvas_reflection_v1", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    CanvasReflectionEventPayload = module.CanvasReflectionEventPayload
    CanvasReflectionTelemetryBroadcasterV1 = module.CanvasReflectionTelemetryBroadcasterV1
    ReflectionEventType = module.ReflectionEventType


class TestCanvasReflectionV1(unittest.TestCase):
    def test_canvas_payload_format(self):
        payload = CanvasReflectionEventPayload(
            event_type="cognitive_shift",
            agent_id="test_agent",
            subconscious_thought="Re-evaluating plan strategy due to test failure",
            confidence=0.85,
            pad_state={"pleasure": -0.2, "arousal": 0.4, "dominance": 0.1},
            memory_delta={"added_fact": "API changed in v2"},
        )

        canvas_envelope = payload.to_canvas_payload()

        self.assertEqual(canvas_envelope["type"], "canvas_reflection_event")
        self.assertEqual(canvas_envelope["layer"], "reflection_canvas")

        data = canvas_envelope["data"]
        self.assertEqual(data["event_type"], "cognitive_shift")
        self.assertEqual(data["agent_id"], "test_agent")
        self.assertEqual(data["subconscious_thought"], "Re-evaluating plan strategy due to test failure")
        self.assertEqual(data["confidence"], 0.85)
        self.assertEqual(data["pad_state"]["pleasure"], -0.2)
        self.assertEqual(data["memory_delta"]["added_fact"], "API changed in v2")

    def test_broadcaster_with_websocket(self):
        async def run_async():
            mock_ws = MagicMock()
            mock_ws.send_text = AsyncMock()

            broadcaster = CanvasReflectionTelemetryBroadcasterV1(
                websocket=mock_ws,
                agent_id="agent_alpha",
            )

            event = await broadcaster.broadcast_reflection_async(
                subconscious_thought="Detected potential circular dependency in workflow",
                event_type=ReflectionEventType.BELIEF_REVISION,
                confidence=0.92,
            )

            self.assertEqual(len(broadcaster.get_history()), 1)
            mock_ws.send_text.assert_called_once()

            # Verify JSON string sent over websocket
            sent_str = mock_ws.send_text.call_args[0][0]
            parsed = json.loads(sent_str)
            self.assertEqual(parsed["type"], "canvas_reflection_event")
            self.assertEqual(parsed["data"]["event_type"], "belief_revision")

        asyncio.run(run_async())

    def test_broadcaster_with_custom_handler(self):
        async def run_async():
            mock_handler = AsyncMock()

            broadcaster = CanvasReflectionTelemetryBroadcasterV1(
                broadcast_handler=mock_handler,
            )

            await broadcaster.broadcast_reflection_async(
                subconscious_thought="Consolidating episodic memory chunk",
                event_type=ReflectionEventType.WORKING_MEMORY_CONSOLIDATION,
            )

            mock_handler.assert_called_once()
            args = mock_handler.call_args[0][0]
            self.assertEqual(args["data"]["event_type"], "working_memory_consolidation")

        asyncio.run(run_async())

    def test_batch_broadcasting(self):
        async def run_async():
            mock_handler = AsyncMock()
            broadcaster = CanvasReflectionTelemetryBroadcasterV1(broadcast_handler=mock_handler)

            e1 = CanvasReflectionEventPayload(subconscious_thought="t1")
            e2 = CanvasReflectionEventPayload(subconscious_thought="t2")

            sent_count = await broadcaster.broadcast_batch_async([e1, e2])
            self.assertEqual(sent_count, 2)
            self.assertEqual(mock_handler.call_count, 2)
            self.assertEqual(len(broadcaster.get_history()), 2)

        asyncio.run(run_async())


if __name__ == "__main__":
    unittest.main()
