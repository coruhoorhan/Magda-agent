"""
Unit tests for OpenClaw Canvas Live Reflection Tracing V2.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.telemetry.canvas_reflection_v2 import (
        CanvasReflectionEventV2,
        CanvasReflectionExporterV2,
        ReflectionTypeV2,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "telemetry"
        / "canvas_reflection_v2.py"
    )
    spec = importlib.util.spec_from_file_location("canvas_reflection_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    CanvasReflectionEventV2 = module.CanvasReflectionEventV2
    CanvasReflectionExporterV2 = module.CanvasReflectionExporterV2
    ReflectionTypeV2 = module.ReflectionTypeV2


class TestCanvasReflectionV2(unittest.TestCase):
    def test_v2_payload_envelope_structure(self):
        event = CanvasReflectionEventV2(
            reflection_type=ReflectionTypeV2.COUNTERFACTUAL_REASONING.value,
            agent_id="test_agent_v2",
            content="If the network timeout had occurred, fallback routine would engage",
            sentiment_score=0.35,
            pad_vector={"pleasure": 0.5, "arousal": 0.2, "dominance": 0.7},
            cognitive_load=0.65,
            attention_focus=["network_layer", "timeout_recovery"],
        )

        envelope = event.to_canvas_v2_envelope()

        self.assertEqual(envelope["type"], "canvas_reflection_event_v2")
        self.assertEqual(envelope["channel"], "canvas_reflection_v2")

        data = envelope["data"]
        self.assertEqual(data["reflection_type"], "counterfactual_reasoning")
        self.assertEqual(data["agent_id"], "test_agent_v2")
        self.assertEqual(data["content"], "If the network timeout had occurred, fallback routine would engage")
        self.assertEqual(data["sentiment_score"], 0.35)
        self.assertEqual(data["cognitive_load"], 0.65)
        self.assertEqual(data["attention_focus"], ["network_layer", "timeout_recovery"])
        self.assertEqual(data["pad_vector"]["dominance"], 0.7)

    def test_async_websocket_export(self):
        async def run_async():
            mock_ws = MagicMock()
            mock_ws.send_text = AsyncMock()

            exporter = CanvasReflectionExporterV2(
                websocket=mock_ws,
                agent_id="magda_agent_v2",
            )

            event = await exporter.export_reflection_async(
                content="Plan re-evaluation complete; all subtasks satisfied",
                reflection_type=ReflectionTypeV2.PLAN_REEVALUATION,
                sentiment_score=0.9,
            )

            self.assertEqual(len(exporter.get_exported_events()), 1)
            mock_ws.send_text.assert_called_once()

            sent_msg = json.loads(mock_ws.send_text.call_args[0][0])
            self.assertEqual(sent_msg["type"], "canvas_reflection_event_v2")
            self.assertEqual(sent_msg["data"]["reflection_type"], "plan_reevaluation")

        asyncio.run(run_async())

    def test_dispatch_handler_and_batch_streaming(self):
        async def run_async():
            mock_handler = AsyncMock()
            exporter = CanvasReflectionExporterV2(dispatch_handler=mock_handler)

            e1 = CanvasReflectionEventV2(content="Step 1 reflection")
            e2 = CanvasReflectionEventV2(content="Step 2 reflection")

            count = await exporter.stream_batch_async([e1, e2])
            self.assertEqual(count, 2)
            self.assertEqual(mock_handler.call_count, 2)
            self.assertEqual(len(exporter.get_exported_events()), 2)

        asyncio.run(run_async())


if __name__ == "__main__":
    unittest.main()
