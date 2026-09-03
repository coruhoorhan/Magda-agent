"""
Unit tests for OpenClaw Context Engine Canvas Visualization V2.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.telemetry.context_canvas_v2 import (
        CanvasContextPayload,
        ContextEngineCanvasVisualizerV2,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "telemetry"
        / "context_canvas_v2.py"
    )
    spec = importlib.util.spec_from_file_location("context_canvas_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    CanvasContextPayload = module.CanvasContextPayload
    ContextEngineCanvasVisualizerV2 = module.ContextEngineCanvasVisualizerV2


class MockContextEngine:
    def __init__(self):
        self.max_context_tokens = 5000
        self.total_tokens = 1200
        self.working_memory = [
            {"id": "node_1", "content": "User preferences node", "tags": ["pref"]},
            {"id": "node_2", "content": "Architecture plan node", "tags": ["arch"]},
        ]
        self.episodic_buffer = ["e1", "e2", "e3"]
        self.active_plugins = ["semantic_clustering", "taint_isolation"]

    def get_entries(self):
        return list(self.working_memory)

    def get_clusters(self):
        return [{"topic": "architecture", "summary": "System Architecture Summary"}]

    def get_weights(self):
        return {"recency": 1.2, "semantic_similarity": 2.0, "importance": 1.5}


class TestContextCanvasV2(unittest.TestCase):
    def setUp(self):
        self.visualizer = ContextEngineCanvasVisualizerV2()
        self.mock_engine = MockContextEngine()

    def test_extract_and_serialize_mock_engine(self):
        envelope = self.visualizer.serialize_to_payload(self.mock_engine)

        self.assertEqual(envelope["type"], "context_engine_canvas_update")
        self.assertEqual(envelope["layer"], "context_visualization_v2")

        data = envelope["data"]
        self.assertEqual(data["token_budget"]["max"], 5000)
        self.assertEqual(data["token_budget"]["used"], 1200)
        self.assertEqual(data["token_budget"]["available"], 3800)
        self.assertEqual(data["token_budget"]["threshold"], 4000)

        self.assertEqual(len(data["working_memory_nodes"]), 2)
        self.assertEqual(len(data["semantic_clusters"]), 1)
        self.assertEqual(data["episodic_buffer_count"], 3)
        self.assertEqual(data["retrieval_weights"]["semantic_similarity"], 2.0)
        self.assertIn("semantic_clustering", data["active_plugins"])

    def test_schema_validation(self):
        envelope = self.visualizer.serialize_to_payload(self.mock_engine)
        is_valid, errors = self.visualizer.validate_canvas_schema(envelope)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        # Invalid envelope
        bad_envelope = {"type": "wrong_type", "data": {}}
        is_bad_valid, bad_errors = self.visualizer.validate_canvas_schema(bad_envelope)
        self.assertFalse(is_bad_valid)
        self.assertGreater(len(bad_errors), 0)

    def test_json_serialization(self):
        json_str = self.visualizer.serialize_to_json(self.mock_engine)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["data"]["token_budget"]["max"], 5000)

    def test_async_stream_dispatch(self):
        async def run_async():
            mock_broadcaster = AsyncMock()
            payload = await self.visualizer.stream_to_canvas_async(self.mock_engine, mock_broadcaster)

            mock_broadcaster.assert_called_once()
            self.assertEqual(payload["type"], "context_engine_canvas_update")

        asyncio.run(run_async())


if __name__ == "__main__":
    unittest.main()
