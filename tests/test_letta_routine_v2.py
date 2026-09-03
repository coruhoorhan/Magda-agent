"""
Unit tests for Letta Virtual Context Routine Builder V2.
"""

import asyncio
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.memory.letta_routine_v2 import (
        LettaRoutineBuilderHookV2,
        ProceduralRoutineV2,
        RoutineStepV2,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "memory"
        / "letta_routine_v2.py"
    )
    spec = importlib.util.spec_from_file_location("letta_routine_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    LettaRoutineBuilderHookV2 = module.LettaRoutineBuilderHookV2
    ProceduralRoutineV2 = module.ProceduralRoutineV2
    RoutineStepV2 = module.RoutineStepV2


class MockProceduralMemoryV2:
    def __init__(self):
        self.snippets = []

    def save_snippet(self, name, code, description, tags):
        self.snippets.append({
            "name": name,
            "code": code,
            "description": description,
            "tags": tags,
        })


class TestLettaRoutineV2(unittest.TestCase):
    def setUp(self):
        self.mock_memory = MockProceduralMemoryV2()
        self.hook = LettaRoutineBuilderHookV2(
            procedural_memory_target=self.mock_memory,
            min_sequence_length=2,
        )

    def test_extract_routine_from_stream(self):
        stream_events = [
            {"action": "tool_call", "tool_name": "fetch_data", "params": {"query": "stats"}},
            {"action": "tool_call", "tool_name": "transform_data", "params": {"format": "json"}},
            {"action": "tool_call", "tool_name": "save_database", "params": {"table": "metrics"}},
        ]

        routines = self.hook.extract_routines_from_stream(stream_events)

        self.assertEqual(len(routines), 1)
        rtn = routines[0]
        self.assertEqual(rtn.name, "routine_fetch_data_to_save_database")
        self.assertEqual(rtn.version, "v2")
        self.assertEqual(len(rtn.action_sequence), 3)

        # Check step details
        self.assertEqual(rtn.action_sequence[0].step_number, 1)
        self.assertEqual(rtn.action_sequence[0].target, "fetch_data")
        self.assertEqual(rtn.action_sequence[1].target, "transform_data")
        self.assertEqual(rtn.action_sequence[2].target, "save_database")

    def test_sync_to_procedural_memory(self):
        stream_events = [
            {"action": "tool_call", "tool": "git_add", "kwargs": {"path": "."}},
            {"action": "tool_call", "tool": "git_commit", "kwargs": {"msg": "update"}},
        ]

        routines, synced = self.hook.execute_hook(stream_events)

        self.assertEqual(len(routines), 1)
        self.assertEqual(synced, 1)
        self.assertEqual(len(self.mock_memory.snippets), 1)

        snippet = self.mock_memory.snippets[0]
        self.assertIn("git_add", snippet["name"])
        self.assertIn("letta_v2", snippet["tags"])

    def test_async_hook_execution(self):
        async def run_async():
            stream_events = [
                {"action": "step1", "target": "auth_check"},
                {"action": "step2", "target": "session_init"},
            ]
            routines, count = await self.hook.execute_hook_async(stream_events)
            self.assertEqual(len(routines), 1)
            self.assertEqual(count, 1)

        asyncio.run(run_async())

    def test_too_short_sequence_ignored(self):
        stream_events = [
            {"action": "single_step", "target": "alone"},
        ]

        routines = self.hook.extract_routines_from_stream(stream_events)
        self.assertEqual(len(routines), 0)


if __name__ == "__main__":
    unittest.main()
