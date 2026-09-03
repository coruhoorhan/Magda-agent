"""
Unit tests for Letta Virtual Context Routine Builder V1.
"""

import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.memory.letta_routine_v1 import (
        LettaRoutineBuilderV1,
        ProceduralRoutine,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "memory"
        / "letta_routine_v1.py"
    )
    spec = importlib.util.spec_from_file_location("letta_routine_v1", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    LettaRoutineBuilderV1 = module.LettaRoutineBuilderV1
    ProceduralRoutine = module.ProceduralRoutine


class MockProceduralMemory:
    def __init__(self):
        self.saved_snippets = []

    def save_snippet(self, name, code, description, tags):
        self.saved_snippets.append({
            "name": name,
            "code": code,
            "description": description,
            "tags": tags,
        })


class TestLettaRoutineV1(unittest.TestCase):
    def setUp(self):
        self.mock_procedural = MockProceduralMemory()
        self.builder = LettaRoutineBuilderV1(
            procedural_memory_target=self.mock_procedural,
            min_pattern_occurrences=2,
        )

    def test_extract_routines_from_episodic_records(self):
        # 2 episodic records matching git_sync_flow
        records = [
            {"content": "Agent performed git add and git commit on subagent worktree"},
            {"content": "Executed git sync and git rebase against main"},
            {"content": "Irrelevant conversation message without pattern"},
        ]

        routines = self.builder.extract_routines_from_records(records)

        self.assertEqual(len(routines), 1)
        rtn = routines[0]
        self.assertEqual(rtn.name, "git_sync_flow")
        self.assertGreaterEqual(rtn.frequency_count, 2)
        self.assertGreater(len(rtn.steps), 0)
        self.assertIn("git", rtn.tags)

    def test_sync_to_procedural_memory(self):
        records = [
            {"content": "Ran unit tests with pytest and verified smoke test report"},
            {"content": "Executed test verification and syntax check on code"},
        ]

        routines, synced_count = self.builder.process_episodic_context(records)

        self.assertEqual(len(routines), 1)
        self.assertEqual(synced_count, 1)
        self.assertEqual(len(self.mock_procedural.saved_snippets), 1)
        saved = self.mock_procedural.saved_snippets[0]
        self.assertEqual(saved["name"], "test_verification_flow")
        self.assertIn("testing", saved["tags"])

    def test_custom_tool_sequence_extraction(self):
        records = [
            {"tool_name": "fetch_url"},
            {"tool_name": "parse_html"},
            {"tool_name": "fetch_url"},
            {"tool_name": "parse_html"},
        ]

        routines = self.builder.extract_routines_from_records(records)
        custom_routines = [r for r in routines if "auto_extracted" in r.tags]

        self.assertEqual(len(custom_routines), 1)
        self.assertEqual(custom_routines[0].name, "routine_fetch_url_parse_html")
        self.assertEqual(custom_routines[0].steps, ["Call fetch_url", "Call parse_html"])

    def test_no_routine_if_threshold_not_met(self):
        records = [
            {"content": "One single mention of mcp tool call"},
        ]

        routines = self.builder.extract_routines_from_records(records)
        self.assertEqual(len(routines), 0)


if __name__ == "__main__":
    unittest.main()
