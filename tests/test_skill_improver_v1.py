"""
Unit tests for Hermes Agent Experience Skill Improvement V1.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.learning.skill_improver_v1 import (
        DistilledSkill,
        HermesSkillImproverV1,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "learning"
        / "skill_improver_v1.py"
    )
    spec = importlib.util.spec_from_file_location("skill_improver_v1", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    DistilledSkill = module.DistilledSkill
    HermesSkillImproverV1 = module.HermesSkillImproverV1


class MockSkillRegistry:
    def __init__(self):
        self.registered = {}

    def register_skill(self, name, description, code):
        self.registered[name] = {"desc": description, "code": code}


class TestSkillImproverV1(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_registry = MockSkillRegistry()
        self.improver = HermesSkillImproverV1(
            llm_client=self.mock_llm,
            skill_registry=self.mock_registry,
        )

    def test_distill_skill_with_mock_llm(self):
        async def run_async():
            mock_llm_code = (
                "```python\n"
                "def parse_csv_records(raw_text: str) -> list:\n"
                "    \"\"\"Parses comma-delimited lines into list.\"\"\"\n"
                "    return [line.split(',') for line in raw_text.strip().split('\\n')]\n"
                "```"
            )
            self.mock_llm.generate = AsyncMock(return_value=mock_llm_code)

            experience = [
                {"id": "mem_1", "content": "Parsed CSV data using line splitting"},
                {"id": "mem_2", "content": "Handled header stripping"},
            ]

            skill = await self.improver.distill_skill_from_experience_async(
                experience_records=experience,
                skill_name_hint="parse_csv_records",
            )

            self.assertEqual(skill.skill_name, "parse_csv_records")
            self.assertEqual(skill.version, 1)
            self.assertIn("def parse_csv_records", skill.code_implementation)
            self.assertEqual(len(skill.derived_from_memory_ids), 2)

            # Check registration
            self.assertIn("parse_csv_records", self.mock_registry.registered)

        asyncio.run(run_async())

    def test_refine_existing_skill(self):
        async def run_async():
            # Initial skill
            self.improver._skills["clean_text"] = DistilledSkill(
                skill_name="clean_text",
                description="Cleans whitespace",
                code_implementation="def clean_text(s): return s.strip()",
                version=1,
            )

            refined_code = (
                "```python\n"
                "def clean_text(s):\n"
                "    if not s: return ''\n"
                "    return s.strip()\n"
                "```"
            )
            self.mock_llm.generate = AsyncMock(return_value=refined_code)

            failures = [
                {"id": "err_1", "error": "AttributeError: 'NoneType' object has no attribute 'strip'"}
            ]

            refined = await self.improver.refine_existing_skill_async("clean_text", failures)

            self.assertEqual(refined.version, 2)
            self.assertEqual(refined.refinement_count, 1)
            self.assertIn("if not s: return ''", refined.code_implementation)

        asyncio.run(run_async())

    def test_fallback_heuristic_distillation(self):
        improver_no_llm = HermesSkillImproverV1(llm_client=None)

        experience = [{"id": "m1", "content": "Direct arithmetic computation"}]
        skill = improver_no_llm.distill_skill_from_experience(experience, skill_name_hint="compute_math")

        self.assertEqual(skill.skill_name, "compute_math")
        self.assertIn("def compute_math", skill.code_implementation)
        self.assertEqual(skill.version, 1)


if __name__ == "__main__":
    unittest.main()
