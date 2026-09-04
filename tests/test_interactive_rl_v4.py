"""
Unit tests for OpenClaw-RL Interactive Learning Integration V4.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.learning.interactive_rl_v4 import (
        InteractiveSignal,
        OpenClawInteractiveRLV4,
        SkillWeightProfileV4,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "learning"
        / "interactive_rl_v4.py"
    )
    spec = importlib.util.spec_from_file_location("interactive_rl_v4", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    InteractiveSignal = module.InteractiveSignal
    OpenClawInteractiveRLV4 = module.OpenClawInteractiveRLV4
    SkillWeightProfileV4 = module.SkillWeightProfileV4


class TestOpenClawInteractiveRLV4(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.learner = OpenClawInteractiveRLV4(
            llm_client=self.mock_llm,
            learning_rate=0.2,
            initial_weights={"web_search": 1.0, "calculator": 1.0, "file_writer": 1.0},
        )

    def test_mock_llm_signal_analysis(self):
        async def run_async():
            self.mock_llm.generate = AsyncMock(return_value=json.dumps({
                "reward": 0.85,
                "sentiment": "positive",
            }))

            reward, sentiment = await self.learner.analyze_signal_async("That solution is amazing and clean!")
            self.assertEqual(reward, 0.85)
            self.assertEqual(sentiment, "positive")

        asyncio.run(run_async())

    def test_process_interaction_positive_update(self):
        async def run_async():
            self.mock_llm.generate = AsyncMock(return_value=json.dumps({
                "reward": 0.5,
                "sentiment": "positive",
            }))

            init_w = self.learner.get_skill_weight("calculator")
            await self.learner.process_interaction_async("calculator", "Good calculation, thank you")

            new_w = self.learner.get_skill_weight("calculator")
            self.assertGreater(new_w, init_w)
            self.assertEqual(new_w, init_w + (0.2 * 0.5))

        asyncio.run(run_async())

    def test_process_interaction_negative_update(self):
        async def run_async():
            self.mock_llm.generate = AsyncMock(return_value=json.dumps({
                "reward": -0.5,
                "sentiment": "negative",
            }))

            init_w = self.learner.get_skill_weight("file_writer")
            await self.learner.process_interaction_async("file_writer", "No, this wrote the wrong data and broke")

            new_w = self.learner.get_skill_weight("file_writer")
            self.assertLess(new_w, init_w)
            self.assertEqual(new_w, init_w - (0.2 * 0.5))

        asyncio.run(run_async())

    def test_weight_clamping(self):
        async def run_async():
            self.mock_llm.generate = AsyncMock(return_value=json.dumps({
                "reward": -1.0,
                "sentiment": "negative",
            }))

            # Apply large number of penalties
            for _ in range(10):
                await self.learner.process_interaction_async("web_search", "bad failed error")

            # Must not drop below min_weight (0.05)
            self.assertGreaterEqual(self.learner.get_skill_weight("web_search"), 0.05)

        asyncio.run(run_async())

    def test_trajectory_batch_discounting(self):
        async def run_async():
            # Heuristic analysis without LLM
            learner_no_llm = OpenClawInteractiveRLV4(llm_client=None, learning_rate=0.1)

            trajectory = [
                ("step_1_plan", "okay let us see"),
                ("step_2_code", "good job on the code"),
                ("step_3_verify", "verified and passed, excellent!"),
            ]

            rewards = await learner_no_llm.process_trajectory_batch_async(trajectory)
            self.assertIn("step_3_verify", rewards)
            self.assertIn("step_1_plan", rewards)
            self.assertGreater(learner_no_llm.get_skill_weight("step_3_verify"), 1.0)

        asyncio.run(run_async())

    def test_skill_ranking(self):
        self.learner.profile.weights = {
            "skill_high": 3.5,
            "skill_low": 0.5,
            "skill_mid": 1.8,
        }

        ranked = self.learner.rank_skills(["skill_low", "skill_high", "skill_mid"])
        self.assertEqual(ranked[0][0], "skill_high")
        self.assertEqual(ranked[1][0], "skill_mid")
        self.assertEqual(ranked[2][0], "skill_low")


if __name__ == "__main__":
    unittest.main()
