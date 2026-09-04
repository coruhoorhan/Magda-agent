"""
Unit tests for OpenClaw Context Engine Online RL Implementation V2.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.learning.online_rl_context_v2 import (
        ContextWeightProfile,
        FeedbackSignal,
        OnlineRLContextEngineV2,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "learning"
        / "online_rl_context_v2.py"
    )
    spec = importlib.util.spec_from_file_location("online_rl_context_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ContextWeightProfile = module.ContextWeightProfile
    FeedbackSignal = module.FeedbackSignal
    OnlineRLContextEngineV2 = module.OnlineRLContextEngineV2


class TestOnlineRLContextEngineV2(unittest.TestCase):
    def setUp(self):
        self.engine = OnlineRLContextEngineV2(
            learning_rate=0.2,
            weight_min=0.1,
            weight_max=5.0,
            baseline_weights={
                "recency": 1.0,
                "semantic_similarity": 1.5,
                "tag_overlap": 1.0,
                "importance": 1.0,
                "emotional_affinity": 0.5,
            },
        )

    def test_sentiment_parsing(self):
        pos_text = "Great job! Exactly what I was looking for, thanks!"
        self.assertGreater(self.engine.parse_feedback_sentiment(pos_text), 0.5)

        neg_text = "Wrong and broken! Misunderstood the question."
        self.assertLess(self.engine.parse_feedback_sentiment(neg_text), -0.5)

        neutral_text = "Okay let us proceed to the next step."
        self.assertEqual(self.engine.parse_feedback_sentiment(neutral_text), 0.0)

    def test_positive_feedback_increases_weights(self):
        initial_weights = self.engine.get_weights()

        retrieved_entries = [
            {"id": "doc1", "semantic_similarity": 0.9, "recency_score": 0.8, "importance": 0.7},
            {"id": "doc2", "semantic_similarity": 0.85, "recency_score": 0.7, "importance": 0.8},
        ]

        # Positive feedback
        profile, reward = self.engine.update_weights_from_feedback(
            user_feedback="Great! That fixed the issue perfectly.",
            retrieved_entries=retrieved_entries,
        )

        self.assertGreater(reward, 0.0)
        self.assertGreater(profile.semantic_similarity, initial_weights["semantic_similarity"])
        self.assertGreater(profile.recency, initial_weights["recency"])
        self.assertEqual(profile.total_updates, 1)

    def test_negative_feedback_decreases_weights(self):
        initial_weights = self.engine.get_weights()

        retrieved_entries = [
            {"id": "doc1", "semantic_similarity": 0.8, "recency_score": 0.8},
        ]

        # Negative feedback
        profile, reward = self.engine.update_weights_from_feedback(
            user_feedback="No, this is completely wrong and failed.",
            retrieved_entries=retrieved_entries,
        )

        self.assertLess(reward, 0.0)
        self.assertLess(profile.semantic_similarity, initial_weights["semantic_similarity"])
        self.assertLess(profile.recency, initial_weights["recency"])

    def test_scoring_and_ranking_context_entries(self):
        entries = [
            {"id": "entry_low", "semantic_similarity": 0.1, "recency_score": 0.1, "importance": 0.1},
            {"id": "entry_high", "semantic_similarity": 0.95, "recency_score": 0.9, "importance": 0.9},
            {"id": "entry_mid", "semantic_similarity": 0.5, "recency_score": 0.5, "importance": 0.5},
        ]

        ranked = self.engine.rank_context_entries(entries, top_k=2)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["id"], "entry_high")
        self.assertEqual(ranked[1]["id"], "entry_mid")
        self.assertIn("_rl_retrieval_score", ranked[0])

    def test_weight_clamping(self):
        # Apply extreme negative updates to test lower bound clamping
        for _ in range(20):
            self.engine.update_weights_from_feedback("bad broken error", explicit_reward=-1.0)

        weights = self.engine.get_weights()
        for k, v in weights.items():
            self.assertGreaterEqual(v, 0.1)

    def test_async_feedback_update(self):
        async def run_async():
            mock_entries = [{"id": "m1", "semantic_similarity": 0.8}]
            profile, r = await self.engine.update_weights_from_feedback_async(
                user_feedback="Awesome job!",
                retrieved_entries=mock_entries,
            )
            self.assertGreater(r, 0.0)
            self.assertGreater(profile.total_updates, 0)

        asyncio.run(run_async())


if __name__ == "__main__":
    unittest.main()
