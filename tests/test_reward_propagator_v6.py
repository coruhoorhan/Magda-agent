"""
Tests for OpenClaw-RL Online Reward Propagator v6.
"""

import threading
import unittest

try:
    from magda_agent.learning.reward_propagator_v6 import (
        OpenClawRewardPropagatorV6,
        PADVector,
        ExecutionNode,
        ImplicitFeedbackParser,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "learning" / "reward_propagator_v6.py"
    spec = importlib.util.spec_from_file_location("reward_propagator_v6", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    OpenClawRewardPropagatorV6 = module.OpenClawRewardPropagatorV6
    PADVector = module.PADVector
    ExecutionNode = module.ExecutionNode
    ImplicitFeedbackParser = module.ImplicitFeedbackParser


class TestRewardPropagatorV6(unittest.TestCase):
    """
    Test suite verifying hierarchical online reward propagation,
    implicit feedback parsing, and subagent weight modulation.
    """

    def setUp(self):
        self.propagator = OpenClawRewardPropagatorV6(learning_rate=0.2, discount_factor=0.8)

    # -------------------------------------------------------------------------
    # 1. Implicit Feedback & Rating Parsing
    # -------------------------------------------------------------------------
    def test_implicit_feedback_parsing_positive(self):
        """Positive implicit phrases should yield positive pleasure and scalar rewards."""
        positive_replies = [
            "Great job, verified and works perfectly! 👍",
            "Awesome, thanks for fixing the bug.",
            "LGTM, clean and precise.",
        ]

        for reply in positive_replies:
            pad = ImplicitFeedbackParser.parse_reply(reply)
            self.assertGreater(pad.pleasure, 0.0)
            self.assertGreater(pad.to_scalar_reward(), 0.0)

    def test_implicit_feedback_parsing_negative(self):
        """Negative implicit phrases should yield negative pleasure and scalar rewards."""
        negative_replies = [
            "This is completely wrong and broken. 👎",
            "Failed with error, please redo.",
            "Horrible, not working.",
        ]

        for reply in negative_replies:
            pad = ImplicitFeedbackParser.parse_reply(reply)
            self.assertLess(pad.pleasure, 0.0)
            self.assertLess(pad.to_scalar_reward(), 0.0)

    def test_explicit_rating_commands(self):
        """Explicit /rate commands should map accurately."""
        pad_high = ImplicitFeedbackParser.parse_reply("/rate 5")
        self.assertEqual(pad_high.pleasure, 1.0)
        self.assertGreater(pad_high.to_scalar_reward(), 0.0)

        pad_low = ImplicitFeedbackParser.parse_reply("/rate 1")
        self.assertEqual(pad_low.pleasure, -1.0)
        self.assertLess(pad_low.to_scalar_reward(), 0.0)

    # -------------------------------------------------------------------------
    # 2. Hierarchical Propagation & Discounting
    # -------------------------------------------------------------------------
    def test_hierarchical_reward_propagation_with_discount(self):
        """Reward should propagate through parent -> child -> grandchild with exponential decay."""
        root = self.propagator.record_action(
            action_id="root_act",
            agent_id="Orchestrator",
            skill_name="decompose_task",
        )
        child = self.propagator.record_action(
            action_id="child_act",
            agent_id="CoderSubagent",
            skill_name="python_code_generator",
            parent_action_id="root_act",
        )
        grandchild = self.propagator.record_action(
            action_id="grandchild_act",
            agent_id="TesterSubagent",
            skill_name="unit_test_executor",
            parent_action_id="child_act",
        )

        self.assertEqual(root.depth, 0)
        self.assertEqual(child.depth, 1)
        self.assertEqual(grandchild.depth, 2)

        # Propagate positive feedback
        report = self.propagator.propagate_feedback(
            user_reply="Perfect, all tests passing!",
            root_action_id="root_act",
        )

        self.assertEqual(report["updated_nodes_count"], 3)
        self.assertGreater(report["scalar_reward"], 0.0)

        # Root should receive full reward; child discounted by gamma; grandchild by gamma^2
        nodes_map = {n["action_id"]: n for n in report["nodes"]}
        r_root = nodes_map["root_act"]["effective_reward"]
        r_child = nodes_map["child_act"]["effective_reward"]
        r_grandchild = nodes_map["grandchild_act"]["effective_reward"]

        self.assertGreater(r_root, r_child)
        self.assertGreater(r_child, r_grandchild)
        self.assertAlmostEqual(r_child, r_root * 0.8, places=4)
        self.assertAlmostEqual(r_grandchild, r_root * 0.8 * 0.8, places=4)

        # Agent weights should be increased
        self.assertGreater(self.propagator.get_agent_weight("CoderSubagent"), 1.0)

    # -------------------------------------------------------------------------
    # 3. Negative Feedback Weight Degradation
    # -------------------------------------------------------------------------
    def test_negative_feedback_reduces_weights(self):
        """Negative user reply should decrease weights of involved agents and skills."""
        self.propagator.record_action(
            action_id="act_bad",
            agent_id="FaultyWorker",
            skill_name="broken_skill",
        )

        self.propagator.propagate_feedback(user_reply="Broken output, failed completely.")

        agent_w = self.propagator.get_agent_weight("FaultyWorker")
        skill_w = self.propagator.get_skill_weight("broken_skill")

        self.assertLess(agent_w, 1.0)
        self.assertLess(skill_w, 1.0)

    # -------------------------------------------------------------------------
    # 4. Thread-Safe Concurrent Recording & Clamping
    # -------------------------------------------------------------------------
    def test_thread_safe_concurrent_action_recording(self):
        """Multiple parallel subagent threads recording actions simultaneously should be thread-safe."""
        threads = []
        for i in range(20):
            t = threading.Thread(
                target=lambda idx=i: self.propagator.record_action(
                    action_id=f"concurrent_{idx}",
                    agent_id=f"agent_{idx % 4}",
                    skill_name="concurrent_tool",
                )
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        tree = self.propagator.get_execution_tree()
        self.assertEqual(tree["total_nodes"], 20)

    def test_min_max_weight_clamping(self):
        """Repeated positive or negative rewards must be bounded by min_weight and max_weight."""
        propagator = OpenClawRewardPropagatorV6(learning_rate=2.0, min_weight=0.1, max_weight=5.0)
        propagator.record_action(action_id="act1", agent_id="TestAgent", skill_name="tool")

        # Exceed max
        for _ in range(10):
            propagator.propagate_feedback("Awesome perfect brilliant")

        self.assertLessEqual(propagator.get_agent_weight("TestAgent"), 5.0)

        # Fall below min
        for _ in range(20):
            propagator.propagate_feedback("Terrible horrible wrong broken failed")

        self.assertGreaterEqual(propagator.get_agent_weight("TestAgent"), 0.1)


if __name__ == "__main__":
    unittest.main()
