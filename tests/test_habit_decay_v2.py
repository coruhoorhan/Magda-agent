"""
Unit tests for OpenClaw-RL Habit Decay Parameterization V2.
"""

import time
import unittest

try:
    from magda_agent.learning.habit_decay_v2 import (
        DecayFunctionType,
        HabitDecayConfig,
        LearnedHabitRecord,
        OpenClawHabitDecayManagerV2,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "learning"
        / "habit_decay_v2.py"
    )
    spec = importlib.util.spec_from_file_location("habit_decay_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    DecayFunctionType = module.DecayFunctionType
    HabitDecayConfig = module.HabitDecayConfig
    LearnedHabitRecord = module.LearnedHabitRecord
    OpenClawHabitDecayManagerV2 = module.OpenClawHabitDecayManagerV2


class TestHabitDecayV2(unittest.TestCase):
    def setUp(self):
        self.manager = OpenClawHabitDecayManagerV2()

    def test_exponential_decay_over_time(self):
        t0 = 1000.0
        cfg = HabitDecayConfig(
            decay_function=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.01,
            baseline_strength=1.0,
        )
        self.manager.register_habit("code_formatting", initial_strength=3.0, config=cfg, creation_time=t0)

        # At t=0 -> 3.0
        val_0 = self.manager.calculate_decayed_strength("code_formatting", current_time=t0)
        self.assertEqual(val_0, 3.0)

        # At t=100s -> delta=2.0, decay_factor=exp(-0.01 * 100) = exp(-1) = 0.367879 -> 1.0 + 2.0 * 0.367879 = 1.7358
        val_100 = self.manager.calculate_decayed_strength("code_formatting", current_time=t0 + 100.0)
        self.assertAlmostEqual(val_100, 1.7358, places=3)
        self.assertLess(val_100, 3.0)
        self.assertGreater(val_100, 1.0)

    def test_half_life_decay_simulation(self):
        t0 = 1000.0
        cfg = HabitDecayConfig(
            decay_function=DecayFunctionType.HALF_LIFE,
            half_life_seconds=3600.0,
            baseline_strength=1.0,
        )
        self.manager.register_habit("prompt_conciseness", initial_strength=5.0, config=cfg, creation_time=t0)

        # After exactly 1 half-life (3600s), excess above baseline (4.0) should halve to 2.0 -> total = 3.0
        val_halved = self.manager.calculate_decayed_strength("prompt_conciseness", current_time=t0 + 3600.0)
        self.assertAlmostEqual(val_halved, 3.0, places=3)

        # After 2 half-lives (7200s), excess (4.0) should be 1.0 -> total = 2.0
        val_two_halves = self.manager.calculate_decayed_strength("prompt_conciseness", current_time=t0 + 7200.0)
        self.assertAlmostEqual(val_two_halves, 2.0, places=3)

    def test_configurable_parameters_adjustment(self):
        t0 = 1000.0
        self.manager.register_habit("unit_test_first", initial_strength=2.0, creation_time=t0)

        # Adjust decay parameters at runtime
        self.manager.configure_habit_decay(
            "unit_test_first",
            decay_function=DecayFunctionType.LINEAR,
            decay_rate=0.005,
            baseline_strength=1.0,
        )

        habit = self.manager.get_habit("unit_test_first")
        self.assertEqual(habit.config.decay_function, DecayFunctionType.LINEAR)
        self.assertEqual(habit.config.decay_rate, 0.005)

        # At t=100s with linear decay of 0.005/s: 2.0 - (0.005 * 100) = 1.5
        val_linear = self.manager.calculate_decayed_strength("unit_test_first", current_time=t0 + 100.0)
        self.assertAlmostEqual(val_linear, 1.5, places=3)

    def test_reinforcement_feedback_boost(self):
        t0 = 1000.0
        cfg = HabitDecayConfig(reinforcement_factor=0.6, baseline_strength=1.0)
        self.manager.register_habit("defensive_checks", initial_strength=1.5, config=cfg, creation_time=t0)

        # Reinforce with positive feedback
        new_val = self.manager.reinforce_habit("defensive_checks", feedback_score=1.0, current_time=t0)
        self.assertEqual(new_val, 2.1)  # 1.5 + 0.6 * 1.0

        # Reinforce with negative feedback
        new_val2 = self.manager.reinforce_habit("defensive_checks", feedback_score=-0.5, current_time=t0)
        self.assertEqual(new_val2, 1.8)  # 2.1 - 0.6 * 0.5

    def test_decay_all_habits_batch(self):
        t0 = 1000.0
        self.manager.register_habit("h1", initial_strength=2.0, creation_time=t0)
        self.manager.register_habit("h2", initial_strength=4.0, creation_time=t0)

        decayed_dict = self.manager.decay_all_habits(current_time=t0 + 50.0)
        self.assertEqual(len(decayed_dict), 2)
        self.assertLess(decayed_dict["h1"], 2.0)
        self.assertLess(decayed_dict["h2"], 4.0)


if __name__ == "__main__":
    unittest.main()
