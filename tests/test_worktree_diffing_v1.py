"""
Unit tests for Agent Teams Worktree State Diffing V1.
"""

import asyncio
import os
import shutil
import tempfile
import unittest

try:
    from magda_agent.architecture.worktree_diffing_v1 import (
        IsolationLeak,
        IsolationLeakType,
        WorktreeDiffReport,
        WorktreeStateDifferV1,
        WorktreeStateSnapshot,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "architecture"
        / "worktree_diffing_v1.py"
    )
    spec = importlib.util.spec_from_file_location("worktree_diffing_v1", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    IsolationLeak = module.IsolationLeak
    IsolationLeakType = module.IsolationLeakType
    WorktreeDiffReport = module.WorktreeDiffReport
    WorktreeStateDifferV1 = module.WorktreeStateDifferV1
    WorktreeStateSnapshot = module.WorktreeStateSnapshot


class TestWorktreeDiffingV1(unittest.TestCase):
    def setUp(self):
        self.differ = WorktreeStateDifferV1(allowed_shared_files={".gitignore", "README.md"})

    def test_clean_isolation_evaluation(self):
        # Two agents modifying completely separate files
        snap_a = self.differ.capture_snapshot_from_data(
            agent_id="agent_auth",
            worktree_path="/tmp/wt_auth",
            modified_files={"src/auth.py", "tests/test_auth.py"},
        )

        snap_b = self.differ.capture_snapshot_from_data(
            agent_id="agent_db",
            worktree_path="/tmp/wt_db",
            modified_files={"src/db.py", "tests/test_db.py"},
        )

        report = self.differ.evaluate_isolation_leaks({
            "agent_auth": snap_a,
            "agent_db": snap_b,
        })

        self.assertTrue(report.is_clean)
        self.assertEqual(len(report.leaks_detected), 0)
        self.assertEqual(len(report.conflicting_files), 0)

    def test_overlapping_modification_leak(self):
        # Both agents modify the same unshared file: src/common.py
        snap_a = self.differ.capture_snapshot_from_data(
            agent_id="agent_1",
            worktree_path="/tmp/wt_1",
            modified_files={"src/common.py", "src/mod1.py"},
        )

        snap_b = self.differ.capture_snapshot_from_data(
            agent_id="agent_2",
            worktree_path="/tmp/wt_2",
            modified_files={"src/common.py", "src/mod2.py"},
        )

        report = self.differ.evaluate_isolation_leaks({
            "agent_1": snap_a,
            "agent_2": snap_b,
        })

        self.assertFalse(report.is_clean)
        self.assertEqual(len(report.leaks_detected), 1)
        leak = report.leaks_detected[0]
        self.assertEqual(leak.leak_type, IsolationLeakType.OVERLAPPING_MODIFICATION)
        self.assertEqual(leak.file_path, "src/common.py")
        self.assertIn("agent_1", leak.involved_agents)
        self.assertIn("agent_2", leak.involved_agents)

    def test_allowed_shared_files_bypass_overlap(self):
        # Both agents modify .gitignore and README.md (which are allowed shared)
        snap_a = self.differ.capture_snapshot_from_data(
            agent_id="agent_1",
            worktree_path="/tmp/wt_1",
            modified_files={".gitignore", "README.md", "file_a.py"},
        )

        snap_b = self.differ.capture_snapshot_from_data(
            agent_id="agent_2",
            worktree_path="/tmp/wt_2",
            modified_files={".gitignore", "README.md", "file_b.py"},
        )

        report = self.differ.evaluate_isolation_leaks({
            "agent_1": snap_a,
            "agent_2": snap_b,
        })

        self.assertTrue(report.is_clean)
        self.assertEqual(len(report.leaks_detected), 0)

    def test_out_of_bounds_write_leak(self):
        snap = self.differ.capture_snapshot_from_data(
            agent_id="rogue_agent",
            worktree_path="/tmp/wt_rogue",
            modified_files={"../../etc/passwd", "normal.py"},
        )

        report = self.differ.evaluate_isolation_leaks({"rogue_agent": snap})

        self.assertFalse(report.is_clean)
        self.assertTrue(any(l.leak_type == IsolationLeakType.OUT_OF_BOUNDS_WRITE for l in report.leaks_detected))

    def test_cross_contamination_namespace_leak(self):
        snap_a = self.differ.capture_snapshot_from_data(
            agent_id="agent_a",
            worktree_path="/tmp/wt_agent_a",
            modified_files={"wt_agent_b/secret.txt"},
        )
        snap_b = self.differ.capture_snapshot_from_data(
            agent_id="agent_b",
            worktree_path="/tmp/wt_agent_b",
            modified_files={"main.py"},
        )

        report = self.differ.evaluate_isolation_leaks({
            "agent_a": snap_a,
            "agent_b": snap_b,
        })

        self.assertFalse(report.is_clean)
        self.assertTrue(any(l.leak_type == IsolationLeakType.CROSS_CONTAMINATION for l in report.leaks_detected))

    def test_filesystem_snapshot_scanning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wt1 = os.path.join(temp_dir, "wt1")
            wt2 = os.path.join(temp_dir, "wt2")
            os.makedirs(wt1)
            os.makedirs(wt2)

            with open(os.path.join(wt1, "mod1.py"), "w") as f:
                f.write("print(1)")
            with open(os.path.join(wt2, "mod2.py"), "w") as f:
                f.write("print(2)")

            report = self.differ.evaluate_worktree_paths({
                "agent1": wt1,
                "agent2": wt2,
            })

            self.assertTrue(report.is_clean)
            self.assertEqual(len(report.agent_file_ownership["agent1"]), 1)
            self.assertEqual(len(report.agent_file_ownership["agent2"]), 1)


if __name__ == "__main__":
    unittest.main()
