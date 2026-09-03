"""
Unit tests for Aider Style Post-Merge Smoke Tests V1.
"""

import os
import shutil
import tempfile
import unittest

try:
    from magda_agent.evaluation.smoke_tester_v1 import (
        AiderPostMergeSmokeTesterV1,
        FileSmokeResult,
        SmokeTestReport,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "evaluation"
        / "smoke_tester_v1.py"
    )
    spec = importlib.util.spec_from_file_location("smoke_tester_v1", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    AiderPostMergeSmokeTesterV1 = module.AiderPostMergeSmokeTesterV1
    FileSmokeResult = module.FileSmokeResult
    SmokeTestReport = module.SmokeTestReport


class TestSmokeTesterV1(unittest.TestCase):
    def setUp(self):
        self.tester = AiderPostMergeSmokeTesterV1()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_check_valid_code_syntax(self):
        valid_code = """
def calculate_sum(a: int, b: int) -> int:
    return a + b

class MyService:
    def execute(self):
        return True
"""
        is_ok, err_type, err_msg, lineno, offset, count = self.tester.check_code_syntax(valid_code)
        self.assertTrue(is_ok)
        self.assertIsNone(err_type)
        self.assertGreater(count, 0)

    def test_check_syntax_error(self):
        invalid_code = """
def broken_func(a, b)
    return a + b
"""
        is_ok, err_type, err_msg, lineno, offset, count = self.tester.check_code_syntax(invalid_code)
        self.assertFalse(is_ok)
        self.assertEqual(err_type, "SyntaxError")
        self.assertEqual(lineno, 2)
        self.assertIsNotNone(err_msg)

    def test_check_indentation_error(self):
        indent_error_code = """
def test_func():
print("bad indentation")
"""
        is_ok, err_type, err_msg, lineno, offset, count = self.tester.check_code_syntax(indent_error_code)
        self.assertFalse(is_ok)
        self.assertIn("Indentation", err_type)

    def test_test_mock_files_in_directory(self):
        # 1. Create valid file
        valid_path = os.path.join(self.temp_dir, "good_module.py")
        with open(valid_path, "w") as f:
            f.write("import os\nprint('hello')\n")

        # 2. Create invalid file
        invalid_path = os.path.join(self.temp_dir, "bad_syntax.py")
        with open(invalid_path, "w") as f:
            f.write("def foo() pass\n")

        report = self.tester.test_files([valid_path, invalid_path])

        self.assertFalse(report.all_passed)
        self.assertEqual(report.total_files, 2)
        self.assertEqual(report.passed_files, 1)
        self.assertEqual(report.failed_files, 1)

        # Check details
        res_bad = [r for r in report.results if r.file_path == invalid_path][0]
        self.assertFalse(res_bad.passed)
        self.assertEqual(res_bad.error_type, "SyntaxError")

        res_good = [r for r in report.results if r.file_path == valid_path][0]
        self.assertTrue(res_good.passed)

    def test_directory_scan(self):
        f1 = os.path.join(self.temp_dir, "f1.py")
        with open(f1, "w") as f:
            f.write("x = 10\n")

        report = self.tester.test_directory(self.temp_dir)
        self.assertTrue(report.all_passed)
        self.assertEqual(report.total_files, 1)


if __name__ == "__main__":
    unittest.main()
