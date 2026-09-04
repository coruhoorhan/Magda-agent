"""
Unit tests for ACS Agent Control Specification Runtime Guard V7.
"""

import asyncio
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.safety.acs_guard_runtime_v7 import (
        ACSCheckpoint,
        ACSGuardResult,
        ACSGuardRuntimeV7,
        ACSRuntimePolicyViolationError,
        ACSValidationOutcome,
        CheckpointEvaluation,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "safety"
        / "acs_guard_runtime_v7.py"
    )
    spec = importlib.util.spec_from_file_location("acs_guard_runtime_v7", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ACSCheckpoint = module.ACSCheckpoint
    ACSGuardResult = module.ACSGuardResult
    ACSGuardRuntimeV7 = module.ACSGuardRuntimeV7
    ACSRuntimePolicyViolationError = module.ACSRuntimePolicyViolationError
    ACSValidationOutcome = module.ACSValidationOutcome
    CheckpointEvaluation = module.CheckpointEvaluation


class TestACSGuardRuntimeV7(unittest.TestCase):
    def setUp(self):
        self.guard = ACSGuardRuntimeV7()

    def test_checkpoint_1_blocks_dangerous_payload(self):
        mock_bash = MagicMock(return_value="executed")

        # Dangerous rm -rf / payload
        res = self.guard.intercept_and_execute(
            tool_name="bash",
            tool_func=mock_bash,
            arguments={"command": "rm -rf /"},
            context={"role": "admin"},
        )

        self.assertFalse(res.success)
        self.assertTrue(res.blocked_by_guard)
        self.assertEqual(res.failed_checkpoint, ACSCheckpoint.CHECKPOINT_1_INPUT.value)
        mock_bash.assert_not_called()

    def test_checkpoint_1_blocks_null_byte(self):
        mock_func = MagicMock(return_value="executed")

        res = self.guard.intercept_and_execute(
            tool_name="read_file",
            tool_func=mock_func,
            arguments={"path": "safe.txt\x00malicious.py"},
        )

        self.assertFalse(res.success)
        self.assertTrue(res.blocked_by_guard)
        self.assertEqual(res.failed_checkpoint, ACSCheckpoint.CHECKPOINT_1_INPUT.value)
        mock_func.assert_not_called()

    def test_checkpoint_2_intent_authorization_block(self):
        mock_code = MagicMock(return_value="output")

        # Regular 'user' role attempting privileged 'system_execute_code'
        res = self.guard.intercept_and_execute(
            tool_name="system_execute_code",
            tool_func=mock_code,
            arguments={"code": "print('hello')"},
            context={"role": "user"},
        )

        self.assertFalse(res.success)
        self.assertTrue(res.blocked_by_guard)
        self.assertEqual(res.failed_checkpoint, ACSCheckpoint.CHECKPOINT_2_INTENT.value)
        mock_code.assert_not_called()

    def test_checkpoint_3_tool_policy_sandbox_block(self):
        mock_fs = MagicMock(return_value="data")

        # Path escapes sandbox_root
        res = self.guard.intercept_and_execute(
            tool_name="read_file",
            tool_func=mock_fs,
            arguments={"path": "../../../etc/passwd"},
            context={"sandbox_root": "/app/data"},
        )

        self.assertFalse(res.success)
        self.assertTrue(res.blocked_by_guard)
        self.assertEqual(res.failed_checkpoint, ACSCheckpoint.CHECKPOINT_3_POLICY.value)
        mock_fs.assert_not_called()

    def test_checkpoint_4_state_transition_block(self):
        mock_tool = MagicMock(return_value="data")

        # Agent in TERMINATED state
        res = self.guard.intercept_and_execute(
            tool_name="search",
            tool_func=mock_tool,
            arguments={"query": "test"},
            context={"agent_state": "TERMINATED"},
        )

        self.assertFalse(res.success)
        self.assertTrue(res.blocked_by_guard)
        self.assertEqual(res.failed_checkpoint, ACSCheckpoint.CHECKPOINT_4_STATE.value)
        mock_tool.assert_not_called()

    def test_checkpoint_5_output_sanitization_block(self):
        # Tool generates output containing leaked API key
        mock_leak = MagicMock(return_value="Config loaded: api_key='sk-proj-1234567890abcdef12345678'")

        res = self.guard.intercept_and_execute(
            tool_name="load_config",
            tool_func=mock_leak,
            arguments={"env": "prod"},
            context={"role": "developer"},
        )

        self.assertFalse(res.success)
        self.assertTrue(res.blocked_by_guard)
        self.assertEqual(res.failed_checkpoint, ACSCheckpoint.CHECKPOINT_5_OUTPUT.value)
        self.assertIsNone(res.output)  # Output was redacted
        mock_leak.assert_called_once()

    def test_all_checkpoints_pass_successfully(self):
        mock_clean = MagicMock(return_value="Clean execution result")

        res = self.guard.intercept_and_execute(
            tool_name="fetch_status",
            tool_func=mock_clean,
            arguments={"service": "auth"},
            context={"role": "user", "agent_state": "ACTIVE"},
        )

        self.assertTrue(res.success)
        self.assertFalse(res.blocked_by_guard)
        self.assertEqual(res.output, "Clean execution result")
        self.assertIsNone(res.error)
        mock_clean.assert_called_once()

    def test_async_interception(self):
        async def run_async():
            async def mock_async_tool(x: int) -> int:
                await asyncio.sleep(0.01)
                return x * 10

            res = await self.guard.intercept_and_execute_async(
                tool_name="compute_metrics",
                tool_func=mock_async_tool,
                arguments={"x": 5},
                context={"role": "developer"},
            )

            self.assertTrue(res.success)
            self.assertFalse(res.blocked_by_guard)
            self.assertEqual(res.output, 50)

        asyncio.run(run_async())

    def test_audit_trail(self):
        self.guard.clear_audit_trail()
        self.assertEqual(len(self.guard.get_audit_trail()), 0)

        self.guard.intercept_and_execute(
            tool_name="safe_tool",
            tool_func=lambda: "ok",
            arguments={},
        )
        self.guard.intercept_and_execute(
            tool_name="system_execute_code",
            tool_func=lambda: "bad",
            arguments={},
            context={"role": "user"},
        )

        trail = self.guard.get_audit_trail()
        self.assertEqual(len(trail), 2)
        self.assertFalse(trail[0].blocked_by_guard)
        self.assertTrue(trail[1].blocked_by_guard)


if __name__ == "__main__":
    unittest.main()
