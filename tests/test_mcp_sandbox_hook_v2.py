"""
Tests for MCP Dynamic Verification Sandbox Hook V2.
"""

import asyncio
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.safety.mcp_sandbox_hook_v2 import (
        MCPDynamicVerificationSandboxHookV2,
        HookDecision,
        HookExecutionOutcome,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "safety" / "mcp_sandbox_hook_v2.py"
    spec = importlib.util.spec_from_file_location("mcp_sandbox_hook_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    MCPDynamicVerificationSandboxHookV2 = module.MCPDynamicVerificationSandboxHookV2
    HookDecision = module.HookDecision
    HookExecutionOutcome = module.HookExecutionOutcome


class TestMCPSandboxHookV2(unittest.IsolatedAsyncioTestCase):
    """
    Test suite verifying dynamic verification hooks, pre/post execution checks,
    and sandboxed tool execution containment.
    """

    def setUp(self):
        self.hook = MCPDynamicVerificationSandboxHookV2()

    # -------------------------------------------------------------------------
    # 1. Allowed & Blocked Tools
    # -------------------------------------------------------------------------
    async def test_allowed_call_passes_sandboxed(self):
        """Valid tool calls should execute within the sandbox and return results."""
        def safe_action(data: str) -> str:
            return f"Processed: {data}"

        outcome = await self.hook.intercept_and_sandbox(
            tool_func=safe_action,
            tool_name="mcp_process_data",
            arguments={"data": "sample_input"},
        )

        self.assertTrue(outcome.success)
        self.assertTrue(outcome.allowed)
        self.assertEqual(outcome.result, "Processed: sample_input")
        self.assertEqual(outcome.decision, HookDecision.ALLOW)
        self.assertTrue(outcome.sandboxed)

    async def test_blocked_tool_intercepted(self):
        """Blacklisted tools must be intercepted before invocation."""
        mock_func = MagicMock()

        outcome = await self.hook.intercept_and_sandbox(
            tool_func=mock_func,
            tool_name="mcp_destroy_volume",
            arguments={},
        )

        self.assertFalse(outcome.allowed)
        self.assertEqual(outcome.decision, HookDecision.BLOCK)
        self.assertIn("blacklisted", outcome.message)
        mock_func.assert_not_called()

    # -------------------------------------------------------------------------
    # 2. Pre-Execution Verification Hooks
    # -------------------------------------------------------------------------
    async def test_pre_hook_denial_blocks_execution(self):
        """Pre-execution hook should inspect arguments and block unsafe calls."""
        def param_check_hook(tool_name: str, args: dict, ctx: dict):
            if ".." in args.get("path", ""):
                return False, "Directory traversal detected in path argument"
            return True, "Parameters verified"

        self.hook.register_pre_hook(param_check_hook)
        mock_tool = MagicMock(return_value="read")

        # Unsafe call
        outcome = await self.hook.intercept_and_sandbox(
            tool_func=mock_tool,
            tool_name="file_reader",
            arguments={"path": "../../etc/passwd"},
        )

        self.assertFalse(outcome.allowed)
        self.assertIn("Directory traversal detected", outcome.message)
        mock_tool.assert_not_called()

        # Safe call
        safe_outcome = await self.hook.intercept_and_sandbox(
            tool_func=mock_tool,
            tool_name="file_reader",
            arguments={"path": "workspace/data.txt"},
        )
        self.assertTrue(safe_outcome.allowed)
        mock_tool.assert_called_once()

    # -------------------------------------------------------------------------
    # 3. Post-Execution Validation Hooks
    # -------------------------------------------------------------------------
    async def test_post_hook_output_validation_and_sanitization(self):
        """Post-execution hook should validate output and apply sanitization."""
        def output_validator_hook(tool_name: str, result: any, args: dict, ctx: dict):
            if "SECRET_KEY" in str(result):
                return False, "Sensitive key leak in output", None
            return True, "Output clean", result

        self.hook.register_post_hook(output_validator_hook)

        def leaker_tool():
            return "Dump: SECRET_KEY_12345"

        outcome = await self.hook.intercept_and_sandbox(
            tool_func=leaker_tool,
            tool_name="dump_env",
        )

        self.assertFalse(outcome.allowed)
        self.assertIn("Sensitive key leak", outcome.message)

    # -------------------------------------------------------------------------
    # 4. Timeout & Error Containment
    # -------------------------------------------------------------------------
    async def test_execution_timeout_in_sandbox(self):
        """Slow tool calls exceeding timeout must be contained cleanly."""
        async def slow_tool():
            await asyncio.sleep(0.5)
            return "Should timeout"

        outcome = await self.hook.intercept_and_sandbox(
            tool_func=slow_tool,
            tool_name="slow_task",
            timeout=0.05,
        )

        self.assertFalse(outcome.success)
        self.assertIn("timed out", outcome.message)

    async def test_runtime_error_trapped(self):
        """Exceptions in tool executions should be captured in outcome without crashing."""
        def faulty_tool():
            raise ConnectionError("External API unreachable")

        outcome = await self.hook.intercept_and_sandbox(
            tool_func=faulty_tool,
            tool_name="api_fetch",
        )

        self.assertFalse(outcome.success)
        self.assertIn("External API unreachable", outcome.message)

    # -------------------------------------------------------------------------
    # 5. Decorator & Sync Wrapper
    # -------------------------------------------------------------------------
    async def test_decorator_and_sync_execution(self):
        """hook_tool decorator and sync execution wrapper should work properly."""
        @self.hook.hook_tool("calc_square")
        def square(x: int) -> int:
            return x * x

        res = square(x=9)
        self.assertEqual(res, 81)


if __name__ == "__main__":
    unittest.main()
