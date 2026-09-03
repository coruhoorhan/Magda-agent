"""
Unit tests for MCPKernel A2A Guardrail Runtime Checkpoint V4.
"""

import asyncio
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.safety.a2a_mcpkernel_guard_v4 import (
        A2AExecutionResponse,
        A2AGuardCheckpoint,
        A2AGuardCheckpointResult,
        A2AMCPKernelExecutionHookV4,
        A2AToolCallPayload,
    )
    from magda_agent.safety.taint import mark_tainted
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "safety"
        / "a2a_mcpkernel_guard_v4.py"
    )
    spec = importlib.util.spec_from_file_location("a2a_mcpkernel_guard_v4", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    A2AExecutionResponse = module.A2AExecutionResponse
    A2AGuardCheckpoint = module.A2AGuardCheckpoint
    A2AGuardCheckpointResult = module.A2AGuardCheckpointResult
    A2AMCPKernelExecutionHookV4 = module.A2AMCPKernelExecutionHookV4
    A2AToolCallPayload = module.A2AToolCallPayload
    mark_tainted = getattr(module, "mark_tainted", lambda x: x)


class TestA2AMCPKernelGuardV4(unittest.TestCase):
    def setUp(self):
        self.guard = A2AMCPKernelExecutionHookV4(
            trusted_peer_ids={"trusted_peer_01", "central_master"}
        )

    def test_tainted_peer_parameter_blocks_execution(self):
        mock_read = MagicMock(return_value="File content")

        # Untrusted peer calls read_file with sensitive path argument
        payload = A2AToolCallPayload(
            sender_agent_id="untrusted_peer_99",
            target_tool_name="read_file",
            arguments={"path": mark_tainted("/etc/shadow")},
            is_peer_trusted=False,
            security_tier="untrusted",
        )

        res = self.guard.intercept_and_execute(payload, mock_read)

        self.assertFalse(res.success)
        self.assertTrue(res.blocked_by_guard)
        self.assertEqual(res.checkpoint_result.failed_checkpoint, A2AGuardCheckpoint.SENSITIVE_ARG_POLICY)
        self.assertIn("Tainted parameter 'path'", res.error)
        mock_read.assert_not_called()

    def test_trusted_peer_allowed_execution(self):
        mock_read = MagicMock(return_value="Protected content")

        payload = A2AToolCallPayload(
            sender_agent_id="trusted_peer_01",
            target_tool_name="read_file",
            arguments={"path": "/var/log/audit.log"},
            is_peer_trusted=True,
            security_tier="enterprise",
        )

        res = self.guard.intercept_and_execute(payload, mock_read)

        self.assertTrue(res.success)
        self.assertFalse(res.blocked_by_guard)
        self.assertEqual(res.result, "Protected content")
        mock_read.assert_called_once_with(path="/var/log/audit.log")

    def test_untrusted_peer_capability_boundary_block(self):
        mock_exec = MagicMock(return_value="Ran")

        # Untrusted peer trying to execute code directly
        payload = A2AToolCallPayload(
            sender_agent_id="anonymous_mesh_node",
            target_tool_name="system_execute_code",
            arguments={"code": "import os; os.system('echo hi')"},
            is_peer_trusted=False,
            security_tier="untrusted",
        )

        res = self.guard.intercept_and_execute(payload, mock_exec)

        self.assertFalse(res.success)
        self.assertTrue(res.blocked_by_guard)
        mock_exec.assert_not_called()

    def test_untrusted_peer_safe_argument_allowed(self):
        mock_echo = MagicMock(return_value="Echo: Hello")

        # 'message' is a non-sensitive argument
        payload = A2AToolCallPayload(
            sender_agent_id="untrusted_peer_42",
            target_tool_name="chat_echo",
            arguments={"message": "Hello mesh!"},
            is_peer_trusted=False,
            security_tier="standard",
        )

        res = self.guard.intercept_and_execute(payload, mock_echo)

        self.assertTrue(res.success)
        self.assertFalse(res.blocked_by_guard)
        self.assertEqual(res.result, "Echo: Hello")
        mock_echo.assert_called_once_with(message="Hello mesh!")

    def test_async_interception_and_execution(self):
        async def run_async_test():
            async def mock_async_tool(text: str) -> str:
                await asyncio.sleep(0.01)
                return f"Processed: {text}"

            # 1. Blocked call (tainted path)
            payload_blocked = A2AToolCallPayload(
                sender_agent_id="peer_sub_02",
                target_tool_name="write_file",
                arguments={"path": mark_tainted("/tmp/out.txt"), "text": "hello"},
                is_peer_trusted=False,
            )
            res_blocked = await self.guard.intercept_and_execute_async(payload_blocked, mock_async_tool)
            self.assertFalse(res_blocked.success)
            self.assertTrue(res_blocked.blocked_by_guard)

            # 2. Allowed call (trusted peer)
            payload_allowed = A2AToolCallPayload(
                sender_agent_id="central_master",
                target_tool_name="format_text",
                arguments={"text": "hello world"},
                is_peer_trusted=True,
            )
            res_allowed = await self.guard.intercept_and_execute_async(payload_allowed, mock_async_tool)
            self.assertTrue(res_allowed.success)
            self.assertEqual(res_allowed.result, "Processed: hello world")

        asyncio.run(run_async_test())

    def test_audit_trail_logging(self):
        self.guard.clear_audit_trail()
        self.assertEqual(len(self.guard.get_audit_trail()), 0)

        p1 = A2AToolCallPayload(sender_agent_id="peer1", target_tool_name="read_file", arguments={"path": "bad"})
        p2 = A2AToolCallPayload(sender_agent_id="trusted_peer_01", target_tool_name="echo", arguments={"msg": "hi"}, is_peer_trusted=True)

        self.guard.intercept_and_execute(p1, lambda **kw: "ok")
        self.guard.intercept_and_execute(p2, lambda **kw: "ok")

        trail = self.guard.get_audit_trail()
        self.assertEqual(len(trail), 2)
        self.assertTrue(trail[0].blocked_by_guard)
        self.assertFalse(trail[1].blocked_by_guard)


if __name__ == "__main__":
    unittest.main()
