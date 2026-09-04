"""
Unit tests for MCP Action Tool Policy Interceptor V8.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.skills.mcp_policy_interceptor_v8 import (
        MCPActionPolicyInterceptOutcome,
        MCPActionToolPolicyInterceptorV8,
        MCPPolicyInterceptAction,
    )
    from magda_agent.safety.taint import mark_tainted
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "skills"
        / "mcp_policy_interceptor_v8.py"
    )
    spec = importlib.util.spec_from_file_location("mcp_policy_interceptor_v8", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    MCPActionPolicyInterceptOutcome = module.MCPActionPolicyInterceptOutcome
    MCPActionToolPolicyInterceptorV8 = module.MCPActionToolPolicyInterceptorV8
    MCPPolicyInterceptAction = module.MCPPolicyInterceptAction
    mark_tainted = getattr(module, "mark_tainted", lambda x: x)


class TestMCPPolicyInterceptorV8(unittest.TestCase):
    def setUp(self):
        self.interceptor = MCPActionToolPolicyInterceptorV8()

    def test_tainted_input_blocked(self):
        mock_handler = MagicMock(return_value="File edited")
        self.interceptor.register_tool("edit_config", mock_handler)

        tainted_path = mark_tainted("../system/config.json")

        res = self.interceptor.execute(
            tool_name="edit_config",
            arguments={"path": tainted_path, "value": "new_val"},
        )

        self.assertFalse(res.success)
        self.assertFalse(res.allowed)
        self.assertTrue(res.taint_detected)
        self.assertEqual(res.action, MCPPolicyInterceptAction.BLOCK_TAINTED)
        self.assertIn("path", res.tainted_fields)
        mock_handler.assert_not_called()

    def test_valid_input_allowed_and_executed(self):
        mock_handler = MagicMock(return_value="Saved successfully")
        self.interceptor.register_tool("save_note", mock_handler)

        res = self.interceptor.execute(
            tool_name="save_note",
            arguments={"note": "Clean note content"},
        )

        self.assertTrue(res.success)
        self.assertTrue(res.allowed)
        self.assertEqual(res.result, "Saved successfully")
        mock_handler.assert_called_once_with(note="Clean note content")

    def test_forbidden_tool_blocked(self):
        res = self.interceptor.execute(
            tool_name="format_disk",
            arguments={"drive": "C:"},
            tool_func=lambda **kw: "done",
        )

        self.assertFalse(res.success)
        self.assertFalse(res.allowed)
        self.assertEqual(res.action, MCPPolicyInterceptAction.BLOCK_FORBIDDEN)
        self.assertIn("blacklisted", res.error)

    def test_custom_tool_validator(self):
        def validate_range(args):
            val = args.get("count", 0)
            if val < 1 or val > 100:
                return False, "Count must be between 1 and 100"
            return True, "OK"

        self.interceptor.register_tool("batch_items", lambda count: f"Batch {count}")
        self.interceptor.add_custom_validator("batch_items", validate_range)

        # 1. Invalid count -> Blocked
        res_invalid = self.interceptor.execute("batch_items", {"count": 500})
        self.assertFalse(res_invalid.success)
        self.assertEqual(res_invalid.action, MCPPolicyInterceptAction.BLOCK_INVALID)
        self.assertIn("Count must be between 1 and 100", res_invalid.error)

        # 2. Valid count -> Allowed
        res_valid = self.interceptor.execute("batch_items", {"count": 25})
        self.assertTrue(res_valid.success)
        self.assertEqual(res_valid.result, "Batch 25")

    def test_async_interceptor_execution(self):
        async def run_async():
            mock_async = AsyncMock(return_value="Async success")
            self.interceptor.register_tool("async_fetch", mock_async)

            # Valid call
            res = await self.interceptor.execute_async("async_fetch", {"url": "https://api.internal/data"})
            self.assertTrue(res.success)
            self.assertEqual(res.result, "Async success")
            mock_async.assert_called_once_with(url="https://api.internal/data")

            # Tainted call
            res_tainted = await self.interceptor.execute_async("async_fetch", {"url": mark_tainted("http://evil.com")})
            self.assertFalse(res_tainted.success)
            self.assertEqual(res_tainted.action, MCPPolicyInterceptAction.BLOCK_TAINTED)

        asyncio.run(run_async())


if __name__ == "__main__":
    unittest.main()
