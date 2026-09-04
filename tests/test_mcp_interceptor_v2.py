"""
Tests for MCP Tool Registry Auth Interceptor V2.
"""

import asyncio
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.safety.mcp_interceptor_v2 import (
        MCPToolRegistryAuthInterceptorV2,
        MCPAuthStatus,
        MCPAuthInterceptResult,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "safety" / "mcp_interceptor_v2.py"
    spec = importlib.util.spec_from_file_location("mcp_interceptor_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    MCPToolRegistryAuthInterceptorV2 = module.MCPToolRegistryAuthInterceptorV2
    MCPAuthStatus = module.MCPAuthStatus
    MCPAuthInterceptResult = module.MCPAuthInterceptResult


class TestMCPInterceptorV2(unittest.IsolatedAsyncioTestCase):
    """
    Test suite verifying MCP dynamic registry runtime interception,
    server boundaries, and tool authorization policies.
    """

    def setUp(self):
        self.interceptor = MCPToolRegistryAuthInterceptorV2()

    # -------------------------------------------------------------------------
    # 1. Allowed vs Blocked Tool Execution
    # -------------------------------------------------------------------------
    async def test_allowed_tool_execution(self):
        """Allowed tools should execute normally and return success result."""
        def safe_query(param: str) -> str:
            return f"Query result: {param}"

        res = await self.interceptor.intercept_and_execute(
            tool_func=safe_query,
            tool_name="github_search_repos",
            arguments={"param": "agentic"},
        )

        self.assertTrue(res.success)
        self.assertTrue(res.allowed)
        self.assertEqual(res.result, "Query result: agentic")
        self.assertEqual(res.status, MCPAuthStatus.ALLOWED)

    async def test_blocked_tool_execution(self):
        """Blacklisted tools must be intercepted and prevented from executing."""
        mock_tool = MagicMock(return_value="Should not execute")

        res = await self.interceptor.intercept_and_execute(
            tool_func=mock_tool,
            tool_name="mcp_system_shutdown",
            arguments={},
        )

        self.assertFalse(res.allowed)
        self.assertFalse(res.success)
        self.assertEqual(res.status, MCPAuthStatus.BLOCKED)
        self.assertIn("blacklisted by runtime registry", res.message)
        mock_tool.assert_not_called()

    # -------------------------------------------------------------------------
    # 2. Server Boundary Enforcement
    # -------------------------------------------------------------------------
    async def test_blocked_server_execution(self):
        """Calls routed to blacklisted server endpoints must be blocked."""
        mock_tool = MagicMock()

        res = await self.interceptor.intercept_and_execute(
            tool_func=mock_tool,
            tool_name="untrusted_external_mesh_fetch_data",
            arguments={},
        )

        self.assertFalse(res.allowed)
        self.assertEqual(res.status, MCPAuthStatus.BLOCKED)
        self.assertIn("blacklisted from receiving MCP tool calls", res.message)
        mock_tool.assert_not_called()

    async def test_server_whitelist_enforcement(self):
        """When allowed_servers is set, servers not in whitelist must be rejected as unauthorized."""
        interceptor = MCPToolRegistryAuthInterceptorV2(
            allowed_servers={"trusted_github", "trusted_slack"}
        )

        def mock_func():
            return "OK"

        # Allowed server
        ok_res = await interceptor.intercept_and_execute(
            tool_func=mock_func,
            tool_name="trusted_github_list_prs",
        )
        self.assertTrue(ok_res.allowed)

        # Unlisted server
        bad_res = await interceptor.intercept_and_execute(
            tool_func=mock_func,
            tool_name="random_server_execute",
        )
        self.assertFalse(bad_res.allowed)
        self.assertEqual(bad_res.status, MCPAuthStatus.UNAUTHORIZED)
        self.assertIn("not in the allowed servers whitelist", bad_res.message)

    # -------------------------------------------------------------------------
    # 3. Auth Token & Custom Validators
    # -------------------------------------------------------------------------
    async def test_auth_token_requirement(self):
        """When require_auth_token is enabled, missing token must fail with UNAUTHORIZED."""
        interceptor = MCPToolRegistryAuthInterceptorV2(require_auth_token=True)

        def action():
            return "Done"

        # Without token
        res_no_tok = await interceptor.intercept_and_execute(action, "deploy_service")
        self.assertFalse(res_no_tok.allowed)
        self.assertEqual(res_no_tok.status, MCPAuthStatus.UNAUTHORIZED)

        # With token
        res_tok = await interceptor.intercept_and_execute(
            action,
            "deploy_service",
            context={"auth_token": "valid-token-xyz"},
        )
        self.assertTrue(res_tok.allowed)
        self.assertEqual(res_tok.result, "Done")

    async def test_custom_auth_validator(self):
        """Custom validator callback should dynamically permit or block."""
        def custom_val(name: str, args: dict, ctx: dict):
            if args.get("env") == "production" and not ctx.get("admin"):
                return False, "Production access requires admin context"
            return True, "Authorized"

        interceptor = MCPToolRegistryAuthInterceptorV2(auth_validator=custom_val)

        def db_mutate(env: str):
            return "Mutated"

        # Non-admin production call -> blocked
        blocked = await interceptor.intercept_and_execute(
            db_mutate, "database_update", arguments={"env": "production"}
        )
        self.assertFalse(blocked.allowed)
        self.assertIn("Production access requires admin context", blocked.message)

        # Admin production call -> allowed
        allowed = await interceptor.intercept_and_execute(
            db_mutate,
            "database_update",
            arguments={"env": "production"},
            context={"admin": True},
        )
        self.assertTrue(allowed.allowed)
        self.assertEqual(allowed.result, "Mutated")

    # -------------------------------------------------------------------------
    # 4. Decorator & Sync Wrapper
    # -------------------------------------------------------------------------
    async def test_decorator_wrapping(self):
        """wrap_tool decorator should raise PermissionError on blocked calls."""
        @self.interceptor.wrap_tool("mcp_format_disk")
        async def dangerous_func():
            return "Disk formatted"

        with self.assertRaises(PermissionError) as ctx:
            await dangerous_func()
        self.assertIn("blacklisted", str(ctx.exception))

    def test_sync_interception(self):
        """intercept_sync should execute synchronous functions properly."""
        def sync_calc(a: int, b: int) -> int:
            return a * b

        res = self.interceptor.intercept_sync(
            tool_func=sync_calc,
            tool_name="math_multiply",
            arguments={"a": 6, "b": 7},
        )
        self.assertTrue(res.success)
        self.assertEqual(res.result, 42)


if __name__ == "__main__":
    unittest.main()
