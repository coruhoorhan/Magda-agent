"""
Unit tests for MCP Server Taint Context Isolation V1.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.integration.mcp_taint_context_v1 import (
        MCPTaintContextIsolationWrapperV1,
        MCPTaintContextResponse,
        TaintIsolationMode,
    )
    from magda_agent.safety.taint import mark_tainted
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "integration"
        / "mcp_taint_context_v1.py"
    )
    spec = importlib.util.spec_from_file_location("mcp_taint_context_v1", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    MCPTaintContextIsolationWrapperV1 = module.MCPTaintContextIsolationWrapperV1
    MCPTaintContextResponse = module.MCPTaintContextResponse
    TaintIsolationMode = module.TaintIsolationMode
    mark_tainted = getattr(module, "mark_tainted", lambda x: x)


class TestMCPTaintContextIsolationV1(unittest.TestCase):
    def setUp(self):
        self.wrapper = MCPTaintContextIsolationWrapperV1()

    def test_clean_tool_execution_untainted(self):
        def add_numbers(a: int, b: int) -> int:
            return a + b

        resp = self.wrapper.execute_and_isolate("add_numbers", add_numbers, {"a": 2, "b": 3})

        self.assertFalse(resp.is_tainted)
        self.assertEqual(resp.raw_output, 5)
        self.assertEqual(resp.taint_level, "low")

        mcp_payload = resp.to_mcp_response()
        self.assertFalse(mcp_payload["_meta"]["tainted"])

    def test_untrusted_tool_automatically_tagged_tainted(self):
        def fetch_url(url: str) -> str:
            return "<html>Untrusted Web Content</html>"

        resp = self.wrapper.execute_and_isolate("fetch_url", fetch_url, {"url": "https://external.site"})

        self.assertTrue(resp.is_tainted)
        self.assertIn("untrusted", resp.taint_origin)
        self.assertEqual(resp.sanitized_output, "<html>Untrusted Web Content</html>")

        mcp_payload = resp.to_mcp_response()
        self.assertTrue(mcp_payload["_meta"]["tainted"])
        self.assertIn("untrusted", mcp_payload["_meta"]["taint_origin"])

    def test_tainted_input_propagates_to_output(self):
        def format_text(text: str) -> str:
            return f"Formatted: {text}"

        tainted_arg = mark_tainted("malicious user input")

        resp = self.wrapper.execute_and_isolate("format_text", format_text, {"text": tainted_arg})

        self.assertTrue(resp.is_tainted)
        self.assertIn("propagated", resp.taint_origin)
        self.assertEqual(resp.taint_level, "high")

    def test_filter_for_memory_storage_sanitization(self):
        tainted_resp = MCPTaintContextResponse(
            tool_name="web_search",
            raw_output=mark_tainted("Raw search result"),
            is_tainted=True,
            sanitized_output="Clean search result",
        )

        safe_content, was_tainted = self.wrapper.filter_for_memory_storage(tainted_resp)
        self.assertTrue(was_tainted)
        self.assertEqual(safe_content, "Clean search result")

    def test_async_execution_and_isolation(self):
        async def run_async():
            async def mock_async_search(query: str) -> str:
                await asyncio.sleep(0.01)
                return f"Results for {query}"

            resp = await self.wrapper.execute_and_isolate_async(
                "web_search",
                mock_async_search,
                {"query": "magda architecture"},
            )

            self.assertTrue(resp.is_tainted)
            self.assertEqual(resp.sanitized_output, "Results for magda architecture")
            self.assertEqual(len(self.wrapper.get_audit_trail()), 1)

        asyncio.run(run_async())


if __name__ == "__main__":
    unittest.main()
