"""
Unit tests for MCP Action Tool Rate Limiting V1.
"""

import asyncio
import time
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.safety.mcp_rate_limiting_v1 import (
        MCPActionToolRateLimiterV1,
        MCPRateLimitDecision,
        MCPRateLimitExceededError,
        RateLimitedExecutionResult,
        TokenBucket,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "safety"
        / "mcp_rate_limiting_v1.py"
    )
    spec = importlib.util.spec_from_file_location("mcp_rate_limiting_v1", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    MCPActionToolRateLimiterV1 = module.MCPActionToolRateLimiterV1
    MCPRateLimitDecision = module.MCPRateLimitDecision
    MCPRateLimitExceededError = module.MCPRateLimitExceededError
    RateLimitedExecutionResult = module.RateLimitedExecutionResult
    TokenBucket = module.TokenBucket


class TestMCPRateLimitingV1(unittest.TestCase):
    def setUp(self):
        self.limiter = MCPActionToolRateLimiterV1(
            default_rate_per_second=5.0,
            default_burst_capacity=5.0,
        )

    def test_token_bucket_consume_and_refill(self):
        bucket = TokenBucket(capacity=3.0, refill_rate=1.0)
        now = time.time()

        # Consume 3 tokens
        self.assertTrue(bucket.consume(1.0, current_time=now))
        self.assertTrue(bucket.consume(2.0, current_time=now))

        # Bucket empty now
        self.assertFalse(bucket.consume(1.0, current_time=now))
        self.assertEqual(bucket.current_tokens, 0.0)

        # Fast forward time by 2 seconds -> should have 2 tokens
        self.assertTrue(bucket.can_consume(2.0, current_time=now + 2.0))
        self.assertTrue(bucket.consume(2.0, current_time=now + 2.0))
        self.assertEqual(bucket.current_tokens, 0.0)

    def test_block_excessive_requests_global(self):
        mock_tool = MagicMock(return_value="executed")

        # Global burst is 5. Execute 5 times successfully
        for _ in range(5):
            res = self.limiter.execute_with_rate_limit("generic_tool", mock_tool, {"param": "x"})
            self.assertTrue(res.success)
            self.assertFalse(res.blocked_by_rate_limit)

        # 6th request should exceed rate limit and be blocked
        res_blocked = self.limiter.execute_with_rate_limit("generic_tool", mock_tool, {"param": "x"})
        self.assertFalse(res_blocked.success)
        self.assertTrue(res_blocked.blocked_by_rate_limit)
        self.assertIn("MCP Rate Limit Exceeded", res_blocked.error)
        self.assertEqual(mock_tool.call_count, 5)

    def test_tool_specific_rate_limit(self):
        # Configure strict limit for expensive tool: burst 2, 1/sec, cost 1.0
        self.limiter.configure_tool_limit(
            tool_name="heavy_query",
            rate_per_second=1.0,
            burst_capacity=2.0,
            cost_per_call=1.0,
        )

        mock_heavy = MagicMock(return_value="Heavy result")

        # 1st and 2nd call pass
        r1 = self.limiter.execute_with_rate_limit("heavy_query", mock_heavy, {})
        r2 = self.limiter.execute_with_rate_limit("heavy_query", mock_heavy, {})
        self.assertTrue(r1.success)
        self.assertTrue(r2.success)

        # 3rd call immediately blocked
        r3 = self.limiter.execute_with_rate_limit("heavy_query", mock_heavy, {})
        self.assertFalse(r3.success)
        self.assertTrue(r3.blocked_by_rate_limit)
        self.assertIn("Tool-level rate limit exceeded", r3.error)
        self.assertEqual(mock_heavy.call_count, 2)

    def test_server_prefix_rate_limit(self):
        self.limiter.configure_server_limit(
            server_prefix="github",
            rate_per_second=2.0,
            burst_capacity=2.0,
        )

        mock_fn = MagicMock(return_value="gh")

        # github__create_pr and github__merge_pr share the 'github' bucket
        r1 = self.limiter.execute_with_rate_limit("github__create_pr", mock_fn, {})
        r2 = self.limiter.execute_with_rate_limit("github__merge_pr", mock_fn, {})
        self.assertTrue(r1.success)
        self.assertTrue(r2.success)

        # 3rd call to github__close_issue blocked
        r3 = self.limiter.execute_with_rate_limit("github__close_issue", mock_fn, {})
        self.assertFalse(r3.success)
        self.assertTrue(r3.blocked_by_rate_limit)
        self.assertIn("Server-level rate limit exceeded", r3.error)

    def test_async_rate_limited_execution(self):
        async def run_async():
            async def mock_async_tool(val: int) -> int:
                await asyncio.sleep(0.01)
                return val * 2

            self.limiter.configure_tool_limit("async_calc", rate_per_second=1.0, burst_capacity=1.0)

            # 1st call ok
            res1 = await self.limiter.execute_with_rate_limit_async("async_calc", mock_async_tool, {"val": 10})
            self.assertTrue(res1.success)
            self.assertEqual(res1.result, 20)

            # 2nd call blocked
            res2 = await self.limiter.execute_with_rate_limit_async("async_calc", mock_async_tool, {"val": 20})
            self.assertFalse(res2.success)
            self.assertTrue(res2.blocked_by_rate_limit)

        asyncio.run(run_async())

    def test_audit_trail_and_reset(self):
        self.limiter.clear_audit_trail()
        self.assertEqual(len(self.limiter.get_audit_trail()), 0)

        self.limiter.execute_with_rate_limit("tool_1", lambda: "ok", {})
        self.assertEqual(len(self.limiter.get_audit_trail()), 1)

        self.limiter.reset_limits()
        self.assertEqual(self.limiter.global_bucket.current_tokens, self.limiter.global_bucket.capacity)


if __name__ == "__main__":
    unittest.main()
