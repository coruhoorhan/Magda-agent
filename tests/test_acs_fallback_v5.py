"""
Tests for ACS Control Fallback Strategy v5.
"""

import asyncio
from typing import Any
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.safety.acs_fallback_v5 import (
        ACSControlFallbackStrategyV5,
        FallbackMode,
        ACSExecutionOutcome,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "safety" / "acs_fallback_v5.py"
    spec = importlib.util.spec_from_file_location("acs_fallback_v5", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ACSControlFallbackStrategyV5 = module.ACSControlFallbackStrategyV5
    FallbackMode = module.FallbackMode
    ACSExecutionOutcome = module.ACSExecutionOutcome


class TestACSControlFallbackStrategyV5(unittest.IsolatedAsyncioTestCase):
    """
    Comprehensive test suite verifying ACS post-execution state checks,
    denial interception, and graceful fallback paths (neutral, redaction, custom handlers).
    """

    async def asyncSetUp(self):
        self.strategy = ACSControlFallbackStrategyV5()

    # -------------------------------------------------------------------------
    # 1. Valid Output & Neutral Fallback
    # -------------------------------------------------------------------------
    async def test_valid_output_passes_without_fallback(self):
        """Clean output conforming to all state rules should pass directly."""
        def safe_tool() -> str:
            return "Operation completed successfully. User ID: 4892."

        outcome = await self.strategy.execute_with_fallback(
            tool_func=safe_tool,
            tool_name="get_user_info",
        )

        self.assertTrue(outcome.success)
        self.assertFalse(outcome.fallback_taken)
        self.assertTrue(outcome.validation_passed)
        self.assertEqual(outcome.output, "Operation completed successfully. User ID: 4892.")
        self.assertIsNone(outcome.denial_reason)

    async def test_denied_output_takes_neutral_fallback(self):
        """Tool output leaking PII or secrets must trigger fallback to neutral response."""
        def leaking_tool() -> str:
            return "Customer found: Alice, SSN: 123-45-6789, Status: Active"

        outcome = await self.strategy.execute_with_fallback(
            tool_func=leaking_tool,
            tool_name="query_customer_record",
            fallback_mode=FallbackMode.NEUTRAL_MESSAGE,
        )

        self.assertTrue(outcome.success)
        self.assertTrue(outcome.fallback_taken)
        self.assertFalse(outcome.validation_passed)
        self.assertIn("Social Security Number detected", outcome.denial_reason)
        self.assertEqual(outcome.output, self.strategy.DEFAULT_NEUTRAL_RESPONSE)
        self.assertEqual(outcome.fallback_mode_applied, FallbackMode.NEUTRAL_MESSAGE.value)

    # -------------------------------------------------------------------------
    # 2. Custom Neutral Message Configuration
    # -------------------------------------------------------------------------
    async def test_custom_neutral_message_respected(self):
        """Custom configured neutral message should be returned on denial."""
        custom_msg = "Safety Check: This specific result has been withheld by policy."
        strategy = ACSControlFallbackStrategyV5(custom_neutral_message=custom_msg)

        def secret_leaker():
            return "System key: api_key=sk-1234567890abcdef1234567890"

        outcome = await strategy.execute_with_fallback(
            tool_func=secret_leaker,
            tool_name="fetch_config",
        )

        self.assertTrue(outcome.fallback_taken)
        self.assertEqual(outcome.output, custom_msg)

    # -------------------------------------------------------------------------
    # 3. Redaction Mode
    # -------------------------------------------------------------------------
    async def test_redact_sensitive_fallback_mode(self):
        """Redact mode should mask sensitive tokens and return the sanitized string."""
        def credit_card_leaker():
            return "Order placed for card 4111 2222 3333 4444."

        outcome = await self.strategy.execute_with_fallback(
            tool_func=credit_card_leaker,
            tool_name="process_payment",
            fallback_mode=FallbackMode.REDACT_SENSITIVE,
        )

        self.assertTrue(outcome.fallback_taken)
        self.assertIn("[REDACTED_BY_ACS]", outcome.output)
        self.assertNotIn("4111 2222 3333 4444", outcome.output)
        self.assertIn("Order placed for card", outcome.output)

    # -------------------------------------------------------------------------
    # 4. Custom Tool Fallback Handler
    # -------------------------------------------------------------------------
    async def test_custom_tool_fallback_handler(self):
        """Custom handler registered for a specific tool should be executed on denial."""
        def custom_handler(raw_output: Any, reason: str, args: dict) -> dict:
            return {"fallback": True, "masked_user": args.get("user_id"), "reason": reason}

        self.strategy.register_tool_fallback("user_exporter", custom_handler)

        def dangerous_exporter(user_id: str):
            return f"User {user_id} raw dump: DROP_TABLE_CONFIRMED"

        outcome = await self.strategy.execute_with_fallback(
            tool_func=dangerous_exporter,
            tool_name="user_exporter",
            args={"user_id": "usr_99"},
            fallback_mode=FallbackMode.CUSTOM_HANDLER,
        )

        self.assertTrue(outcome.fallback_taken)
        self.assertEqual(outcome.output["fallback"], True)
        self.assertEqual(outcome.output["masked_user"], "usr_99")
        self.assertIn("Forbidden state mutation indicator", outcome.output["reason"])

    # -------------------------------------------------------------------------
    # 5. Forbidden State Indicators & Custom State Rules
    # -------------------------------------------------------------------------
    async def test_forbidden_state_indicator_denial(self):
        """Outputs with forbidden state mutation tokens must be blocked."""
        def malicious_tool():
            return "Result status: MALICIOUS_PAYLOAD_EXECUTED on server"

        outcome = await self.strategy.execute_with_fallback(
            tool_func=malicious_tool,
            tool_name="remote_runner",
        )

        self.assertTrue(outcome.fallback_taken)
        self.assertIn("Forbidden state mutation indicator", outcome.denial_reason)

    async def test_custom_state_validation_rule(self):
        """User-defined state rule should be able to deny output."""
        def no_even_numbers_rule(output: Any, ctx: dict):
            if isinstance(output, int) and output % 2 == 0:
                return False, "Even numbers are disallowed by domain policy."
            return True, "Number OK"

        self.strategy.register_state_rule("NoEvenNumbersRule", no_even_numbers_rule)

        def calc_tool(val: int):
            return val

        # Valid odd number
        ok_outcome = await self.strategy.execute_with_fallback(
            tool_func=calc_tool,
            tool_name="calc",
            args={"val": 5},
        )
        self.assertFalse(ok_outcome.fallback_taken)
        self.assertEqual(ok_outcome.output, 5)

        # Denied even number
        denied_outcome = await self.strategy.execute_with_fallback(
            tool_func=calc_tool,
            tool_name="calc",
            args={"val": 8},
        )
        self.assertTrue(denied_outcome.fallback_taken)
        self.assertIn("Even numbers are disallowed", denied_outcome.denial_reason)

    # -------------------------------------------------------------------------
    # 6. Tool Execution Errors & Sync Wrapper
    # -------------------------------------------------------------------------
    async def test_tool_execution_error_graceful_fallback(self):
        """Runtime exception in tool should be caught and routed through fallback."""
        def broken_tool():
            raise ConnectionResetError("Remote server closed socket")

        outcome = await self.strategy.execute_with_fallback(
            tool_func=broken_tool,
            tool_name="faulty_service",
        )

        self.assertTrue(outcome.fallback_taken)
        self.assertIn("Remote server closed socket", outcome.denial_reason)
        self.assertEqual(outcome.output, self.strategy.DEFAULT_NEUTRAL_RESPONSE)

    async def test_sync_and_async_execution(self):
        """Sync and async methods should both work seamlessly."""
        async def async_worker(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 10

        def sync_worker(x: int) -> int:
            return x * 20

        res_async = await self.strategy.execute_with_fallback(async_worker, "async_tool", args={"x": 3})
        self.assertEqual(res_async.output, 30)

        res_sync = self.strategy.execute_with_fallback_sync(sync_worker, "sync_tool", args={"x": 3})
        self.assertEqual(res_sync.output, 60)

    # -------------------------------------------------------------------------
    # 7. Audit History
    # -------------------------------------------------------------------------
    async def test_audit_history_recording(self):
        """All executions and fallback transitions must be recorded in history."""
        def dummy_tool():
            return "Clean output"

        await self.strategy.execute_with_fallback(dummy_tool, "dummy_tool")

        history = self.strategy.get_audit_history()
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[-1]["tool_name"], "dummy_tool")
        self.assertTrue(history[-1]["validation_passed"])


if __name__ == "__main__":
    unittest.main()
