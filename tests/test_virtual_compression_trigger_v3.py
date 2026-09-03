"""
Tests for MemGPT Virtual Context Compression Trigger V3.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.memory.virtual_compression_trigger_v3 import (
        VirtualContextCompressionTriggerV3,
        CompressionTriggerResult,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "memory" / "virtual_compression_trigger_v3.py"
    spec = importlib.util.spec_from_file_location("virtual_compression_trigger_v3", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VirtualContextCompressionTriggerV3 = module.VirtualContextCompressionTriggerV3
    CompressionTriggerResult = module.CompressionTriggerResult


class TestVirtualCompressionTriggerV3(unittest.IsolatedAsyncioTestCase):
    """
    Test suite verifying token threshold monitoring, proactive trigger initiation,
    and delegating context compression to engines and subagents.
    """

    # -------------------------------------------------------------------------
    # 1. Threshold Detection
    # -------------------------------------------------------------------------
    def test_should_compress_threshold_detection(self):
        """Trigger should accurately calculate threshold (e.g. 80% of 100 = 80 tokens)."""
        trigger = VirtualContextCompressionTriggerV3(
            max_context_tokens=100,
            trigger_threshold_ratio=0.80,
        )

        self.assertEqual(trigger.threshold_tokens, 80)

        # Under threshold
        small_dialogue = [{"content": "Hello world"}]
        should_c, toks, thresh = trigger.should_compress(small_dialogue)
        self.assertFalse(should_c)
        self.assertLess(toks, 80)

        # Over threshold
        large_content = " ".join(["detailed long message phrase"] * 25)
        large_dialogue = [{"content": large_content}]
        should_c, toks, thresh = trigger.should_compress(large_dialogue)
        self.assertTrue(should_c)
        self.assertGreaterEqual(toks, 80)

    # -------------------------------------------------------------------------
    # 2. Trigger Evaluation & Execution
    # -------------------------------------------------------------------------
    async def test_evaluate_and_trigger_initiates_compression(self):
        """When dialogue crosses threshold, compression must be triggered and tokens saved."""
        trigger = VirtualContextCompressionTriggerV3(
            max_context_tokens=50,
            trigger_threshold_ratio=0.70,  # 35 tokens threshold
        )

        dialogue = [
            {"role": "user", "content": "Let us discuss comprehensive database schema migration steps."},
            {"role": "assistant", "content": "Sure, we should first back up tables, apply alter table queries."},
            {"role": "user", "content": "What about indexes and foreign key constraints on the user entity?"},
            {"role": "assistant", "content": "We can build indexes concurrently to avoid blocking active queries."},
        ]

        result = await trigger.evaluate_and_trigger(dialogue)

        self.assertTrue(result.triggered)
        self.assertGreater(result.initial_tokens, 35)
        self.assertLess(result.final_tokens, result.initial_tokens)
        self.assertGreater(result.tokens_saved, 0)
        self.assertEqual(trigger.metrics["triggers_count"], 1)

    async def test_evaluate_and_trigger_under_threshold_noop(self):
        """When dialogue is below threshold, compression should not trigger."""
        trigger = VirtualContextCompressionTriggerV3(
            max_context_tokens=2000,
            trigger_threshold_ratio=0.80,
        )

        dialogue = [{"role": "user", "content": "Brief message."}]
        result = await trigger.evaluate_and_trigger(dialogue)

        self.assertFalse(result.triggered)
        self.assertEqual(result.tokens_saved, 0)
        self.assertEqual(len(result.compressed_history), 1)

    # -------------------------------------------------------------------------
    # 3. Custom Compressor Callback Integration
    # -------------------------------------------------------------------------
    async def test_custom_compress_fn_invocation(self):
        """Trigger should delegate to custom compress_fn when supplied."""
        mock_compress_fn = AsyncMock()
        mock_compress_fn.return_value = [{"content": "Custom summarized dialogue"}]

        trigger = VirtualContextCompressionTriggerV3(
            max_context_tokens=30,
            trigger_threshold_ratio=0.50,
            compress_fn=mock_compress_fn,
        )

        dialogue = [
            {"content": "Turn 1 with lots of words to exceed the low limit."},
            {"content": "Turn 2 with even more words to ensure it triggers."},
        ]

        result = await trigger.evaluate_and_trigger(dialogue, context_metadata={"user_id": 10})

        self.assertTrue(result.triggered)
        mock_compress_fn.assert_called_once()
        self.assertEqual(result.compressed_history[0]["content"], "Custom summarized dialogue")

    # -------------------------------------------------------------------------
    # 4. Compressor Object Integration & Sync Wrapper
    # -------------------------------------------------------------------------
    async def test_compressor_object_integration(self):
        """Trigger should delegate to compressor object implementing compact_context."""
        mock_compressor = AsyncMock()
        mock_compressor.compact_context.return_value = [{"content": "Compacted via plugin"}]

        trigger = VirtualContextCompressionTriggerV3(
            max_context_tokens=20,
            trigger_threshold_ratio=0.50,
            compressor=mock_compressor,
        )

        dialogue = [
            {"content": "Long detailed message A."},
            {"content": "Long detailed message B."},
        ]

        result = await trigger.evaluate_and_trigger(dialogue)

        self.assertTrue(result.triggered)
        mock_compressor.compact_context.assert_called_once()
        self.assertEqual(result.compressed_history[0]["content"], "Compacted via plugin")

    def test_sync_evaluation_wrapper(self):
        """evaluate_and_trigger_sync should run synchronously."""
        trigger = VirtualContextCompressionTriggerV3(max_context_tokens=1000)
        result = trigger.evaluate_and_trigger_sync([{"content": "Hi"}])
        self.assertFalse(result.triggered)


if __name__ == "__main__":
    unittest.main()
