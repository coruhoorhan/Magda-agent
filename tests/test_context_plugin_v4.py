"""
Tests for Virtual Context Compression V4 Plugin.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.memory.context_plugin_v4 import VirtualContextCompressionPluginV4
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "memory" / "context_plugin_v4.py"
    spec = importlib.util.spec_from_file_location("context_plugin_v4", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VirtualContextCompressionPluginV4 = module.VirtualContextCompressionPluginV4


class TestContextPluginV4(unittest.IsolatedAsyncioTestCase):
    """
    Test suite verifying selective context compression, token limit enforcement,
    and episodic/working memory store interaction.
    """

    # -------------------------------------------------------------------------
    # 1. Compaction Triggers & Importance Partitioning
    # -------------------------------------------------------------------------
    async def test_context_compression_triggers_on_exceeding_tokens(self):
        """When token count exceeds max_tokens, low-importance items must be compressed."""
        plugin = VirtualContextCompressionPluginV4(
            max_tokens=25,
            importance_threshold=0.7,
        )

        items = [
            {"id": "msg1", "content": "Routine status check step 1 completed successfully with all unit tests.", "importance": 0.2},
            {"id": "msg2", "content": "Routine status check step 2 in progress with background worker pool.", "importance": 0.3},
            {"id": "msg3", "content": "Routine status check step 3 verified across staging servers.", "importance": 0.2},
            {"id": "critical_rule", "content": "CRITICAL: System database password rotation rule active.", "importance": 0.9},
        ]

        compacted = await plugin.compact_context(items, target_max_tokens=20)

        # High importance critical rule must be preserved
        crit_present = any(it.get("id") == "critical_rule" for it in compacted)
        self.assertTrue(crit_present)

        # A compressed summary entry should have been created
        summary_present = any(it.get("is_compressed_summary") for it in compacted)
        self.assertTrue(summary_present)

        # Token count metrics should record savings
        self.assertGreater(plugin.metrics["tokens_saved"], 0)
        self.assertEqual(plugin.metrics["items_compressed"], 3)

    async def test_under_limit_no_compression(self):
        """When context is below limit, items should be preserved without modifications."""
        plugin = VirtualContextCompressionPluginV4(max_tokens=1000)

        items = [
            {"id": "1", "content": "Short text A"},
            {"id": "2", "content": "Short text B"},
        ]

        compacted = await plugin.compact_context(items)
        self.assertEqual(len(compacted), 2)
        self.assertEqual(plugin.metrics["tokens_saved"], 0)

    # -------------------------------------------------------------------------
    # 2. Paging to Mock Memory Stores
    # -------------------------------------------------------------------------
    async def test_paging_to_mock_episodic_and_working_memory(self):
        """Compressed candidate items should be paged to episodic memory and removed from working memory."""
        mock_working = MagicMock()
        mock_episodic = AsyncMock()

        plugin = VirtualContextCompressionPluginV4(
            max_tokens=10,
            importance_threshold=0.8,
            working_memory=mock_working,
            episodic_memory=mock_episodic,
        )

        items = [
            {"id": "w1", "content": "Old log 1 containing extensive background details", "importance": 0.1},
            {"id": "w2", "content": "Old log 2 containing extensive background details", "importance": 0.1},
            {"id": "w3", "content": "High priority goal with crucial action steps", "importance": 0.9},
        ]

        compacted = await plugin.compact_context(items, target_max_tokens=8, user_id=42)

        # Working memory should have removed w1 and w2
        self.assertEqual(mock_working.remove.call_count, 2)
        mock_working.remove.assert_any_call("w1", user_id=42)
        mock_working.remove.assert_any_call("w2", user_id=42)

        # Episodic memory should have received paged events
        self.assertEqual(mock_episodic.add_event.call_count, 2)

    # -------------------------------------------------------------------------
    # 3. LLM Summarization Integration
    # -------------------------------------------------------------------------
    async def test_llm_client_summarization_integration(self):
        """When LLMClient is supplied, its generate method should synthesize the summary."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "User initialized setup and authenticated."

        plugin = VirtualContextCompressionPluginV4(
            max_tokens=10,
            llm_client=mock_llm,
        )

        items = [
            {"id": "1", "content": "Step 1 started with full configuration.", "importance": 0.1},
            {"id": "2", "content": "Step 2 started with full configuration.", "importance": 0.1},
            {"id": "3", "content": "Step 3 started with full configuration.", "importance": 0.1},
        ]

        compacted = await plugin.compact_context(items, target_max_tokens=5)

        mock_llm.generate.assert_called_once()
        summary_item = next(it for it in compacted if it.get("is_compressed_summary"))
        self.assertIn("User initialized setup and authenticated.", summary_item["content"])

    # -------------------------------------------------------------------------
    # 4. ContextPlugin Lifecycle Methods
    # -------------------------------------------------------------------------
    async def test_context_plugin_lifecycle_methods(self):
        """Lifecycle hooks should execute cleanly."""
        plugin = VirtualContextCompressionPluginV4()

        await plugin.bootstrap({"max_tokens": 2000, "importance_threshold": 0.6})
        self.assertEqual(plugin.max_tokens, 2000)
        self.assertEqual(plugin.importance_threshold, 0.6)

        ingested = await plugin.ingest("Sample data", {})
        self.assertEqual(ingested, "Sample data")

        assembled = await plugin.assemble([{"content": "Entry 1"}, {"content": "Entry 2"}], {})
        self.assertEqual(assembled, "Entry 1\nEntry 2")

        q = plugin.before_retrieval("test query", user_id=1)
        self.assertEqual(q, "test query")

        ctx = plugin.after_retrieval(["item_a"], query="q", user_id=1)
        self.assertEqual(ctx, ["item_a"])


if __name__ == "__main__":
    unittest.main()
