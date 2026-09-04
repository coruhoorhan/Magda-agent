"""
Unit tests for Context Engine Semantic Clustering V2.
"""

import asyncio
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.memory.semantic_clustering_v2 import (
        ContextEngineClusteringResult,
        ContextEngineSemanticClusteringHookV2,
        MemoryItem,
        SemanticTagCluster,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "memory"
        / "semantic_clustering_v2.py"
    )
    spec = importlib.util.spec_from_file_location("semantic_clustering_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ContextEngineClusteringResult = module.ContextEngineClusteringResult
    ContextEngineSemanticClusteringHookV2 = module.ContextEngineSemanticClusteringHookV2
    MemoryItem = module.MemoryItem
    SemanticTagCluster = module.SemanticTagCluster


class MockMemoryStore:
    def __init__(self, entries):
        self.entries = list(entries)

    def get_all(self):
        return list(self.entries)

    def replace_all(self, new_entries):
        self.entries = list(new_entries)


class TestSemanticClusteringV2(unittest.TestCase):
    def setUp(self):
        self.hook = ContextEngineSemanticClusteringHookV2(min_cluster_size=2)

    def test_cluster_by_semantic_tags(self):
        memories = [
            {"id": "m1", "content": "User prefers concise Python code", "tags": ["preferences", "python"], "tokens": 15},
            {"id": "m2", "content": "User works in UTC+3 timezone", "tags": ["preferences", "location"], "tokens": 12},
            {"id": "m3", "content": "Refactored database connector to async sqlite", "tags": ["database", "sqlite"], "tokens": 20},
            {"id": "m4", "content": "Added connection pooling for sqlite", "tags": ["database", "performance"], "tokens": 18},
            {"id": "m5", "content": "Deployed API service", "tags": ["deployment"], "tokens": 10},
        ]

        clusters = self.hook.cluster_by_semantic_tags(memories)
        self.assertEqual(len(clusters), 3)
        self.assertIn("preferences", clusters)
        self.assertIn("database", clusters)
        self.assertIn("deployment", clusters)

        self.assertEqual(len(clusters["preferences"].items), 2)
        self.assertEqual(len(clusters["database"].items), 2)
        self.assertEqual(len(clusters["deployment"].items), 1)

    def test_compress_memory_context_and_token_reduction(self):
        memories = [
            {"id": "m1", "content": "User prefers concise code style without fluff", "tags": ["user_pref"], "tokens": 20},
            {"id": "m2", "content": "User prefers pytest over standard unittest", "tags": ["user_pref"], "tokens": 20},
            {"id": "m3", "content": "User prefers type hints everywhere", "tags": ["user_pref"], "tokens": 20},
            {"id": "m4", "content": "Singleton standalone note", "tags": ["misc"], "tokens": 10},
        ]

        result = self.hook.compress_memory_context(memories)
        self.assertEqual(result.initial_item_count, 4)
        self.assertEqual(result.initial_tokens, 70)
        # 3 user_pref items merged into 1 cluster + 1 singleton = 2 items
        self.assertEqual(result.compressed_item_count, 2)
        self.assertGreater(result.token_reduction, 0)
        self.assertLess(result.final_tokens, result.initial_tokens)

    def test_mock_memory_store_execution(self):
        raw_entries = [
            {"id": "d1", "content": "Feature A implemented successfully", "tags": ["progress"], "tokens": 15},
            {"id": "d2", "content": "Feature B implemented successfully", "tags": ["progress"], "tokens": 15},
            {"id": "d3", "content": "Feature C verified by smoke tests", "tags": ["progress"], "tokens": 15},
        ]
        store = MockMemoryStore(raw_entries)

        result = self.hook.execute_hook(store)
        self.assertTrue(result.token_reduction > 0)
        self.assertEqual(len(store.entries), 1)
        self.assertTrue(store.entries[0]["is_cluster_summary"])
        self.assertIn("PROGRESS", store.entries[0]["content"])

    def test_custom_summarizer(self):
        def custom_sum(topic, items):
            return f"CUSTOM_SUMMARY for {topic} covering {len(items)} items"

        custom_hook = ContextEngineSemanticClusteringHookV2(
            min_cluster_size=2,
            custom_summarizer=custom_sum,
        )

        memories = [
            {"id": "1", "content": "item 1", "tags": ["security"], "tokens": 10},
            {"id": "2", "content": "item 2", "tags": ["security"], "tokens": 10},
        ]
        res = custom_hook.compress_memory_context(memories)
        self.assertEqual(len(res.compressed_entries), 1)
        self.assertIn("CUSTOM_SUMMARY for security covering 2 items", res.compressed_entries[0]["content"])

    def test_async_hook_execution(self):
        async def run_async():
            store_list = [
                {"id": "1", "content": "task 1", "tags": ["task"], "tokens": 10},
                {"id": "2", "content": "task 2", "tags": ["task"], "tokens": 10},
            ]
            res = await self.hook.execute_hook_async(store_list)
            self.assertEqual(res.compressed_item_count, 1)
            self.assertEqual(len(store_list), 1)

        asyncio.run(run_async())


if __name__ == "__main__":
    unittest.main()
