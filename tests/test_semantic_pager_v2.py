"""
Tests for MemGPT Semantic Cluster Virtual Context Pager v2.
"""

import asyncio
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.memory.semantic_pager_v2 import (
        MemGPTSemanticClusterPagerV2,
        SemanticCluster,
        ClusterCompactionStrategy,
        cosine_similarity,
        default_local_embed,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "memory" / "semantic_pager_v2.py"
    spec = importlib.util.spec_from_file_location("semantic_pager_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    MemGPTSemanticClusterPagerV2 = module.MemGPTSemanticClusterPagerV2
    SemanticCluster = module.SemanticCluster
    ClusterCompactionStrategy = module.ClusterCompactionStrategy
    cosine_similarity = module.cosine_similarity
    default_local_embed = module.default_local_embed


class TestSemanticPagerV2(unittest.IsolatedAsyncioTestCase):
    """
    Comprehensive test suite verifying local embeddings clustering,
    thematic discovery, and MemGPT virtual context compaction.
    """

    # -------------------------------------------------------------------------
    # 1. Cosine Similarity & Vector Math
    # -------------------------------------------------------------------------
    def test_cosine_similarity_values(self):
        """Verify vector cosine math for identical, orthogonal, and opposite vectors."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        v4 = [-1.0, 0.0, 0.0]

        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0, places=4)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0, places=4)
        self.assertAlmostEqual(cosine_similarity(v1, v4), -1.0, places=4)
        self.assertEqual(cosine_similarity([], []), 0.0)

    # -------------------------------------------------------------------------
    # 2. Semantic Clustering of Thematic Groups
    # -------------------------------------------------------------------------
    async def test_cluster_memories_thematic_grouping(self):
        """Memories with distinct topics should form separate semantic clusters."""
        # Custom mock embedding function mapping topics to orthogonal vectors
        def mock_embed(text: str):
            t_lower = text.lower()
            if "database" in t_lower or "sql" in t_lower or "postgres" in t_lower:
                return [1.0, 0.0, 0.0, 0.0]
            elif "weather" in t_lower or "rain" in t_lower or "sunny" in t_lower:
                return [0.0, 1.0, 0.0, 0.0]
            else:
                return [0.0, 0.0, 1.0, 0.0]

        pager = MemGPTSemanticClusterPagerV2(embedding_fn=mock_embed, similarity_threshold=0.8)

        memories = [
            {"id": "m1", "content": "Postgres database connection established on port 5432."},
            {"id": "m2", "content": "Database query performance tuned with new SQL index."},
            {"id": "m3", "content": "Today weather forecast shows rain and clouds in London."},
            {"id": "m4", "content": "Tomorrow will be sunny and warm across the city."},
        ]

        clusters = await pager.cluster_memories(memories)
        self.assertEqual(len(clusters), 2)

        # One cluster should have 2 database memories, one should have 2 weather memories
        cluster_sizes = sorted([len(c.members) for c in clusters])
        self.assertEqual(cluster_sizes, [2, 2])

    # -------------------------------------------------------------------------
    # 3. Context Compaction via Cluster Summarization
    # -------------------------------------------------------------------------
    async def test_compact_context_summarizes_redundant_cluster(self):
        """When token limit is exceeded, redundant clusters should be replaced with summary."""
        def mock_embed(text: str):
            if "security" in text.lower() or "auth" in text.lower():
                return [1.0, 0.0]
            return [0.0, 1.0]

        pager = MemGPTSemanticClusterPagerV2(
            max_tokens=30,  # Small limit to force compaction
            embedding_fn=mock_embed,
            compaction_strategy=ClusterCompactionStrategy.SUMMARIZE_CLUSTER,
        )

        memories = [
            {"id": "sec_1", "content": "Security audit log recorded successful user authentication."},
            {"id": "sec_2", "content": "Security certificate renewed for OAuth auth gateway server."},
            {"id": "sec_3", "content": "Security firewall updated with latest auth whitelist IP rules."},
            {"id": "indep_1", "content": "General user preferences loaded for dark mode theme."},
        ]

        compacted = await pager.compact_context(memories, target_max_tokens=35)

        # Should contain the summary record and the independent record
        summary_items = [item for item in compacted if isinstance(item, dict) and item.get("is_summary")]
        self.assertEqual(len(summary_items), 1)
        self.assertIn("Cluster Summary (3 events)", summary_items[0]["content"])

        # Total items should be reduced from 4 to 2 (1 summary + 1 untouched)
        self.assertEqual(len(compacted), 2)
        self.assertGreater(pager.metrics["tokens_saved"], 0)

    # -------------------------------------------------------------------------
    # 4. Context Compaction via Evict Redundant Strategy
    # -------------------------------------------------------------------------
    async def test_compact_context_evict_redundant_strategy(self):
        """EVICT_REDUNDANT mode should retain only the latest representative."""
        def mock_embed(text: str):
            return [1.0, 0.0]

        pager = MemGPTSemanticClusterPagerV2(
            max_tokens=20,
            embedding_fn=mock_embed,
            compaction_strategy=ClusterCompactionStrategy.EVICT_REDUNDANT,
        )

        memories = [
            {"id": "dup_1", "content": "Duplicate status heartbeat report 1"},
            {"id": "dup_2", "content": "Duplicate status heartbeat report 2"},
            {"id": "dup_3", "content": "Duplicate status heartbeat report 3"},
        ]

        compacted = await pager.compact_context(memories, target_max_tokens=15)
        # Should keep only 1 representative item
        self.assertEqual(len(compacted), 1)
        self.assertEqual(compacted[0]["id"], "dup_3")

    # -------------------------------------------------------------------------
    # 5. Under-Limit No-Op Behavior
    # -------------------------------------------------------------------------
    async def test_under_limit_no_compaction(self):
        """When total tokens are within limit, context should be preserved intact."""
        pager = MemGPTSemanticClusterPagerV2(max_tokens=5000)

        memories = [
            {"id": "1", "content": "Short memory item 1"},
            {"id": "2", "content": "Short memory item 2"},
        ]

        compacted = await pager.compact_context(memories)
        self.assertEqual(len(compacted), 2)
        self.assertEqual(pager.metrics["tokens_saved"], 0)

    # -------------------------------------------------------------------------
    # 6. ContextPlugin Protocol Lifecycle Hooks
    # -------------------------------------------------------------------------
    async def test_context_plugin_protocol_lifecycle(self):
        """Verify standard lifecycle hook execution."""
        pager = MemGPTSemanticClusterPagerV2()

        # Bootstrap
        await pager.bootstrap({"max_tokens": 1500, "similarity_threshold": 0.85})
        self.assertEqual(pager.max_tokens, 1500)
        self.assertEqual(pager.similarity_threshold, 0.85)

        # Ingest & Assemble
        content = await pager.ingest("New content", {})
        self.assertEqual(content, "New content")

        assembled = await pager.assemble([{"content": "Line A"}, {"content": "Line B"}], {})
        self.assertEqual(assembled, "Line A\nLine B")

        # Query & context pass-through
        q = pager.before_retrieval("search query", user_id=1)
        self.assertEqual(q, "search query")

        ctx = pager.after_retrieval(["item1"], query="q", user_id=1)
        self.assertEqual(ctx, ["item1"])


if __name__ == "__main__":
    unittest.main()
