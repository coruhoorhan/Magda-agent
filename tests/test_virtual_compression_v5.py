"""
Unit tests for MemGPT Virtual Context Semantic Compressor V5.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.memory.virtual_compression_v5 import (
        EpisodicChunk,
        MemGPTVirtualContextSemanticCompressorV5,
        SemanticCluster,
        SemanticFact,
        VirtualContextCompressionResult,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "memory"
        / "virtual_compression_v5.py"
    )
    spec = importlib.util.spec_from_file_location("virtual_compression_v5", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    EpisodicChunk = module.EpisodicChunk
    MemGPTVirtualContextSemanticCompressorV5 = module.MemGPTVirtualContextSemanticCompressorV5
    SemanticCluster = module.SemanticCluster
    SemanticFact = module.SemanticFact
    VirtualContextCompressionResult = module.VirtualContextCompressionResult


class TestVirtualCompressionV5(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.compressor = MemGPTVirtualContextSemanticCompressorV5(
            llm_client=self.mock_llm,
            working_memory_limit_tokens=500,
            compression_threshold_ratio=0.5,
            min_chunks_for_compression=2,
        )

    def test_add_episodic_chunk_and_token_estimation(self):
        chunk = self.compressor.add_episodic_chunk("The user prefers python code style", source="user")
        self.assertIsNotNone(chunk.chunk_id)
        self.assertGreater(chunk.token_count, 0)
        self.assertEqual(len(self.compressor._episodic_buffer), 1)

    def test_should_compress_threshold(self):
        self.assertFalse(self.compressor.should_compress())

        # Add heavy chunks
        self.compressor.add_episodic_chunk("word " * 150)
        self.compressor.add_episodic_chunk("word " * 150)

        self.assertTrue(self.compressor.should_compress())

    def test_clustering_chunks_by_topic(self):
        c1 = EpisodicChunk(content="The user timezone is UTC+3 and name is Orhan", source="user")
        c2 = EpisodicChunk(content="Refactor the architecture pipeline and schema", source="agent")
        c3 = EpisodicChunk(content="Add unit test and verification for completed task", source="system")

        clusters = self.compressor.cluster_episodic_chunks([c1, c2, c3])
        self.assertIn("user_profile", clusters)
        self.assertIn("architecture", clusters)
        self.assertIn("task_progress", clusters)

    def test_compression_with_mock_llm(self):
        async def run_async():
            mock_llm_response = json.dumps({
                "summary": "User Profile Summary",
                "facts": [
                    "User timezone is UTC+3",
                    "User name is Orhan",
                    "User prefers concise code",
                ],
            })

            # Mock LLM async generate
            self.mock_llm.generate = AsyncMock(return_value=mock_llm_response)

            self.compressor.add_episodic_chunk(
                content="User lives in UTC+3 timezone.",
                source="user",
                metadata={"topic": "user_profile"},
            )
            self.compressor.add_episodic_chunk(
                content="User's name is Orhan and dislikes boilerplate.",
                source="user",
                metadata={"topic": "user_profile"},
            )

            res = await self.compressor.compress_episodic_to_semantic()

            self.assertEqual(res.original_chunk_count, 2)
            self.assertEqual(res.compressed_fact_count, 3)
            self.assertEqual(len(res.clusters), 1)

            # Check facts stored
            facts = self.compressor.get_semantic_facts("user_profile")
            self.assertEqual(len(facts), 3)
            self.assertEqual(facts[0].fact_statement, "User timezone is UTC+3")

            # Episodic buffer should be cleared
            self.assertEqual(len(self.compressor._episodic_buffer), 0)

        asyncio.run(run_async())

    def test_fallback_heuristic_compression_without_llm(self):
        compressor_no_llm = MemGPTVirtualContextSemanticCompressorV5(llm_client=None)

        compressor_no_llm.add_episodic_chunk(
            content="Fact A: Python 3.12 is supported\nFact B: Linux environment",
            source="system",
            metadata={"topic": "technical_stack"},
        )

        res = compressor_no_llm.compress_sync()
        self.assertEqual(res.original_chunk_count, 1)
        self.assertEqual(res.compressed_fact_count, 2)
        facts = compressor_no_llm.get_semantic_facts("technical_stack")
        self.assertEqual(len(facts), 2)
        self.assertIn("Fact A", facts[0].fact_statement)


if __name__ == "__main__":
    unittest.main()
