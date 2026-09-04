"""
Unit tests for Claude Agent Teams Context Compression V4.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.architecture.context_compression_v4 import (
        ClaudeAgentTeamsContextCompressorV4,
        ContextCompressionResultV4,
        HierarchicalContextNode,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "architecture"
        / "context_compression_v4.py"
    )
    spec = importlib.util.spec_from_file_location("context_compression_v4", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ClaudeAgentTeamsContextCompressorV4 = module.ClaudeAgentTeamsContextCompressorV4
    ContextCompressionResultV4 = module.ContextCompressionResultV4
    HierarchicalContextNode = module.HierarchicalContextNode


class TestContextCompressionV4(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.compressor = ClaudeAgentTeamsContextCompressorV4(
            llm_client=self.mock_llm,
            max_prompt_chars=1200,
        )

    def test_extract_constraints_and_contracts(self):
        raw_text = """
This is standard conversation text.
You MUST never use external shell commands without validation.
The user is asking for optimization.
Contract: class PaymentGatewayInterface: def charge(): pass
Never delete database files without authorization.
"""
        constraints, contracts = self.compressor.extract_constraints_and_contracts(raw_text)

        self.assertGreater(len(constraints), 0)
        self.assertTrue(any("MUST never" in c for c in constraints))
        self.assertTrue(any("PaymentGatewayInterface" in c for c in contracts))

    def test_hierarchical_context_formatting(self):
        node = HierarchicalContextNode(
            depth_level=2,
            role="coder",
            task_objective="Implement payment webhook",
            ancestor_context_summary="User requested Stripe integration.",
            critical_constraints=["Must validate HMAC header."],
            contract_interfaces=["def handle_webhook(payload: dict) -> bool:"],
        )

        formatted = node.to_formatted_context()

        self.assertIn("## Subagent Context [Depth: 2] [Role: CODER]", formatted)
        self.assertIn("Task Objective: Implement payment webhook", formatted)
        self.assertIn("Must validate HMAC header.", formatted)
        self.assertIn("def handle_webhook", formatted)

    def test_compress_context_with_mock_llm(self):
        async def run_async():
            self.mock_llm.generate = AsyncMock(
                return_value="Orchestrator initiated full multi-agent refactor of the caching subsystem."
            )

            root_ctx = "Long context with details " * 100 + "\nYou MUST strictly run tests."
            parent_sum = "Lead subagent planned 3 steps."
            task = "Write cache invalidation logic"

            res = await self.compressor.compress_hierarchical_context_async(
                root_context=root_ctx,
                parent_summary=parent_sum,
                subagent_task=task,
                subagent_role="coder",
                depth_level=2,
            )

            self.assertLess(res.compressed_char_count, res.original_char_count)
            self.assertLess(res.compression_ratio, 0.5)
            self.assertIn("MUST strictly run tests", res.condensed_prompt)

        asyncio.run(run_async())

    def test_heuristic_compression_without_llm(self):
        compressor_no_llm = ClaudeAgentTeamsContextCompressorV4(llm_client=None, max_prompt_chars=500)

        root_ctx = "Very long root context with background " * 50 + "\nNever commit directly to main."
        res = compressor_no_llm.compress_hierarchical_context(
            root_context=root_ctx,
            parent_summary="Parent summary",
            subagent_task="Review security",
            subagent_role="reviewer",
        )

        self.assertLessEqual(res.compressed_char_count, 600)
        self.assertIn("Never commit directly to main", res.condensed_prompt)


if __name__ == "__main__":
    unittest.main()
