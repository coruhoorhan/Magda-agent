"""
Unit tests for MCP Dynamic Skill Marketplace Poller V2.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.integration.mcp_marketplace_poller_v2 import (
        MarketplaceSkillEntryV2,
        MCPMarketplacePollerV2,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "integration"
        / "mcp_marketplace_poller_v2.py"
    )
    spec = importlib.util.spec_from_file_location("mcp_marketplace_poller_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    MarketplaceSkillEntryV2 = module.MarketplaceSkillEntryV2
    MCPMarketplacePollerV2 = module.MCPMarketplacePollerV2


class MockMCPRegistryV2:
    def __init__(self):
        self.loaded_tools = []

    def load_tool(self, tool_dict):
        self.loaded_tools.append(tool_dict)
        return True


class TestMCPMarketplacePollerV2(unittest.TestCase):
    def setUp(self):
        self.mock_registry = MockMCPRegistryV2()
        self.mock_scheduler = MagicMock()
        self.poller = MCPMarketplacePollerV2(
            registry=self.mock_registry,
            scheduler=self.mock_scheduler,
            marketplace_url="https://api.agentskills.io/v2/skills",
        )

    def test_parse_agentskills_json_formats(self):
        # Format 1: Direct list
        list_payload = [
            {
                "name": "csv_parser",
                "description": "Parses CSV streams",
                "version": "1.2.0",
                "author": "alice",
                "parameters": {"type": "object", "properties": {"delimiter": {"type": "string"}}},
                "tags": ["data", "csv"],
            }
        ]

        entries, errors = self.poller.parse_marketplace_payload(list_payload)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].name, "csv_parser")
        self.assertEqual(entries[0].author, "alice")
        self.assertEqual(len(errors), 0)

        # Format 2: Wrapped {skills: [...]}
        wrapped_payload = {
            "skills": [
                {
                    "name": "pdf_reader",
                    "description": "Extracts text from PDF",
                    "category": "document",
                }
            ]
        }
        entries2, errors2 = self.poller.parse_marketplace_payload(wrapped_payload)
        self.assertEqual(len(entries2), 1)
        self.assertEqual(entries2[0].name, "pdf_reader")

    def test_sync_marketplace_with_mock_http(self):
        async def run_async():
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "tools": [
                    {
                        "name": "image_resizer",
                        "description": "Resizes images",
                        "version": "2.0.0",
                        "inputSchema": {"type": "object", "properties": {"width": {"type": "integer"}}},
                    },
                    {
                        "name": "markdown_linter",
                        "description": "Lints markdown documents",
                    },
                ]
            }
            mock_client.get = AsyncMock(return_value=mock_resp)

            count, entries, errors = await self.poller.sync_marketplace_async(http_client=mock_client)

            self.assertEqual(count, 2)
            self.assertEqual(len(entries), 2)
            self.assertEqual(len(self.mock_registry.loaded_tools), 2)
            self.assertEqual(self.mock_registry.loaded_tools[0]["name"], "image_resizer")
            mock_client.get.assert_called_once_with("https://api.agentskills.io/v2/skills")

        asyncio.run(run_async())

    def test_schedule_sync(self):
        scheduled = self.poller.schedule_sync("0 12 * * *")
        self.assertTrue(scheduled)
        self.mock_scheduler.schedule.assert_called_once()
        args = self.mock_scheduler.schedule.call_args[0]
        self.assertEqual(args[0], "0 12 * * *")

    def test_error_handling_malformed_json(self):
        entries, errors = self.poller.parse_marketplace_payload("{invalid json")
        self.assertEqual(len(entries), 0)
        self.assertGreater(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
