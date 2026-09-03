"""
Unit tests for A2A Agent Discovery Cards Integration V3.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.integration.a2a_cards_v3 import (
        A2ADiscoveryCardsV3,
        AgentCardResolverV3,
        AgentCardV3,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "integration"
        / "a2a_cards_v3.py"
    )
    spec = importlib.util.spec_from_file_location("a2a_cards_v3", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    A2ADiscoveryCardsV3 = module.A2ADiscoveryCardsV3
    AgentCardResolverV3 = module.AgentCardResolverV3
    AgentCardV3 = module.AgentCardV3


class TestA2ACardsV3(unittest.TestCase):
    def setUp(self):
        self.service = A2ADiscoveryCardsV3()

    def test_parse_valid_agent_card(self):
        data = {
            "agent_id": "coder_agent_01",
            "name": "Code Specialist",
            "description": "Writes clean python code",
            "capabilities": ["python", "code_review", "refactor"],
            "endpoints": {"http": "https://coder.local:8080"},
        }

        card = AgentCardV3.from_dict(data)
        self.assertEqual(card.agent_id, "coder_agent_01")
        self.assertEqual(card.name, "Code Specialist")
        self.assertTrue(card.has_capability("python"))
        self.assertTrue(card.has_capability("code_review"))
        self.assertFalse(card.has_capability("kubernetes"))

    def test_capability_matching_methods(self):
        card = AgentCardV3(
            agent_id="test_id",
            name="Test Agent",
            description="",
            capabilities=["analysis", "synthesis", "search"],
            endpoints={},
        )

        self.assertTrue(card.matches_any_capability(["search", "database"]))
        self.assertFalse(card.matches_any_capability(["database", "docker"]))

        self.assertTrue(card.matches_all_capabilities(["analysis", "search"]))
        self.assertFalse(card.matches_all_capabilities(["analysis", "docker"]))

    def test_resolve_endpoint_with_mock_http(self):
        async def run_async():
            mock_http = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "agent_id": "remote_agent_99",
                "name": "Remote Node",
                "description": "Remote mesh participant",
                "capabilities": ["remote_exec"],
                "endpoints": {"rpc": "rpc://192.168.1.50:9000"},
            }
            mock_http.get = AsyncMock(return_value=mock_response)

            resolver = AgentCardResolverV3()
            card = await resolver.resolve_endpoint_async("http://remote.node", mock_http)

            self.assertIsNotNone(card)
            self.assertEqual(card.agent_id, "remote_agent_99")
            self.assertTrue(card.has_capability("remote_exec"))
            mock_http.get.assert_called_once_with("http://remote.node/agent.json")

        asyncio.run(run_async())

    def test_invalid_card_handling(self):
        # Missing agent_id
        invalid_data = {"name": "No ID"}
        success, card, msg = self.service.parse_and_ingest_card(invalid_data)
        self.assertFalse(success)
        self.assertIsNone(card)

        # Malformed JSON string
        success_json, card_json, msg_json = self.service.parse_and_ingest_card("{bad json")
        self.assertFalse(success_json)
        self.assertIsNone(card_json)

    def test_discover_peers_and_query_by_capability(self):
        async def run_async():
            mock_http = MagicMock()

            def mock_get_impl(url):
                mock_resp = MagicMock()
                if "peer1" in url:
                    mock_resp.json.return_value = {
                        "agent_id": "p1",
                        "name": "Peer 1",
                        "description": "",
                        "capabilities": ["vision", "ocr"],
                        "endpoints": {},
                    }
                else:
                    mock_resp.json.return_value = {
                        "agent_id": "p2",
                        "name": "Peer 2",
                        "description": "",
                        "capabilities": ["nlp", "translation"],
                        "endpoints": {},
                    }
                return mock_resp

            mock_http.get = AsyncMock(side_effect=mock_get_impl)

            discovered = await self.service.discover_peers_async(
                ["http://peer1.local", "http://peer2.local"],
                http_client=mock_http,
            )

            self.assertEqual(len(discovered), 2)
            self.assertEqual(len(self.service.get_all_peers()), 2)

            vision_peers = self.service.find_peers_by_capability("vision")
            self.assertEqual(len(vision_peers), 1)
            self.assertEqual(vision_peers[0].agent_id, "p1")

            nlp_peers = self.service.find_peers_by_capability("translation")
            self.assertEqual(len(nlp_peers), 1)
            self.assertEqual(nlp_peers[0].agent_id, "p2")

        asyncio.run(run_async())


if __name__ == "__main__":
    unittest.main()
