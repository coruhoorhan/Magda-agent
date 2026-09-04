"""
Tests for A2A Discovery Agent Card Broadcaster V4.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.integration.a2a_discovery_v4 import (
        AgentCardV4,
        A2ADiscoveryRegistryV4,
        A2ADiscoveryAgentCardBroadcasterV4,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "integration" / "a2a_discovery_v4.py"
    spec = importlib.util.spec_from_file_location("a2a_discovery_v4", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    AgentCardV4 = module.AgentCardV4
    A2ADiscoveryRegistryV4 = module.A2ADiscoveryRegistryV4
    A2ADiscoveryAgentCardBroadcasterV4 = module.A2ADiscoveryAgentCardBroadcasterV4


class TestA2ADiscoveryV4(unittest.IsolatedAsyncioTestCase):
    """
    Test suite verifying AgentCardV4 formatting, network payload generation,
    discovery broadcasting, and capability registry queries.
    """

    def setUp(self):
        self.local_card = AgentCardV4(
            agent_id="agent_coder_v4",
            name="Coder Agent Prime",
            description="Specialized code implementation and unit testing subagent",
            capabilities=["python_coding", "unit_testing", "refactoring"],
            endpoints={"rest": "http://10.0.1.5:8000/a2a", "grpc": "10.0.1.5:50051"},
        )
        self.registry = A2ADiscoveryRegistryV4()

    # -------------------------------------------------------------------------
    # 1. Card Formatting & Serialization
    # -------------------------------------------------------------------------
    def test_agent_card_v4_serialization_and_capabilities(self):
        """Card should serialize to/from JSON and accurately match capabilities."""
        json_str = self.local_card.to_json()
        deserialized = AgentCardV4.from_json(json_str)

        self.assertEqual(deserialized.agent_id, "agent_coder_v4")
        self.assertEqual(deserialized.name, "Coder Agent Prime")
        self.assertTrue(deserialized.has_capability("python_coding"))
        self.assertTrue(deserialized.has_capability("testing"))
        self.assertFalse(deserialized.has_capability("web_scraping"))

    # -------------------------------------------------------------------------
    # 2. Broadcast Payload Format
    # -------------------------------------------------------------------------
    def test_generate_broadcast_payload(self):
        """Broadcast payload must conform to A2A V4 protocol specifications."""
        broadcaster = A2ADiscoveryAgentCardBroadcasterV4(local_card=self.local_card)
        payload = broadcaster.generate_broadcast_payload()

        self.assertEqual(payload["type"], "a2a_discovery_broadcast")
        self.assertEqual(payload["protocol"], "a2a_v4")
        self.assertTrue(payload["broadcast_id"].startswith("bcast_"))
        self.assertIn("capabilities", payload)
        self.assertEqual(payload["capabilities"], ["python_coding", "unit_testing", "refactoring"])
        self.assertEqual(payload["agent_card"]["agent_id"], "agent_coder_v4")

    # -------------------------------------------------------------------------
    # 3. Broadcasting with Transport
    # -------------------------------------------------------------------------
    async def test_broadcast_once_invokes_transport(self):
        """Single broadcast should invoke the network transport callback."""
        mock_transport = AsyncMock()
        broadcaster = A2ADiscoveryAgentCardBroadcasterV4(
            local_card=self.local_card,
            broadcast_transport_fn=mock_transport,
        )

        sent_payload = await broadcaster.broadcast_once()

        mock_transport.assert_called_once_with(sent_payload)
        self.assertEqual(broadcaster.broadcast_count, 1)
        self.assertIsNotNone(broadcaster.last_broadcast_time)

    # -------------------------------------------------------------------------
    # 4. Remote Broadcast Reception & Registry
    # -------------------------------------------------------------------------
    def test_receive_remote_broadcast_and_registry_query(self):
        """Receiving a remote broadcast should register the peer in the local discovery registry."""
        broadcaster = A2ADiscoveryAgentCardBroadcasterV4(
            local_card=self.local_card,
            registry=self.registry,
        )

        remote_payload = {
            "type": "a2a_discovery_broadcast",
            "protocol": "a2a_v4",
            "agent_card": {
                "agent_id": "agent_surfer_01",
                "name": "Web Surfer Subagent",
                "description": "Searches documentation and web",
                "capabilities": ["web_search", "scraping"],
                "endpoints": {"rest": "http://10.0.1.9:8000/a2a"},
            },
        }

        registered_card = broadcaster.receive_remote_broadcast(remote_payload)
        self.assertIsNotNone(registered_card)
        self.assertEqual(registered_card.agent_id, "agent_surfer_01")

        # Query registry
        surfer_agents = self.registry.find_agents_by_capability("web_search")
        self.assertEqual(len(surfer_agents), 1)
        self.assertEqual(surfer_agents[0].name, "Web Surfer Subagent")

    # -------------------------------------------------------------------------
    # 5. Raw Cards Parsing
    # -------------------------------------------------------------------------
    def test_parse_and_register_cards(self):
        """Batch JSON parsing of agent cards should populate the registry."""
        card1_json = AgentCardV4(
            agent_id="a1", name="Agent 1", description="desc", capabilities=["c1"], endpoints={}
        ).to_json()
        card2_json = AgentCardV4(
            agent_id="a2", name="Agent 2", description="desc", capabilities=["c2"], endpoints={}
        ).to_json()

        parsed = self.registry.parse_and_register_cards([card1_json, card2_json, "invalid json"])
        self.assertEqual(len(parsed), 2)
        self.assertEqual(len(self.registry.get_all_agents()), 2)

    # -------------------------------------------------------------------------
    # 6. Periodic Broadcast Loop Lifecycle
    # -------------------------------------------------------------------------
    async def test_periodic_broadcaster_start_and_stop(self):
        """Broadcaster loop should run periodically and terminate cleanly on stop."""
        mock_transport = AsyncMock()
        broadcaster = A2ADiscoveryAgentCardBroadcasterV4(
            local_card=self.local_card,
            broadcast_transport_fn=mock_transport,
            broadcast_interval_seconds=0.01,
        )

        await broadcaster.start_broadcasting()
        await asyncio.sleep(0.035)
        broadcaster.stop_broadcasting()

        self.assertGreaterEqual(broadcaster.broadcast_count, 2)
        self.assertGreaterEqual(mock_transport.call_count, 2)


if __name__ == "__main__":
    unittest.main()
