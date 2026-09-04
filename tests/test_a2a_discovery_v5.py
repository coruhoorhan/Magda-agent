"""
Unit tests for A2A Agent Card Dynamic Validation V5.
"""

import json
import unittest
from unittest.mock import MagicMock

try:
    from magda_agent.integration.a2a_discovery_v5 import (
        A2ADiscoveryRegistryV5,
        A2ADiscoveryV5,
        AgentCardSchemaValidatorV5,
        AgentCardV5,
        AgentSecurityTier,
        ValidationResult,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "integration"
        / "a2a_discovery_v5.py"
    )
    spec = importlib.util.spec_from_file_location("a2a_discovery_v5", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    A2ADiscoveryRegistryV5 = module.A2ADiscoveryRegistryV5
    A2ADiscoveryV5 = module.A2ADiscoveryV5
    AgentCardSchemaValidatorV5 = module.AgentCardSchemaValidatorV5
    AgentCardV5 = module.AgentCardV5
    AgentSecurityTier = module.AgentSecurityTier
    ValidationResult = module.ValidationResult


class TestA2ADiscoveryV5(unittest.TestCase):
    def setUp(self):
        self.validator = AgentCardSchemaValidatorV5()
        self.rejection_callback = MagicMock()
        self.registry = A2ADiscoveryRegistryV5(
            validator=self.validator,
            on_rejection_callback=self.rejection_callback,
        )
        self.discovery = A2ADiscoveryV5(registry=self.registry)

    def test_valid_agent_card_dict_validation(self):
        valid_card_dict = {
            "agent_id": "agent_alpha_01",
            "name": "Alpha Coder Agent",
            "description": "Specialist in python refactoring",
            "capabilities": ["code_generation", "refactoring"],
            "endpoints": {"http": "https://agent-alpha.internal/api/v5"},
            "protocol_version": "v5",
            "security_tier": "enterprise",
            "metadata": {"version": "1.0.0"},
        }

        res = self.validator.validate(valid_card_dict)
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.errors), 0)

        success, card, errors = self.registry.register_agent(valid_card_dict)
        self.assertTrue(success)
        self.assertIsNotNone(card)
        self.assertEqual(card.agent_id, "agent_alpha_01")
        self.assertEqual(card.security_tier, AgentSecurityTier.ENTERPRISE)

    def test_valid_agent_card_json_string(self):
        json_str = json.dumps({
            "agent_id": "agent_beta_02",
            "name": "Beta Research Agent",
            "description": "Deep researcher",
            "capabilities": ["web_search", "synthesis"],
            "endpoints": {"grpc": "grpc://agent-beta:50051"},
        })

        success, card, errors = self.registry.register_agent(json_str)
        self.assertTrue(success)
        self.assertIsNotNone(card)
        self.assertEqual(card.agent_id, "agent_beta_02")
        self.assertTrue(card.has_capability("web_search"))

    def test_invalid_cards_rejected_and_quarantined(self):
        # 1. Missing required field (endpoints)
        invalid_missing_endpoints = {
            "agent_id": "agent_bad_01",
            "name": "Bad Agent",
            "capabilities": ["chat"],
        }
        res1 = self.validator.validate(invalid_missing_endpoints)
        self.assertFalse(res1.is_valid)
        self.assertTrue(any("endpoints" in err for err in res1.errors))

        # Register through registry
        success, card, errors = self.registry.register_agent(invalid_missing_endpoints)
        self.assertFalse(success)
        self.assertIsNone(card)
        self.rejection_callback.assert_called_once()

        # 2. Empty capabilities list
        invalid_empty_caps = {
            "agent_id": "agent_bad_02",
            "name": "No Caps Agent",
            "capabilities": [],
            "endpoints": {"http": "http://localhost:8000"},
        }
        res2 = self.validator.validate(invalid_empty_caps)
        self.assertFalse(res2.is_valid)
        self.assertTrue(any("capabilities" in err for err in res2.errors))

        # 3. Invalid agent_id format (spaces / special chars)
        invalid_id = {
            "agent_id": "bad id with spaces!",
            "name": "Invalid ID Agent",
            "capabilities": ["test"],
            "endpoints": {"http": "http://localhost:8000"},
        }
        res3 = self.validator.validate(invalid_id)
        self.assertFalse(res3.is_valid)
        self.assertTrue(any("agent_id" in err for err in res3.errors))

        # 4. Malformed JSON string
        malformed_json = "{'agent_id': 'bad_json', 'name': unquoted}"
        success_json, _, errs_json = self.registry.register_agent(malformed_json)
        self.assertFalse(success_json)
        self.assertTrue(any("Malformed JSON" in e for e in errs_json))

        # Verify quarantine list contains all rejections
        quarantine = self.registry.get_quarantined_cards()
        self.assertGreaterEqual(len(quarantine), 2)

    def test_batch_parse_and_register_cards(self):
        cards = [
            json.dumps({
                "agent_id": "valid_agent_01",
                "name": "Valid 1",
                "capabilities": ["calc"],
                "endpoints": {"ipc": "/tmp/agent1.sock"},
            }),
            json.dumps({
                "agent_id": "invalid_no_name",
                "capabilities": ["calc"],
                "endpoints": {"ipc": "/tmp/agent2.sock"},
            }),
            json.dumps({
                "agent_id": "valid_agent_02",
                "name": "Valid 2",
                "capabilities": ["calc", "stats"],
                "endpoints": {"ipc": "/tmp/agent3.sock"},
            }),
        ]

        valid_cards, rejected = self.registry.parse_and_register_cards(cards)
        self.assertEqual(len(valid_cards), 2)
        self.assertEqual(len(rejected), 1)

        # Check find by capability
        stats_agents = self.registry.find_agents_by_capability("stats")
        self.assertEqual(len(stats_agents), 1)
        self.assertEqual(stats_agents[0].agent_id, "valid_agent_02")

    def test_discovery_orchestrator_ingestion(self):
        card_data = {
            "agent_id": "mesh_agent_99",
            "name": "Mesh Node 99",
            "capabilities": ["router"],
            "endpoints": {"mesh": "mesh://node99.a2a"},
        }
        success, card, errors = self.discovery.ingest_discovered_card(card_data)
        self.assertTrue(success)
        self.assertIsNotNone(card)

        active = self.discovery.get_active_mesh_agents()
        self.assertEqual(len(active), 1)


if __name__ == "__main__":
    unittest.main()
