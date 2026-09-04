"""
Unit tests for A2A Agent Peer Authentication V2.
"""

import time
import unittest

try:
    from magda_agent.integration.a2a_auth_v2 import (
        A2AAuthScope,
        A2AAuthV2,
        A2APeerAuthenticatorV2,
        A2APeerAuthTokenV2,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "integration"
        / "a2a_auth_v2.py"
    )
    spec = importlib.util.spec_from_file_location("a2a_auth_v2", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    A2AAuthScope = module.A2AAuthScope
    A2AAuthV2 = module.A2AAuthV2
    A2APeerAuthenticatorV2 = module.A2APeerAuthenticatorV2
    A2APeerAuthTokenV2 = module.A2APeerAuthTokenV2


class TestA2AAuthV2(unittest.TestCase):
    def setUp(self):
        self.secret = "my_shared_secret_key_1234567890123456"
        self.auth = A2APeerAuthenticatorV2(
            local_agent_id="magda_peer_receiver",
            shared_secret_key=self.secret,
        )

        # Peer generator
        self.peer_sender = A2APeerAuthenticatorV2(
            local_agent_id="external_peer_sender",
            shared_secret_key=self.secret,
        )

    def test_token_generation_and_verification(self):
        token = self.peer_sender.generate_token(
            target_agent_id="magda_peer_receiver",
            scopes=[A2AAuthScope.DELEGATE_EXECUTE.value],
            ttl_seconds=60,
        )

        token_str = token.to_token_string()
        is_valid, verified_tok, reason = self.auth.verify_token(token_str)

        self.assertTrue(is_valid)
        self.assertIsNotNone(verified_tok)
        self.assertEqual(verified_tok.issuer_agent_id, "external_peer_sender")
        self.assertEqual(verified_tok.target_agent_id, "magda_peer_receiver")

    def test_tampered_signature_rejected(self):
        token = self.peer_sender.generate_token("magda_peer_receiver")
        token.signature = "bad_tampered_signature_00000000"

        is_valid, _, reason = self.auth.verify_token(token)
        self.assertFalse(is_valid)
        self.assertIn("Invalid HMAC", reason)

    def test_expired_token_rejected(self):
        token = self.peer_sender.generate_token("magda_peer_receiver", ttl_seconds=-10)

        is_valid, _, reason = self.auth.verify_token(token)
        self.assertFalse(is_valid)
        self.assertIn("expired", reason)

    def test_replay_attack_rejected(self):
        token = self.peer_sender.generate_token("magda_peer_receiver")

        # 1st verification succeeds
        is_ok1, _, _ = self.auth.verify_token(token)
        self.assertTrue(is_ok1)

        # 2nd verification with same nonce fails
        is_ok2, _, reason = self.auth.verify_token(token)
        self.assertFalse(is_ok2)
        self.assertIn("replay detected", reason)

    def test_authenticate_incoming_request_dict(self):
        token = self.peer_sender.generate_token("magda_peer_receiver", scopes=[A2AAuthScope.DELEGATE_EXECUTE.value])

        # 1. Valid request via headers
        valid_request = {
            "headers": {"X-A2A-Auth-Token": token.to_token_string()},
            "payload": {"task": "do_work"},
        }
        ok, claims, msg = self.auth.authenticate_request(valid_request, required_scope=A2AAuthScope.DELEGATE_EXECUTE.value)
        self.assertTrue(ok)
        self.assertEqual(claims["issuer"], "external_peer_sender")

        # 2. Missing token request
        bad_request = {"headers": {}, "payload": {}}
        bad_ok, _, _ = self.auth.authenticate_request(bad_request)
        self.assertFalse(bad_ok)

    def test_legacy_mtls_class(self):
        legacy = A2AAuthV2(cert_path="", key_path="")
        self.assertFalse(legacy.authenticate())


if __name__ == "__main__":
    unittest.main()
