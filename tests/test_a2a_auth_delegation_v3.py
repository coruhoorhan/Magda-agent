"""
Unit tests for A2A Workflow Auth Delegation Manager V3.
"""

import time
import unittest

try:
    from magda_agent.integration.a2a_auth_delegation_v3 import (
        A2ADelegationTokenV3,
        A2AWorkflowAuthDelegationManagerV3,
        DelegationScopeV3,
        TokenExchangeResult,
        TokenStatusV3,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "integration"
        / "a2a_auth_delegation_v3.py"
    )
    spec = importlib.util.spec_from_file_location("a2a_auth_delegation_v3", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    A2ADelegationTokenV3 = module.A2ADelegationTokenV3
    A2AWorkflowAuthDelegationManagerV3 = module.A2AWorkflowAuthDelegationManagerV3
    DelegationScopeV3 = module.DelegationScopeV3
    TokenExchangeResult = module.TokenExchangeResult
    TokenStatusV3 = module.TokenStatusV3


class TestA2AAuthDelegationV3(unittest.TestCase):
    def setUp(self):
        self.secret = "mesh_secret_test_key_12345678901234567890"
        self.manager = A2AWorkflowAuthDelegationManagerV3(
            local_agent_id="magda_primary",
            secret_key=self.secret,
            default_ttl_seconds=300,
        )

    def test_issue_and_verify_active_token(self):
        token = self.manager.issue_token(
            subject_id="peer_worker_01",
            task_id="task_calc_sum",
            scopes=[DelegationScopeV3.EXECUTE.value, DelegationScopeV3.READ.value],
            ttl_seconds=60,
        )

        self.assertEqual(token.issuer_id, "magda_primary")
        self.assertEqual(token.subject_id, "peer_worker_01")
        self.assertTrue(token.is_valid())

        # Verify
        is_ok, verified_tok, reason = self.manager.verify_token(token, expected_subject="peer_worker_01")
        self.assertTrue(is_ok)
        self.assertIsNotNone(verified_tok)
        self.assertEqual(verified_tok.task_id, "task_calc_sum")

    def test_central_active_token_listing(self):
        self.manager.issue_token("peer_a", "task_1")
        self.manager.issue_token("peer_b", "task_2")

        active_tokens = self.manager.list_active_tokens()
        self.assertEqual(len(active_tokens), 2)

    def test_sub_token_derivation_and_depth_limits(self):
        # 1. Root token with DELEGATE scope, max_depth=2
        root_token = self.manager.issue_token(
            subject_id="lead_subagent",
            task_id="root_task",
            scopes=[DelegationScopeV3.EXECUTE.value, DelegationScopeV3.DELEGATE.value],
            max_delegation_depth=2,
        )

        # 2. Derive depth 1
        child_tok = self.manager.derive_sub_token(
            parent_token_id=root_token.token_id,
            new_subject_id="leaf_subagent",
            sub_task_id="sub_task_1",
            restricted_scopes=[DelegationScopeV3.EXECUTE.value],
        )
        self.assertEqual(child_tok.current_depth, 1)
        self.assertEqual(child_tok.parent_token_id, root_token.token_id)

        # Child lacks DELEGATE scope, so deriving further must fail
        with self.assertRaises(ValueError):
            self.manager.derive_sub_token(
                parent_token_id=child_tok.token_id,
                new_subject_id="sub_leaf",
                sub_task_id="sub_task_2",
            )

    def test_token_exchange_verification(self):
        token = self.manager.issue_token(
            subject_id="magda_primary",
            task_id="task_peer_handoff",
        )

        result = self.manager.exchange_peer_token(
            peer_agent_id="external_agent_alpha",
            token_id_or_obj=token.token_id,
            subtask_id="subtask_execute",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.peer_agent_id, "external_agent_alpha")
        self.assertEqual(result.exchanged_token.task_id, "task_peer_handoff")

    def test_token_revocation(self):
        token = self.manager.issue_token("peer_x", "task_x")
        self.manager.revoke_token(token.token_id, reason="Security policy audit")

        is_ok, tok, reason = self.manager.verify_token(token.token_id)
        self.assertFalse(is_ok)
        self.assertIn("revoked", reason)

    def test_prune_expired_tokens(self):
        # Issue an expired token
        self.manager.issue_token("peer_y", "task_y", ttl_seconds=-10)

        pruned = self.manager.prune_expired_tokens()
        self.assertGreaterEqual(pruned, 1)


if __name__ == "__main__":
    unittest.main()
