"""
Tests for A2A Workflow Auth Delegation Tokens.
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.integration.a2a_auth_delegation import (
        A2ADelegationToken,
        A2AAuthTokenManager,
        DelegationTokenScope,
        A2ATaskDelegationPayload,
        A2AWorkflowAuthDelegator,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent / "magda_agent" / "integration" / "a2a_auth_delegation.py"
    spec = importlib.util.spec_from_file_location("a2a_auth_delegation", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    A2ADelegationToken = module.A2ADelegationToken
    A2AAuthTokenManager = module.A2AAuthTokenManager
    DelegationTokenScope = module.DelegationTokenScope
    A2ATaskDelegationPayload = module.A2ATaskDelegationPayload
    A2AWorkflowAuthDelegator = module.A2AWorkflowAuthDelegator


class TestA2AAuthDelegation(unittest.IsolatedAsyncioTestCase):
    """
    Comprehensive test suite verifying A2A authentication delegation tokens,
    HMAC validation, multi-hop derivation, and task workflow dispatch.
    """

    def setUp(self):
        self.secret = "test-secret-key-1234567890abcdef"
        self.token_manager = A2AAuthTokenManager(secret_key=self.secret, default_ttl_seconds=60)

    # -------------------------------------------------------------------------
    # 1. Token Issuance & Verification
    # -------------------------------------------------------------------------
    def test_token_issuance_and_signature_validity(self):
        """Token should be generated with valid HMAC signature and matching fields."""
        token = self.token_manager.issue_token(
            issuer_id="agent-alpha",
            subject_id="agent-beta",
            task_id="task-100",
            scopes=[DelegationTokenScope.EXECUTE.value],
            ttl_seconds=120,
        )

        self.assertTrue(token.token_id.startswith("a2a_tok_"))
        self.assertEqual(token.issuer_id, "agent-alpha")
        self.assertEqual(token.subject_id, "agent-beta")
        self.assertTrue(len(token.signature) > 0)

        is_valid, reason, validated = self.token_manager.verify_token(token, expected_peer_id="agent-beta")
        self.assertTrue(is_valid, f"Verification failed: {reason}")
        self.assertEqual(validated.token_id, token.token_id)

    def test_tampered_token_signature_rejected(self):
        """Modifying token fields must invalidate the cryptographic signature."""
        token = self.token_manager.issue_token(
            issuer_id="agent-alpha",
            subject_id="agent-beta",
            task_id="task-100",
        )

        # Tamper payload
        token_dict = token.to_dict()
        token_dict["scopes"] = [DelegationTokenScope.ADMIN.value]  # Privilege escalation attempt

        is_valid, reason, _ = self.token_manager.verify_token(token_dict)
        self.assertFalse(is_valid)
        self.assertIn("Invalid token cryptographic signature", reason)

    # -------------------------------------------------------------------------
    # 2. Expiration, Revocation & Subject Validation
    # -------------------------------------------------------------------------
    def test_expired_token_rejected(self):
        """Expired tokens must fail verification."""
        token = self.token_manager.issue_token(
            issuer_id="agent-alpha",
            subject_id="agent-beta",
            task_id="task-100",
            ttl_seconds=-10,  # Already expired
        )

        is_valid, reason, _ = self.token_manager.verify_token(token)
        self.assertFalse(is_valid)
        self.assertIn("has expired", reason)

    def test_revoked_token_rejected(self):
        """Revoked tokens must be blocked."""
        token = self.token_manager.issue_token(
            issuer_id="agent-alpha",
            subject_id="agent-beta",
            task_id="task-100",
        )

        self.token_manager.revoke_token(token.token_id)

        is_valid, reason, _ = self.token_manager.verify_token(token.token_id)
        self.assertFalse(is_valid)
        self.assertIn("is revoked", reason)

    def test_subject_mismatch_rejected(self):
        """Token intended for agent-beta must be rejected if presented to agent-gamma."""
        token = self.token_manager.issue_token(
            issuer_id="agent-alpha",
            subject_id="agent-beta",
            task_id="task-100",
        )

        is_valid, reason, _ = self.token_manager.verify_token(token, expected_peer_id="agent-gamma")
        self.assertFalse(is_valid)
        self.assertIn("does not match recipient", reason)

    def test_wildcard_subject_accepted_by_any_peer(self):
        """Wildcard '*' subject token should be accepted by any expected peer."""
        token = self.token_manager.issue_token(
            issuer_id="agent-alpha",
            subject_id="*",
            task_id="broadcast-task",
        )

        is_valid, reason, _ = self.token_manager.verify_token(token, expected_peer_id="agent-any-node")
        self.assertTrue(is_valid)

    # -------------------------------------------------------------------------
    # 3. Scope Enforcement
    # -------------------------------------------------------------------------
    def test_missing_required_scope_rejected(self):
        """Token lacking requested scope must fail validation."""
        token = self.token_manager.issue_token(
            issuer_id="agent-alpha",
            subject_id="agent-beta",
            task_id="task-100",
            scopes=[DelegationTokenScope.READ.value],
        )

        is_valid, reason, _ = self.token_manager.verify_token(
            token,
            required_scope=DelegationTokenScope.EXECUTE.value,
        )
        self.assertFalse(is_valid)
        self.assertIn("lacks required scope", reason)

    # -------------------------------------------------------------------------
    # 4. Multi-Hop Token Derivation & Depth Limits
    # -------------------------------------------------------------------------
    def test_multi_hop_token_derivation_success(self):
        """Downstream tokens can be derived with incremented depth."""
        parent_token = self.token_manager.issue_token(
            issuer_id="agent-root",
            subject_id="agent-hop1",
            task_id="parent-task",
            scopes=[DelegationTokenScope.EXECUTE.value, DelegationTokenScope.DELEGATE.value],
            max_delegation_depth=2,
        )

        child_token = self.token_manager.derive_downstream_token(
            parent_token=parent_token,
            downstream_peer_id="agent-hop2",
            new_task_id="child-task-1",
        )

        self.assertEqual(child_token.current_depth, 1)
        self.assertEqual(child_token.issuer_id, "agent-hop1")
        self.assertEqual(child_token.subject_id, "agent-hop2")
        self.assertEqual(child_token.task_id, "child-task-1")

        is_valid, _, _ = self.token_manager.verify_token(child_token, expected_peer_id="agent-hop2")
        self.assertTrue(is_valid)

    def test_max_delegation_depth_exceeded_rejected(self):
        """Deriving past max_delegation_depth should fail."""
        token = self.token_manager.issue_token(
            issuer_id="agent-root",
            subject_id="agent-hop1",
            task_id="task-1",
            max_delegation_depth=1,
            scopes=[DelegationTokenScope.EXECUTE.value, DelegationTokenScope.DELEGATE.value],
        )

        child = self.token_manager.derive_downstream_token(token, "agent-hop2")
        self.assertEqual(child.current_depth, 1)

        # Attempt second hop when max is 1
        with self.assertRaises(ValueError) as ctx:
            self.token_manager.derive_downstream_token(child, "agent-hop3")
        self.assertIn("Max delegation depth", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 5. End-to-End Workflow Delegation
    # -------------------------------------------------------------------------
    async def test_e2e_workflow_delegation_success(self):
        """Test full delegation between peer delegator and peer receiver."""
        shared_manager = A2AAuthTokenManager(secret_key=self.secret)
        agent_alpha = A2AWorkflowAuthDelegator(local_agent_id="agent-alpha", token_manager=shared_manager)
        agent_beta = A2AWorkflowAuthDelegator(local_agent_id="agent-beta", token_manager=shared_manager)

        async def mock_network_dispatch(payload: dict) -> dict:
            return await agent_beta.receive_delegated_task(payload)

        agent_alpha.peer_dispatch_fn = mock_network_dispatch

        # Register custom handler on agent_beta
        def analyze_handler(task_data: dict, token: A2ADelegationToken) -> dict:
            return {"status": "analyzed", "input": task_data.get("metric"), "by": token.subject_id}

        agent_beta.register_task_handler("analyze_metrics", analyze_handler)

        result = await agent_alpha.delegate_task(
            target_peer_id="agent-beta",
            task_name="analyze_metrics",
            task_data={"metric": "cpu_load_88"},
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["status_code"], 200)
        self.assertEqual(result["result"]["status"], "analyzed")
        self.assertEqual(result["result"]["input"], "cpu_load_88")
        self.assertEqual(agent_alpha.delegation_metrics["dispatched"], 1)
        self.assertEqual(agent_beta.delegation_metrics["authorized"], 1)

    async def test_e2e_workflow_delegation_unauthorized_token_rejected(self):
        """Receiver should reject requests carrying invalid token from different secret."""
        manager_a = A2AAuthTokenManager(secret_key="secret-A")
        manager_b = A2AAuthTokenManager(secret_key="secret-B")

        agent_alpha = A2AWorkflowAuthDelegator(local_agent_id="agent-alpha", token_manager=manager_a)
        agent_beta = A2AWorkflowAuthDelegator(local_agent_id="agent-beta", token_manager=manager_b)

        async def mock_network_dispatch(payload: dict) -> dict:
            return await agent_beta.receive_delegated_task(payload)

        agent_alpha.peer_dispatch_fn = mock_network_dispatch

        result = await agent_alpha.delegate_task(
            target_peer_id="agent-beta",
            task_name="execute_job",
            task_data={"job_id": 99},
        )

        self.assertEqual(result["status"], "unauthorized")
        self.assertEqual(result["status_code"], 401)
        self.assertIn("Token Verification Failed", result["error"])
        self.assertEqual(agent_beta.delegation_metrics["rejected"], 1)


if __name__ == "__main__":
    unittest.main()
