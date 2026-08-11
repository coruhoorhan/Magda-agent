import pytest
from magda_agent.integration.a2a_auth_handshake_v1 import A2AAuthHandshake

def test_successful_handshake():
    """Tests that a valid token results in a successful handshake."""
    handshake = A2AAuthHandshake()
    agent_id = "agent_42"

    # Generate token for agent
    token = handshake.generate_handshake_token(agent_id)

    # Verify the handshake with the generated token
    result = handshake.verify_handshake(agent_id, token)

    assert result is True
    assert handshake.is_peer_authorized(agent_id) is True

def test_rejected_handshake_invalid_token():
    """Tests that an invalid token is rejected."""
    handshake = A2AAuthHandshake()
    agent_id = "agent_42"

    # Try to verify with a fake token
    result = handshake.verify_handshake(agent_id, "fake_token_123")

    assert result is False
    assert handshake.is_peer_authorized(agent_id) is False

def test_rejected_handshake_wrong_agent():
    """Tests that a valid token used by the wrong agent is rejected."""
    handshake = A2AAuthHandshake()
    agent1_id = "agent_42"
    agent2_id = "agent_99"

    # Generate token for agent 1
    token = handshake.generate_handshake_token(agent1_id)

    # Try to verify with agent 2 using agent 1's token
    result = handshake.verify_handshake(agent2_id, token)

    assert result is False
    assert handshake.is_peer_authorized(agent2_id) is False
    assert handshake.is_peer_authorized(agent1_id) is False

def test_token_is_single_use():
    """Tests that a token cannot be reused after a successful handshake."""
    handshake = A2AAuthHandshake()
    agent_id = "agent_42"

    token = handshake.generate_handshake_token(agent_id)

    # First handshake should succeed
    assert handshake.verify_handshake(agent_id, token) is True

    # Second handshake with the same token should fail
    assert handshake.verify_handshake(agent_id, token) is False

def test_revoke_authorization():
    """Tests that an authorized agent can have its authorization revoked."""
    handshake = A2AAuthHandshake()
    agent_id = "agent_42"

    token = handshake.generate_handshake_token(agent_id)
    handshake.verify_handshake(agent_id, token)

    assert handshake.is_peer_authorized(agent_id) is True

    # Revoke authorization
    revoke_result = handshake.revoke_authorization(agent_id)

    assert revoke_result is True
    assert handshake.is_peer_authorized(agent_id) is False

def test_revoke_unauthorized_agent():
    """Tests revoking an agent that is not authorized."""
    handshake = A2AAuthHandshake()

    assert handshake.revoke_authorization("unknown_agent") is False
