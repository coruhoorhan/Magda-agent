import unittest
import pytest
from unittest.mock import AsyncMock, patch
import httpx
from magda_agent.integration.a2a_auth_v4 import A2AEnterpriseAuthTraversalV4

@pytest.fixture
def auth_traversal():
    return A2AEnterpriseAuthTraversalV4()

def test_initiate_traversal(auth_traversal):
    token = auth_traversal.initiate_traversal("node_a", max_hops=3)
    assert token.startswith("a2a_trav_")

    assert token in auth_traversal._active_traversals
    state = auth_traversal._active_traversals[token]
    assert state["origin"] == "node_a"
    assert state["current_node"] == "node_a"
    assert state["path"] == ["node_a"]
    assert state["max_hops"] == 3

def test_verify_traversal_auth_valid(auth_traversal):
    token = auth_traversal.initiate_traversal("node_a", max_hops=3)
    assert auth_traversal.verify_traversal_auth(token, "node_a") is True

def test_verify_traversal_auth_invalid_token(auth_traversal):
    assert auth_traversal.verify_traversal_auth("invalid_token", "node_a") is False

def test_verify_traversal_auth_wrong_node(auth_traversal):
    token = auth_traversal.initiate_traversal("node_a", max_hops=3)
    assert auth_traversal.verify_traversal_auth(token, "node_b") is False

def test_forward_traversal_state(auth_traversal):
    token = auth_traversal.initiate_traversal("node_a", max_hops=3)

    success = auth_traversal.forward_traversal_state(token, "node_b")
    assert success is True
    assert auth_traversal.verify_traversal_auth(token, "node_b") is True

    state = auth_traversal._active_traversals[token]
    assert state["path"] == ["node_a", "node_b"]

def test_forward_traversal_state_max_hops(auth_traversal):
    token = auth_traversal.initiate_traversal("node_a", max_hops=1)

    # Hop 1
    assert auth_traversal.forward_traversal_state(token, "node_b") is True

    # Hop 2 - should fail
    assert auth_traversal.forward_traversal_state(token, "node_c") is False

@pytest.mark.asyncio
@patch("magda_agent.integration.a2a_auth_v4.httpx.AsyncClient")
async def test_forward_traversal_network_success(mock_async_client, auth_traversal):
    token = auth_traversal.initiate_traversal("node_a", max_hops=3)

    mock_post = AsyncMock()
    mock_post.return_value.raise_for_status = unittest.mock.MagicMock()

    mock_client_instance = mock_async_client.return_value.__aenter__.return_value
    mock_client_instance.post = mock_post

    result = await auth_traversal.forward_traversal_network(token, "http://node-b.local/rpc")

    assert result is True
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://node-b.local/rpc"
    assert kwargs["json"]["method"] == "receive_traversal_auth"
    assert kwargs["json"]["params"]["traversal_token"] == token

@pytest.mark.asyncio
@patch("magda_agent.integration.a2a_auth_v4.httpx.AsyncClient")
async def test_forward_traversal_network_failure(mock_async_client, auth_traversal):
    token = auth_traversal.initiate_traversal("node_a", max_hops=3)

    mock_post = AsyncMock(side_effect=httpx.HTTPError("Network timeout"))

    mock_client_instance = mock_async_client.return_value.__aenter__.return_value
    mock_client_instance.post = mock_post

    result = await auth_traversal.forward_traversal_network(token, "http://node-b.local/rpc")

    assert result is False
