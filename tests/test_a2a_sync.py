import pytest
from unittest.mock import AsyncMock

from magda_agent.integration.a2a_sync import A2AStateSync, NetworkClient


@pytest.fixture
def mock_network_client() -> AsyncMock:
    """Fixture providing a mocked NetworkClient."""
    client = AsyncMock(spec=NetworkClient)
    client.send_state_patch.return_value = True
    return client


@pytest.fixture
def sync_manager(mock_network_client: AsyncMock) -> A2AStateSync:
    """Fixture providing an A2AStateSync instance with a mocked client."""
    return A2AStateSync(network_client=mock_network_client)


def test_initial_state(sync_manager: A2AStateSync) -> None:
    """Test that the initial local and peer states are empty."""
    assert sync_manager.get_local_state() == {}
    assert sync_manager.get_peer_state("unknown_peer") == {}


def test_update_local_state(sync_manager: A2AStateSync) -> None:
    """Test updating the local state."""
    sync_manager.update_local_state("status", "idle")
    assert sync_manager.get_local_state() == {"status": "idle"}

    sync_manager.update_local_state("task", "processing")
    assert sync_manager.get_local_state() == {"status": "idle", "task": "processing"}

    sync_manager.update_local_state("status", "active")
    assert sync_manager.get_local_state() == {"status": "active", "task": "processing"}


def test_receive_state_sync(sync_manager: A2AStateSync) -> None:
    """Test receiving state patches from peers."""
    peer_id = "agent_x"

    sync_manager.receive_state_sync(peer_id, {"status": "busy"})
    assert sync_manager.get_peer_state(peer_id) == {"status": "busy"}

    sync_manager.receive_state_sync(peer_id, {"memory": "low"})
    assert sync_manager.get_peer_state(peer_id) == {"status": "busy", "memory": "low"}

    sync_manager.receive_state_sync(peer_id, {"status": "available"})
    assert sync_manager.get_peer_state(peer_id) == {"status": "available", "memory": "low"}

    assert sync_manager.get_peer_state("another_agent") == {}


@pytest.mark.asyncio
async def test_broadcast_state(sync_manager: A2AStateSync, mock_network_client: AsyncMock) -> None:
    """Test broadcasting a state patch to a peer."""
    peer_id = "agent_y"
    state_patch = {"load": 0.8}

    success = await sync_manager.broadcast_state(peer_id, state_patch)

    assert success is True
    mock_network_client.send_state_patch.assert_called_once_with(peer_id, state_patch)


@pytest.mark.asyncio
async def test_broadcast_state_failure(sync_manager: A2AStateSync, mock_network_client: AsyncMock) -> None:
    """Test broadcasting a state patch to a peer when the network fails."""
    peer_id = "agent_y"
    state_patch = {"load": 0.8}
    mock_network_client.send_state_patch.return_value = False

    success = await sync_manager.broadcast_state(peer_id, state_patch)

    assert success is False
    mock_network_client.send_state_patch.assert_called_once_with(peer_id, state_patch)
