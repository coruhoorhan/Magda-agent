import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, call

from magda_agent.integration.a2a_cards import A2ADiscoveryMeshV4
from magda_agent.integration.a2a_gossip_v5 import A2ACapabilityGossipV5

@pytest.fixture
def mock_mesh():
    mesh = MagicMock(spec=A2ADiscoveryMeshV4)
    mesh.broadcast_gossip = AsyncMock()
    return mesh

@pytest.mark.asyncio
async def test_a2a_capability_gossip_v5_start_stop(mock_mesh):
    """
    Test that the gossip protocol starts and stops cleanly.
    Using a very short interval and allowing the event loop to run.
    """
    peer_urls = ["http://peer1.local", "http://peer2.local"]

    # Standard asyncio sleep rather than mocking for test reliability in event loop
    gossip = A2ACapabilityGossipV5(mesh=mock_mesh, peer_urls=peer_urls, interval_seconds=0.01)

    gossip.start()

    # Allow the loop to run a few times
    await asyncio.sleep(0.05)

    await gossip.stop()

    # Should have called broadcast_gossip at least once
    assert mock_mesh.broadcast_gossip.call_count >= 1

    # Check args for the calls
    mock_mesh.broadcast_gossip.assert_has_calls([call(peer_urls)])


@pytest.mark.asyncio
async def test_a2a_capability_gossip_v5_exception_handling(mock_mesh):
    """
    Test that an exception during broadcast doesn't crash the background thread.
    """
    peer_urls = ["http://peer1.local", "http://peer2.local"]

    # Setup mock to fail on the first call, succeed on subsequent ones
    mock_mesh.broadcast_gossip.side_effect = [Exception("Network error"), None, None]

    gossip = A2ACapabilityGossipV5(mesh=mock_mesh, peer_urls=peer_urls, interval_seconds=0.01)

    gossip.start()

    # Wait for the loop to run a few times
    await asyncio.sleep(0.05)

    await gossip.stop()

    # Should have called broadcast_gossip multiple times despite the first failure
    assert mock_mesh.broadcast_gossip.call_count >= 2


@pytest.mark.asyncio
async def test_a2a_capability_gossip_v5_multiple_start(mock_mesh):
    """
    Test that starting the gossip multiple times doesn't spawn multiple tasks.
    """
    peer_urls = ["http://peer1.local"]

    gossip = A2ACapabilityGossipV5(mesh=mock_mesh, peer_urls=peer_urls, interval_seconds=0.01)

    gossip.start()
    task1 = gossip._task

    # Second start should log a warning and return without changing the task
    gossip.start()
    task2 = gossip._task

    assert task1 is task2

    await gossip.stop()
