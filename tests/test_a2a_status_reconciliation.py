import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import httpx

from magda_agent.integration.a2a_status_reconciliation import A2AStatusReconciliationWorker
from magda_agent.integration.a2a_discovery_v3_unique import A2ADiscoveryServiceV3Unique
from magda_agent.integration.a2a_cards import AgentCardV3

@pytest.fixture
def mock_discovery_service():
    service = MagicMock(spec=A2ADiscoveryServiceV3Unique)
    # Mock an agent card
    card = AgentCardV3(
        agent_id="peer-123",
        name="TestPeer",
        description="A test peer",
        capabilities=["test"],
        endpoints={"rpc": "http://peer-123.local"}
    )
    service.get_agent_card.return_value = card
    return service

@pytest.fixture
def worker(mock_discovery_service):
    return A2AStatusReconciliationWorker(
        discovery_service=mock_discovery_service,
        poll_interval=0.1, # fast for tests
        timeout=1.0
    )

def test_add_task(worker):
    worker.add_task("task-1", "peer-123", {"data": "test"})
    assert "task-1" in worker.local_tasks
    assert worker.local_tasks["task-1"]["status"] == "pending"
    assert worker.local_tasks["task-1"]["peer_id"] == "peer-123"
    assert worker.local_tasks["task-1"]["retry_count"] == 0

@pytest.mark.asyncio
async def test_reconcile_tasks_successful_update(worker):
    worker.add_task("task-1", "peer-123", {"data": "test"})

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "completed"}
    mock_response.raise_for_status.return_value = None

    with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        await worker.reconcile_tasks()

        mock_get.assert_called_once_with("http://peer-123.local/status/task-1", timeout=1.0)
        assert worker.local_tasks["task-1"]["status"] == "completed"

@pytest.mark.asyncio
async def test_reconcile_tasks_missing_peer(worker):
    worker.add_task("task-2", "unknown-peer", {"data": "test"})

    # Make get_agent_card return None for unknown peer
    worker.discovery_service.get_agent_card.return_value = None

    await worker.reconcile_tasks()

    assert worker.local_tasks["task-2"]["status"] == "orphaned"

@pytest.mark.asyncio
async def test_reconcile_tasks_404_not_found(worker):
    worker.add_task("task-3", "peer-123", {"data": "test"})

    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        await worker.reconcile_tasks()

        assert worker.local_tasks["task-3"]["status"] == "orphaned"

@pytest.mark.asyncio
async def test_reconcile_tasks_network_failure_retries(worker):
    worker.add_task("task-4", "peer-123", {"data": "test"})

    with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.RequestError("Network error", request=MagicMock())

        # 1st attempt
        await worker.reconcile_tasks()
        assert worker.local_tasks["task-4"]["status"] == "pending"
        assert worker.local_tasks["task-4"]["retry_count"] == 1

        # 2nd attempt
        await worker.reconcile_tasks()
        assert worker.local_tasks["task-4"]["status"] == "pending"
        assert worker.local_tasks["task-4"]["retry_count"] == 2

        # 3rd attempt (exceeds max retries)
        await worker.reconcile_tasks()
        assert worker.local_tasks["task-4"]["status"] == "orphaned"
        assert worker.local_tasks["task-4"]["retry_count"] == 3

@pytest.mark.asyncio
async def test_start_stop_worker(worker):
    # Just to ensure start and stop don't block or raise errors
    await worker.start()
    assert worker._running is True
    assert worker._task is not None

    await worker.stop()
    assert worker._running is False
