import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.operations.cron_v3 import HermesCronSchedulerV3
from magda_agent.memory.compaction import MemoryCompactor
from magda_agent.operations.memory_compaction_cron import schedule_memory_compaction

@pytest.fixture
def mock_scheduler() -> MagicMock:
    """Fixture that provides a mocked HermesCronSchedulerV3."""
    return MagicMock(spec=HermesCronSchedulerV3)

@pytest.fixture
def mock_compactor() -> MagicMock:
    """Fixture that provides a mocked MemoryCompactor."""
    return MagicMock(spec=MemoryCompactor)

def test_schedule_memory_compaction(mock_scheduler: MagicMock, mock_compactor: MagicMock) -> None:
    """Test that the memory compaction job is correctly scheduled."""

    # Schedule the job
    schedule_memory_compaction(mock_scheduler, mock_compactor)

    # Assert scheduler was called with correct arguments
    mock_scheduler.schedule.assert_called_once()
    args, kwargs = mock_scheduler.schedule.call_args

    assert args[0] == "0 3 * * *"
    assert kwargs.get("name") == "memory_compaction"
    assert callable(args[1])

@pytest.mark.asyncio
async def test_compaction_job_execution(mock_scheduler: MagicMock, mock_compactor: MagicMock) -> None:
    """Test that the executed job calls compactor.compact_memory."""

    # Mock to_thread to just call the function synchronously in the test
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # Schedule the job
        schedule_memory_compaction(mock_scheduler, mock_compactor)

        # Get the scheduled function
        args, _ = mock_scheduler.schedule.call_args
        compaction_job = args[1]

        # Execute the scheduled function
        await compaction_job()

        # Assert that to_thread was called with compact_memory
        mock_to_thread.assert_called_once_with(mock_compactor.compact_memory)

@pytest.mark.asyncio
async def test_compaction_job_execution_error_handling(mock_scheduler: MagicMock, mock_compactor: MagicMock) -> None:
    """Test that the job handles exceptions gracefully."""

    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.side_effect = Exception("Compaction failed")

        schedule_memory_compaction(mock_scheduler, mock_compactor)
        args, _ = mock_scheduler.schedule.call_args
        compaction_job = args[1]

        # Should not raise
        await compaction_job()

        mock_to_thread.assert_called_once_with(mock_compactor.compact_memory)
