import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.agents.compression_cron_v2 import CompressionCronTaskV2

def test_compression_cron_v2_check_and_compress():
    # Setup mocks
    episodic_memory = EpisodicMemory(persist_directory=":memory:")

    # Store dummy events
    for i in range(5):
        episodic_memory.store_event(f"Event {i}")

    # Mock LLM client
    llm_client_func = AsyncMock(return_value="Summary of events")

    # Instantiate task
    cron_task = CompressionCronTaskV2(
        episodic_memory=episodic_memory,
        llm_client_func=llm_client_func,
        check_interval=0.1,
        size_threshold=3
    )

    # Run check_and_compress directly
    asyncio.run(cron_task.check_and_compress())

    # Verify LLM was called
    llm_client_func.assert_called_once()
    assert "Event 0" in llm_client_func.call_args[0][0]

    # Verify some events were decayed and new summary stored
    events = episodic_memory.get_all_events(include_decayed=False)
    # 5 original - 3 decayed + 1 summary = 3 active events
    assert len(events) == 3
    assert events[-1]["text"] == "Summary of events"

def test_compression_cron_v2_loop():
    episodic_memory = EpisodicMemory(persist_directory=":memory:")
    llm_client_func = AsyncMock(return_value="Summary of events")

    cron_task = CompressionCronTaskV2(
        episodic_memory=episodic_memory,
        llm_client_func=llm_client_func,
        check_interval=0.1,
        size_threshold=1
    )

    async def run_test():
        # Add event to trigger
        episodic_memory.store_event("Test event")

        await cron_task.start()
        # Give it a moment to run the loop
        await asyncio.sleep(0.15)
        await cron_task.stop()

    asyncio.run(run_test())

    # Verify LLM was called due to the background loop
    llm_client_func.assert_called_once()
