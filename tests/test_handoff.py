import pytest
from unittest.mock import AsyncMock, MagicMock
import asyncio

from magda_agent.agents.handoff import HandoffProtocol, HandoffContext
from magda_agent.agents.sub_agent import SubAgent

def test_handoff_protocol_preserves_context():
    """
    Tests that the HandoffProtocol correctly preserves context and passes it to the target agent.
    We use a synchronous test function with asyncio.run as per project memory rules.
    """
    async def run_test():
        # Setup mock agents
        source_agent = MagicMock(spec=SubAgent)
        target_agent = MagicMock(spec=SubAgent)
        target_agent.execute = AsyncMock(return_value="Target agent result")

        # Setup handoff context
        partial_results = {"extracted_data": "some data", "step": 1}
        context = HandoffContext(
            original_task="Process the data",
            context="Initial context",
            partial_results=partial_results
        )

        # Execute handoff
        result = await HandoffProtocol.execute_handoff(
            source_agent=source_agent,
            target_agent=target_agent,
            handoff_context=context,
            handoff_reason="Needs further processing"
        )

        # Assertions
        assert result == "Target agent result"
        target_agent.execute.assert_called_once()

        call_kwargs = target_agent.execute.call_args.kwargs
        assert call_kwargs["task"] == "Process the data"
        assert "Initial context" in call_kwargs["context"]
        assert "Needs further processing" in call_kwargs["context"]
        assert "extracted_data: some data" in call_kwargs["context"]
        assert "step: 1" in call_kwargs["context"]

    asyncio.run(run_test())
