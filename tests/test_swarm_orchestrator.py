import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.architecture.swarm_orchestrator import SwarmAgent
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.chat_completion = AsyncMock()
    return llm

@pytest.mark.asyncio
async def test_swarm_agent_direct_execution(mock_llm):
    # Setup mock to return a decision not to delegate
    mock_llm.chat_completion.side_effect = [
        '{"delegate": false, "subtasks": []}',
        'Final result of simple task'
    ]

    agent = SwarmAgent(llm=mock_llm, max_depth=2)
    result = await agent.execute("Do a simple task")

    assert result == 'Final result of simple task'
    assert mock_llm.chat_completion.call_count == 2

    # Analyze task call
    assert "Analyze the following task:" in mock_llm.chat_completion.call_args_list[0][0][0][0]["content"]
    # Execute direct call
    assert "Execute the following task" in mock_llm.chat_completion.call_args_list[1][0][0][0]["content"]

@pytest.mark.asyncio
async def test_swarm_agent_hierarchical_delegation(mock_llm):
    # Setup mock responses for a complex task with one level of delegation
    # Call 1: Analyze complex task -> delegates to 2 subtasks
    # Call 2 & 3: Analyze subtasks -> both decide not to delegate
    # Call 4 & 5: Execute subtasks directly
    # Call 6: Synthesize results

    mock_llm.chat_completion.side_effect = [
        '{"delegate": true, "subtasks": ["subtask A", "subtask B"]}', # Root analysis
        '{"delegate": false, "subtasks": []}', # Child A analysis
        '{"delegate": false, "subtasks": []}', # Child B analysis
        'Result A', # Child A execution
        'Result B', # Child B execution
        'Synthesized Final Result' # Root synthesis
    ]

    agent = SwarmAgent(llm=mock_llm, max_depth=2)
    result = await agent.execute("Do a complex task")

    assert result == 'Synthesized Final Result'
    assert mock_llm.chat_completion.call_count == 6

@pytest.mark.asyncio
async def test_swarm_agent_max_depth_enforced(mock_llm):
    # Set max depth to 0, which should force direct execution immediately
    # without even analyzing the task for delegation.

    mock_llm.chat_completion.side_effect = [
        'Final result from max depth'
    ]

    agent = SwarmAgent(llm=mock_llm, max_depth=0)
    result = await agent.execute("Any task")

    assert result == 'Final result from max depth'
    assert mock_llm.chat_completion.call_count == 1

    # Should only be the direct execution prompt, not the analysis one
    assert "Execute the following task" in mock_llm.chat_completion.call_args_list[0][0][0][0]["content"]

@pytest.mark.asyncio
async def test_swarm_agent_fallback_on_json_error(mock_llm):
    # Setup mock to return invalid JSON during analysis
    mock_llm.chat_completion.side_effect = [
        'this is not valid json',
        'Fallback direct execution result'
    ]

    agent = SwarmAgent(llm=mock_llm, max_depth=2)
    result = await agent.execute("Task that causes json error")

    assert result == 'Fallback direct execution result'
    assert mock_llm.chat_completion.call_count == 2
