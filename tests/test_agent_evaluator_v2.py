import pytest
import json
from unittest.mock import AsyncMock, patch
from magda_agent.evaluation.agent_evaluator_v2 import AgentEvaluatorV2
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_llm_client():
    return AsyncMock(spec=LLMClient)

@pytest.fixture
def agent_evaluator(mock_llm_client):
    return AgentEvaluatorV2(llm=mock_llm_client)

@pytest.mark.asyncio
async def test_evaluate_generator_output_success(agent_evaluator):
    agent_evaluator.evaluator_agent.execute = AsyncMock()
    mock_json_response = '''
    ```json
    {
      "scores": {"accuracy": 8, "style": 9},
      "approved": true,
      "feedback": "Good output"
    }
    ```
    '''
    agent_evaluator.evaluator_agent.execute.return_value = mock_json_response

    task_desc = "Write a python script."
    gen_output = "print('hello')"
    rubric = {"accuracy": "Code runs without errors.", "style": "Code follows pep8."}

    result = await agent_evaluator.evaluate_generator_output(task_desc, gen_output, rubric)

    agent_evaluator.evaluator_agent.execute.assert_called_once()
    assert result["approved"] is True
    assert result["scores"]["accuracy"] == 8
    assert result["feedback"] == "Good output"

@pytest.mark.asyncio
async def test_evaluate_generator_output_retry_success(agent_evaluator):
    agent_evaluator.evaluator_agent.execute = AsyncMock()
    invalid_json = "this is not valid json"
    valid_json = '''{
      "scores": {"completeness": 10},
      "approved": true,
      "feedback": "Perfect"
    }'''
    agent_evaluator.evaluator_agent.execute.side_effect = [invalid_json, valid_json]

    result = await agent_evaluator.evaluate_generator_output("task", "output", {"completeness": "desc"})

    assert result["approved"] is True
    assert result["scores"]["completeness"] == 10
    assert agent_evaluator.evaluator_agent.execute.call_count == 2

@pytest.mark.asyncio
async def test_evaluate_generator_output_retry_failure(agent_evaluator):
    agent_evaluator.evaluator_agent.execute = AsyncMock()
    invalid_json = "this is not valid json"
    agent_evaluator.evaluator_agent.execute.side_effect = [invalid_json, invalid_json, invalid_json]

    result = await agent_evaluator.evaluate_generator_output("task", "output", {"completeness": "desc"})

    assert result["approved"] is False
    assert "Failed to parse" in result["feedback"]
    assert agent_evaluator.evaluator_agent.execute.call_count == 3
