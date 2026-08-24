import pytest
from unittest.mock import AsyncMock
from magda_agent.evaluation.assert_evaluator_v3 import AssertEvaluatorV3
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_llm_client():
    mock = AsyncMock(spec=LLMClient)
    return mock

@pytest.fixture
def evaluator(mock_llm_client):
    return AssertEvaluatorV3(llm=mock_llm_client)

@pytest.mark.asyncio
async def test_evaluate_output_compliant(evaluator, mock_llm_client):
    mock_json_response = '''
    {
      "is_compliant": true,
      "violations": [],
      "score": 1.0
    }
    '''
    mock_llm_client.chat_completion.return_value = mock_json_response
    policies = ["Must be helpful"]
    output = "I can help with that."
    result = await evaluator.evaluate(output, policies)

    assert result["is_compliant"] is True
    assert result["score"] == 1.0
    assert result["violations"] == []
    mock_llm_client.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_evaluate_output_non_compliant(evaluator, mock_llm_client):
    mock_json_response = '''
    {
      "is_compliant": false,
      "violations": ["Must be polite"],
      "score": 0.0
    }
    '''
    mock_llm_client.chat_completion.return_value = mock_json_response
    policies = ["Must be polite"]
    output = "Do it yourself."
    result = await evaluator.evaluate(output, policies)

    assert result["is_compliant"] is False
    assert result["score"] == 0.0
    assert result["violations"] == ["Must be polite"]
    mock_llm_client.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_evaluate_output_json_formatting(evaluator, mock_llm_client):
    mock_json_response = '''```json
    {
      "is_compliant": true,
      "violations": [],
      "score": 1.0
    }
    ```'''
    mock_llm_client.chat_completion.return_value = mock_json_response
    policies = ["Must be helpful"]
    output = "I can help with that."
    result = await evaluator.evaluate(output, policies)

    assert result["is_compliant"] is True
    assert result["score"] == 1.0
    assert result["violations"] == []

@pytest.mark.asyncio
async def test_evaluate_output_failure(evaluator, mock_llm_client):
    mock_llm_client.chat_completion.side_effect = Exception("API Error")
    policies = ["Policy"]
    output = "Output"
    result = await evaluator.evaluate(output, policies)

    assert result["is_compliant"] is False
    assert result["score"] == 0.0
    assert result["violations"] == ["Evaluation failed"]
