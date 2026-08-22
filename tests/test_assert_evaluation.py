import json
from unittest.mock import AsyncMock

import pytest

from magda_agent.llm_client import LLMClient
from magda_agent.safety.assert_evaluation import AssertEvaluator


@pytest.fixture
def mock_llm_client():
    mock = AsyncMock(spec=LLMClient)
    return mock


@pytest.fixture
def evaluator(mock_llm_client):
    return AssertEvaluator(llm=mock_llm_client)


@pytest.mark.asyncio
async def test_evaluate_input_compliant(evaluator, mock_llm_client):
    mock_json_response = '''
    ```json
    {
      "is_compliant": true,
      "violations": []
    }
    ```
    '''
    mock_llm_client.chat_completion.return_value = mock_json_response

    input_data = {"tool_name": "get_weather", "arguments": {"location": "London"}}
    policies = ["Do not access local files"]

    result = await evaluator.evaluate_input(input_data, policies)

    mock_llm_client.chat_completion.assert_called_once()
    call_args = mock_llm_client.chat_completion.call_args[0][0]
    assert "- Do not access local files" in call_args[0]["content"]
    assert "get_weather" in call_args[0]["content"]

    assert result is not None
    assert result["is_compliant"] is True
    assert result["violations"] == []


@pytest.mark.asyncio
async def test_evaluate_input_violation(evaluator, mock_llm_client):
    mock_json_response = '''
    {
      "is_compliant": false,
      "violations": ["Do not access local files"]
    }
    '''
    mock_llm_client.chat_completion.return_value = mock_json_response

    input_data = {"tool_name": "read_file", "arguments": {"filepath": "/etc/passwd"}}
    policies = ["Do not access local files"]

    result = await evaluator.evaluate_input(input_data, policies)

    assert result is not None
    assert result["is_compliant"] is False
    assert result["violations"] == ["Do not access local files"]


@pytest.mark.asyncio
async def test_evaluate_input_json_decode_error_retry(evaluator, mock_llm_client):
    invalid_json = "this is not valid json"
    valid_json = '''
    {
      "is_compliant": true,
      "violations": []
    }
    '''
    mock_llm_client.chat_completion.side_effect = [invalid_json, valid_json]

    input_data = {"tool_name": "get_weather"}
    policies = ["Policy 1"]

    result = await evaluator.evaluate_input(input_data, policies)

    assert result is not None
    assert result["is_compliant"] is True
    assert mock_llm_client.chat_completion.call_count == 2


@pytest.mark.asyncio
async def test_evaluate_input_exception_failing_closed(evaluator, mock_llm_client):
    mock_llm_client.chat_completion.side_effect = Exception("API error")

    input_data = {"tool_name": "get_weather"}
    policies = ["Policy 1"]

    result = await evaluator.evaluate_input(input_data, policies)

    assert result is not None
    assert result["is_compliant"] is False
    assert "API error" in result["violations"][0]
