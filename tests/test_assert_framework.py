import pytest
import json
from unittest.mock import AsyncMock

from magda_agent.metacognition.assert_framework import AssertFramework
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_llm_client():
    mock = AsyncMock(spec=LLMClient)
    return mock

@pytest.fixture
def assert_framework(mock_llm_client):
    return AssertFramework(llm=mock_llm_client)

@pytest.mark.asyncio
async def test_evaluate_plan_compliant(assert_framework, mock_llm_client):
    mock_json_response = '''
    {
      "is_compliant": true,
      "violations": []
    }
    '''
    mock_llm_client.chat_completion.return_value = mock_json_response

    plan = ["Run tests"]
    policies = ["Do no harm"]

    result = await assert_framework.evaluate_plan(plan, policies)

    assert result["is_compliant"] is True
    assert result["violations"] == []
    mock_llm_client.chat_completion.assert_called_once()

    # Check that prompt formatting is correct
    call_args = mock_llm_client.chat_completion.call_args[0][0]
    prompt = call_args[0]["content"]
    assert "- Do no harm" in prompt
    assert "1. Run tests" in prompt

@pytest.mark.asyncio
async def test_evaluate_plan_non_compliant(assert_framework, mock_llm_client):
    mock_json_response = '''
    {
      "is_compliant": false,
      "violations": ["Must not delete user data"]
    }
    '''
    mock_llm_client.chat_completion.return_value = mock_json_response

    plan = ["Delete user database"]
    policies = ["Must not delete user data"]

    result = await assert_framework.evaluate_plan(plan, policies)

    assert result["is_compliant"] is False
    assert result["violations"] == ["Must not delete user data"]

@pytest.mark.asyncio
async def test_evaluate_plan_json_decode_retry_success(assert_framework, mock_llm_client):
    invalid_json = "This is not json"
    valid_json = '''
    {
      "is_compliant": true,
      "violations": []
    }
    '''

    mock_llm_client.chat_completion.side_effect = [invalid_json, valid_json]

    plan = ["Safe plan"]
    policies = ["Safe policy"]

    result = await assert_framework.evaluate_plan(plan, policies)

    assert result["is_compliant"] is True
    assert mock_llm_client.chat_completion.call_count == 2

@pytest.mark.asyncio
async def test_evaluate_plan_json_decode_failure(assert_framework, mock_llm_client):
    invalid_json = "This is not json"
    mock_llm_client.chat_completion.side_effect = [invalid_json, invalid_json, invalid_json]

    plan = ["Safe plan"]
    policies = ["Safe policy"]

    result = await assert_framework.evaluate_plan(plan, policies)

    assert result["is_compliant"] is False
    assert "Evaluation failed due to JSON decoding error." in result["violations"][0]
    assert mock_llm_client.chat_completion.call_count == 3

@pytest.mark.asyncio
async def test_evaluate_plan_exception(assert_framework, mock_llm_client):
    mock_llm_client.chat_completion.side_effect = Exception("Network error")

    plan = ["Safe plan"]
    policies = ["Safe policy"]

    result = await assert_framework.evaluate_plan(plan, policies)

    assert result["is_compliant"] is False
    assert "Evaluation failed: Network error" in result["violations"][0]

from magda_agent.safety.assert_framework import AssertActionEvaluator

@pytest.fixture
def assert_action_evaluator(mock_llm_client):
    return AssertActionEvaluator(llm=mock_llm_client)

@pytest.mark.asyncio
async def test_evaluate_action_compliant(assert_action_evaluator, mock_llm_client):
    mock_json_response = '''
    {
      "is_compliant": true,
      "violations": []
    }
    '''
    mock_llm_client.chat_completion.return_value = mock_json_response

    action = {"tool_name": "read_file", "args": {"file_path": "docs/readme.md"}}
    policies = ["Do no harm", "Do not modify files"]

    result = await assert_action_evaluator.evaluate_action(action, policies)

    assert result["is_compliant"] is True
    assert result["violations"] == []
    # Chat completion is called twice in this test file because mock_llm_client might not be reset between tests?
    # No, it's a fixture, so it's a new instance or we can just assert it was called
    mock_llm_client.chat_completion.assert_called_once()

    # Check that prompt formatting is correct
    call_args = mock_llm_client.chat_completion.call_args[0][0]
    prompt = call_args[0]["content"]
    assert "- Do no harm" in prompt
    assert "read_file" in prompt

@pytest.mark.asyncio
async def test_evaluate_action_non_compliant(assert_action_evaluator, mock_llm_client):
    mock_json_response = '''
    {
      "is_compliant": false,
      "violations": ["Do not modify files"]
    }
    '''
    mock_llm_client.chat_completion.return_value = mock_json_response

    action = {"tool_name": "write_file", "args": {"file_path": "system.cfg", "content": "hack"}}
    policies = ["Do not modify files"]

    result = await assert_action_evaluator.evaluate_action(action, policies)

    assert result["is_compliant"] is False
    assert result["violations"] == ["Do not modify files"]

@pytest.mark.asyncio
async def test_evaluate_action_json_decode_retry_success(assert_action_evaluator, mock_llm_client):
    invalid_json = "This is not json"
    valid_json = '''
    {
      "is_compliant": true,
      "violations": []
    }
    '''

    mock_llm_client.chat_completion.side_effect = [invalid_json, valid_json]

    action = {"tool_name": "safe_tool"}
    policies = ["Safe policy"]

    result = await assert_action_evaluator.evaluate_action(action, policies)

    assert result["is_compliant"] is True
    assert mock_llm_client.chat_completion.call_count == 2

@pytest.mark.asyncio
async def test_evaluate_action_json_decode_failure(assert_action_evaluator, mock_llm_client):
    invalid_json = "This is not json"
    mock_llm_client.chat_completion.side_effect = [invalid_json, invalid_json, invalid_json]

    action = {"tool_name": "safe_tool"}
    policies = ["Safe policy"]

    result = await assert_action_evaluator.evaluate_action(action, policies)

    assert result["is_compliant"] is False
    assert "Evaluation failed due to JSON decoding error." in result["violations"][0]
    assert mock_llm_client.chat_completion.call_count == 3

@pytest.mark.asyncio
async def test_evaluate_action_exception(assert_action_evaluator, mock_llm_client):
    mock_llm_client.chat_completion.side_effect = Exception("Network error")

    action = {"tool_name": "safe_tool"}
    policies = ["Safe policy"]

    result = await assert_action_evaluator.evaluate_action(action, policies)

    assert result["is_compliant"] is False
    assert "Evaluation failed: Network error" in result["violations"][0]
