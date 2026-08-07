import json
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from magda_agent.llm_client import LLMClient
from magda_agent.planning.evaluator import PlannerEvaluator


def test_evaluate_plan_approved():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_completion = AsyncMock(return_value='{"approved": true, "feedback": ""}')

    evaluator = PlannerEvaluator(llm=mock_llm)

    plan = [{"id": "step_0", "description": "do something"}]
    result = asyncio.run(evaluator.evaluate_plan(plan, "test user input"))

    assert result["approved"] is True
    assert result["feedback"] == ""
    mock_llm.chat_completion.assert_called_once()


def test_evaluate_plan_rejected():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_completion = AsyncMock(return_value='{"approved": false, "feedback": "Missing validation step."}')

    evaluator = PlannerEvaluator(llm=mock_llm)

    plan = [{"id": "step_0", "description": "do something"}]
    result = asyncio.run(evaluator.evaluate_plan(plan, "test user input"))

    assert result["approved"] is False
    assert "Missing validation step." in result["feedback"]


def test_evaluate_plan_json_decode_error():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_completion = AsyncMock(return_value='invalid json output')

    evaluator = PlannerEvaluator(llm=mock_llm)

    plan = [{"id": "step_0", "description": "do something"}]
    result = asyncio.run(evaluator.evaluate_plan(plan, "test user input"))

    # Fallback behavior
    assert result["approved"] is True
    assert "parsing failed" in result["feedback"]


def test_evaluate_plan_missing_keys():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_completion = AsyncMock(return_value='{"status": "ok"}')

    evaluator = PlannerEvaluator(llm=mock_llm)

    plan = [{"id": "step_0", "description": "do something"}]
    result = asyncio.run(evaluator.evaluate_plan(plan, "test user input"))

    # Fallback behavior
    assert result["approved"] is True
    assert "missing keys" in result["feedback"]


def test_evaluate_plan_llm_exception():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_completion = AsyncMock(side_effect=Exception("API error"))

    evaluator = PlannerEvaluator(llm=mock_llm)

    plan = [{"id": "step_0", "description": "do something"}]
    result = asyncio.run(evaluator.evaluate_plan(plan, "test user input"))

    # Fallback behavior
    assert result["approved"] is True
    assert "evaluation error" in result["feedback"]
