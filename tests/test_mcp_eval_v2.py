import pytest
from unittest.mock import patch, MagicMock
from magda_agent.skills.mcp_eval_v2 import MCPEvaluatorPluginV2
import httpx


@pytest.fixture
def evaluator():
    return MCPEvaluatorPluginV2()


@pytest.mark.asyncio
async def test_evaluate_skill_valid_schema_success(evaluator):
    valid_schema = {
        "name": "test_tool",
        "description": "A test tool"
    }
    sandbox_url = "http://test-sandbox.local/evaluate"

    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "success", "score": 95}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        result = await evaluator.evaluate_skill(valid_schema, sandbox_url)

        mock_post.assert_called_once_with(
            sandbox_url,
            json={"schema": valid_schema, "action": "evaluate"}
        )
        assert result == {"status": "success", "score": 95}


@pytest.mark.asyncio
async def test_evaluate_skill_valid_schema_failure(evaluator):
    valid_schema = {
        "name": "test_tool",
        "description": "A test tool"
    }
    sandbox_url = "http://test-sandbox.local/evaluate"

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server error", request=MagicMock(), response=mock_response
    )

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        result = await evaluator.evaluate_skill(valid_schema, sandbox_url)
        assert result["status"] == "error"
        assert "HTTP error occurred" in result["message"]


@pytest.mark.asyncio
async def test_evaluate_skill_invalid_schema(evaluator):
    invalid_schema = {
        "name": "test_tool"
        # missing description
    }
    sandbox_url = "http://test-sandbox.local/evaluate"

    result = await evaluator.evaluate_skill(invalid_schema, sandbox_url)
    assert result["status"] == "error"
    assert "Invalid schema format" in result["message"]


@pytest.mark.asyncio
async def test_evaluate_skill_connection_error(evaluator):
    valid_schema = {
        "name": "test_tool",
        "description": "A test tool"
    }
    sandbox_url = "http://test-sandbox.local/evaluate"

    with patch("httpx.AsyncClient.post", side_effect=httpx.RequestError("Failed to connect")):
        result = await evaluator.evaluate_skill(valid_schema, sandbox_url)
        assert result["status"] == "error"
        assert "Failed to evaluate skill at" in result["message"]
