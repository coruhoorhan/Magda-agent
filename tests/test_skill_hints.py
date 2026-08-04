import pytest
from unittest.mock import MagicMock
from magda_agent.learning.skill_hints import SkillHintsExtractor

def test_extract_hints_no_logs():
    """Test extracting hints with an empty log list."""
    extractor = SkillHintsExtractor()
    hints = extractor.extract_hints([])
    assert hints == {}

def test_extract_hints_only_failed_executions():
    """Test that failed executions are ignored."""
    extractor = SkillHintsExtractor()
    logs = [
        {"tool_name": "search", "arguments": {"query": "python"}, "success": False},
        {"tool_name": "read_file", "arguments": {"path": "main.py"}, "success": False},
    ]
    hints = extractor.extract_hints(logs)
    assert hints == {}

def test_extract_hints_heuristic_fallback():
    """Test the fallback heuristic when no LLM client is provided."""
    extractor = SkillHintsExtractor()
    logs = [
        {"tool_name": "search", "arguments": {"query": "python", "limit": 10}, "success": True},
        {"tool_name": "search", "arguments": {"query": "pytest", "limit": 5}, "success": True},
        {"tool_name": "search", "arguments": {"query": "mypy"}, "success": True},
        {"tool_name": "read_file", "arguments": {"path": "main.py"}, "success": True},
    ]
    hints = extractor.extract_hints(logs)

    assert "search" in hints
    assert "read_file" in hints

    # "query" is common to all successful 'search' executions
    assert "query" in hints["search"]
    assert "successfully used 3 times" in hints["search"]

    assert "path" in hints["read_file"]
    assert "successfully used 1 times" in hints["read_file"]

def test_extract_hints_with_mocked_llm():
    """Test extracting hints using a mocked LLM client."""
    mock_llm_client = MagicMock()
    mock_llm_client.generate.return_value = "Mocked LLM hint for tool."

    extractor = SkillHintsExtractor(llm_client=mock_llm_client)
    logs = [
        {"tool_name": "calculator", "arguments": {"expression": "2+2"}, "success": True},
        {"tool_name": "calculator", "arguments": {"expression": "3*4"}, "success": False}, # should be ignored
    ]

    hints = extractor.extract_hints(logs)

    assert "calculator" in hints
    assert hints["calculator"] == "Mocked LLM hint for tool."

    # Verify the LLM was called with the correct prompt format
    mock_llm_client.generate.assert_called_once()
    call_args = mock_llm_client.generate.call_args[0][0]
    assert "calculator" in call_args
    assert "{'expression': '2+2'}" in call_args
    assert "3*4" not in call_args
