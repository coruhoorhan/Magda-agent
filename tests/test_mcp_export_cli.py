import json
import pytest
from unittest.mock import patch, MagicMock

from magda_agent.skills.mcp_export_cli import main

def test_mcp_export_cli_help(capsys):
    """Test that the CLI prints help and exits with 0 on --help."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "CLI tool to export Magda's active skills as an MCP-compatible JSON schema" in captured.out
    assert "export" in captured.out

def test_mcp_export_cli_export(capsys):
    """Test that the CLI exports skills to standard output as valid JSON."""
    result = main(["export"])

    assert result == 0
    captured = capsys.readouterr()

    # Verify the output is valid JSON
    output_json = json.loads(captured.out)

    # Verify it is a list of tools
    assert isinstance(output_json, list)

    # Verify the structure matches MCP JSON schema requirements
    if len(output_json) > 0:
        first_tool = output_json[0]
        assert "name" in first_tool
        assert "description" in first_tool
        assert "inputSchema" in first_tool

        schema = first_tool["inputSchema"]
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema

@patch("magda_agent.skills.mcp_export_cli.initialize_skills")
def test_mcp_export_cli_export_error(mock_initialize, capsys):
    """Test that the CLI handles errors during export gracefully."""
    mock_initialize.side_effect = Exception("Test exception")

    result = main(["export"])

    assert result == 1
    captured = capsys.readouterr()

    # Should print to stderr
    assert "Failed to export skills: Test exception" in captured.err

def test_mcp_export_cli_invalid_command(capsys):
    """Test that the CLI handles invalid commands."""
    with pytest.raises(SystemExit) as excinfo:
        main(["invalid_command"])

    assert excinfo.value.code != 0
