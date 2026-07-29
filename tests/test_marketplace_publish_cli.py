"""Tests for the agentskills.io marketplace publish CLI."""

import json
from unittest.mock import patch, MagicMock

import httpx
import pytest
import respx

from magda_agent.skills.marketplace_publish_cli import build_parser, publish_skills, main
from magda_agent.skills.registry import SkillRegistry


@pytest.fixture
def mock_registry():
    """Provides a mock SkillRegistry for testing without loading real skills."""
    registry = SkillRegistry()

    def dummy_skill(arg1: str, arg2: int = 5) -> str:
        """A test skill."""
        return f"{arg1} {arg2}"

    registry.skills["dummy_skill"] = dummy_skill
    registry.descriptions["dummy_skill"] = "A test skill."
    return registry


def test_build_parser():
    """Test that the argument parser is configured correctly."""
    parser = build_parser()
    args = parser.parse_args(["--endpoint", "http://test.local", "--auth-token", "secret", "--dry-run"])

    assert args.endpoint == "http://test.local"
    assert args.auth_token == "secret"
    assert args.dry_run is True


def test_build_parser_defaults():
    """Test the parser default values."""
    parser = build_parser()
    args = parser.parse_args([])

    assert args.endpoint == "https://agentskills.io/api/publish"
    assert args.auth_token is None
    assert args.dry_run is False


@patch("magda_agent.skills.marketplace_publish_cli.initialize_skills")
def test_publish_skills_dry_run(mock_init_skills, mock_registry, capsys):
    """Test that dry-run only prints the payload and does not make requests."""
    mock_init_skills.return_value = mock_registry

    # Run with dry_run
    code = publish_skills("http://test.local", dry_run=True)
    assert code == 0

    out, err = capsys.readouterr()
    assert "DRY RUN: Skill Payload" in out

    # Verify the JSON payload is what we expect
    # Need to extract the JSON from the output text
    lines = out.split("\n")
    json_lines = []
    in_json = False
    for line in lines:
        if line == "--- DRY RUN: Skill Payload ---":
            in_json = True
            continue
        elif line == "------------------------------":
            in_json = False
            continue
        if in_json:
            json_lines.append(line)

    payload_str = "\n".join(json_lines)
    payload_data = json.loads(payload_str)

    assert "skills" in payload_data
    assert len(payload_data["skills"]) == 1
    assert payload_data["skills"][0]["name"] == "dummy_skill"


@respx.mock
@patch("magda_agent.skills.marketplace_publish_cli.initialize_skills")
def test_publish_skills_success(mock_init_skills, mock_registry, capsys):
    """Test successful publishing via POST request."""
    mock_init_skills.return_value = mock_registry
    endpoint = "https://agentskills.io/api/publish"

    # Mock successful response
    respx.post(endpoint).mock(return_value=httpx.Response(200, json={"status": "ok"}))

    code = publish_skills(endpoint, auth_token="my_token")
    assert code == 0

    out, err = capsys.readouterr()
    assert "Publishing skills to" in out
    assert "Successfully published skills" in out

    # Verify request was made
    assert respx.calls.call_count == 1
    request = respx.calls[0].request
    assert request.headers["authorization"] == "Bearer my_token"
    assert request.headers["content-type"] == "application/json"

    payload = json.loads(request.content)
    assert "skills" in payload
    assert payload["skills"][0]["name"] == "dummy_skill"


@respx.mock
@patch("magda_agent.skills.marketplace_publish_cli.initialize_skills")
def test_publish_skills_http_error(mock_init_skills, mock_registry, capsys):
    """Test handling of HTTP error responses."""
    mock_init_skills.return_value = mock_registry
    endpoint = "https://agentskills.io/api/publish"

    # Mock HTTP 403 Forbidden
    respx.post(endpoint).mock(return_value=httpx.Response(403, text="Forbidden"))

    code = publish_skills(endpoint)
    assert code == 1

    out, err = capsys.readouterr()
    assert "HTTP Error: Server returned 403" in err


@respx.mock
@patch("magda_agent.skills.marketplace_publish_cli.initialize_skills")
def test_publish_skills_network_error(mock_init_skills, mock_registry, capsys):
    """Test handling of network connection errors."""
    mock_init_skills.return_value = mock_registry
    endpoint = "https://agentskills.io/api/publish"

    # Mock Network Error
    respx.post(endpoint).mock(side_effect=httpx.ConnectError("Connection refused"))

    code = publish_skills(endpoint)
    assert code == 1

    out, err = capsys.readouterr()
    assert "Network Error: Failed to connect" in err


@patch("magda_agent.skills.marketplace_publish_cli.publish_skills")
def test_main(mock_publish_skills):
    """Test the main CLI entry point."""
    mock_publish_skills.return_value = 0

    # Test arguments parsing
    code = main(["--endpoint", "http://test.local", "--auth-token", "tok", "--dry-run"])

    assert code == 0
    mock_publish_skills.assert_called_once_with(
        endpoint="http://test.local",
        auth_token="tok",
        dry_run=True
    )
