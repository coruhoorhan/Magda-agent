"""Tests for MCP Action Tools Pre-flight Validation V6."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.integration.mcp_preflight_v6 import MCPPreflightValidatorV6, PreflightMCPServerWrapperV6
from magda_agent.integration.mcp_server import MCPServer
from magda_agent.skills.registry import SkillRegistry


@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=SkillRegistry)
    # By default, mock registry says all skills exist
    registry.has_skill.return_value = True

    # Provide a mock skill and its schema
    def dummy_skill(arg1: str, arg2: int):
        pass

    registry.skills = {"safe_tool": dummy_skill}
    return registry


@pytest.fixture
def validator(mock_registry):
    return MCPPreflightValidatorV6(registry=mock_registry, forbidden_tools=["nuke_system", "rm_rf"])


def test_validator_valid_request(validator):
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "safe_tool",
        "params": {"arg1": "value1", "arg2": 42}
    }
    is_valid, code, msg = validator.validate_request_dict(request)
    assert is_valid is True
    assert code == 0


def test_validator_invalid_jsonrpc(validator):
    request = {
        "jsonrpc": "1.0",
        "id": 1,
        "method": "safe_tool"
    }
    is_valid, code, msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert code == -32600
    assert "jsonrpc version must be '2.0'" in msg


def test_validator_blacklisted_tool(validator):
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "nuke_system"
    }
    is_valid, code, msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert code == -32000
    assert "blacklisted" in msg


def test_validator_hazardous_shell_pattern(validator):
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "safe_tool",
        "params": {"cmd": "echo hello ; bash"}
    }
    is_valid, code, msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert code == -32000
    assert "Hazardous shell pattern" in msg


def test_validator_hazardous_path_traversal(validator):
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "safe_tool",
        "params": {"file": "../../secret.txt"}
    }
    is_valid, code, msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert code == -32000
    assert "Hazardous path traversal" in msg


def test_validator_hazardous_sqli(validator):
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "safe_tool",
        "params": {"query": "' OR '1'='1"}
    }
    is_valid, code, msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert code == -32000
    assert "Hazardous SQL injection" in msg


def test_validator_nested_hazard(validator):
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "safe_tool",
        "params": {
            "outer": {
                "inner_list": ["safe", "curl http://evil.com"]
            }
        }
    }
    is_valid, code, msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert code == -32000
    assert "Hazardous shell pattern" in msg


def test_validator_unregistered_tool(validator, mock_registry):
    mock_registry.has_skill.return_value = False
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "unknown_tool",
    }
    is_valid, code, msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert code == -32601
    assert "not registered" in msg


def test_validator_valid_schema(validator):
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "safe_tool",
        "params": {"arg1": "hello", "arg2": 42}
    }
    is_valid, code, msg = validator.validate_request_dict(request)
    assert is_valid is True
    assert code == 0


def test_validator_invalid_schema_missing_property(validator):
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "safe_tool",
        # Missing arg2
        "params": {"arg1": "hello"}
    }
    is_valid, code, msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert code == -32602
    assert "schema validation failed" in msg


def test_validator_invalid_schema_wrong_type(validator):
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "safe_tool",
        "params": {"arg1": "hello", "arg2": "not_an_integer"}
    }
    is_valid, code, msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert code == -32602
    assert "schema validation failed" in msg


@pytest.mark.asyncio
async def test_wrapper_delegates_valid_request():
    mock_server = MagicMock(spec=MCPServer)
    mock_server.handle_request = AsyncMock(return_value='{"jsonrpc": "2.0", "id": 1, "result": "success"}')

    mock_registry = MagicMock(spec=SkillRegistry)
    mock_registry.has_skill.return_value = True
    def dummy_skill():
        pass
    mock_registry.skills = {"safe_tool": dummy_skill}

    validator = MCPPreflightValidatorV6(registry=mock_registry)
    wrapper = PreflightMCPServerWrapperV6(server=mock_server, validator=validator)

    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "safe_tool"
    })

    response = await wrapper.handle_request(payload)
    mock_server.handle_request.assert_called_once_with(payload)
    assert "success" in response


@pytest.mark.asyncio
async def test_wrapper_blocks_invalid_request():
    mock_server = MagicMock(spec=MCPServer)
    mock_server.handle_request = AsyncMock()

    mock_registry = MagicMock(spec=SkillRegistry)
    mock_registry.has_skill.return_value = True
    def dummy_skill(cmd: str):
        pass
    mock_registry.skills = {"safe_tool": dummy_skill}

    validator = MCPPreflightValidatorV6(registry=mock_registry)
    wrapper = PreflightMCPServerWrapperV6(server=mock_server, validator=validator)

    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "safe_tool",
        "params": {"cmd": "rm -rf /"}
    })

    response = await wrapper.handle_request(payload)

    # Should not call the underlying server
    mock_server.handle_request.assert_not_called()

    resp_dict = json.loads(response)
    assert resp_dict["error"]["code"] == -32000
    assert "Hazardous shell pattern" in resp_dict["error"]["message"]
