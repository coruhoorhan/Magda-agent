import json
import pytest
from unittest.mock import MagicMock
from magda_agent.safety.mcp_preflight_validation_v10 import MCPActionToolPreflightValidatorV10


@pytest.fixture
def weather_tool_schema():
    return {
        "name": "get_weather",
        "description": "Get weather for a location",
        "inputSchema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }


@pytest.fixture
def mock_policy():
    policy = MagicMock()
    policy.evaluate.return_value = (True, "Allowed")
    return policy


@pytest.fixture
def mock_executor():
    def _executor(tool_name, arguments):
        if tool_name == "get_weather":
            loc = arguments.get("location")
            return {"temperature": 22, "location": loc, "units": arguments.get("units", "celsius")}
        if tool_name == "failing_tool":
            raise RuntimeError("Execution catastrophic failure")
        return {"status": "success"}

    return MagicMock(side_effect=_executor)


@pytest.fixture
def validator(weather_tool_schema, mock_policy, mock_executor):
    val = MCPActionToolPreflightValidatorV10(policy_layer=mock_policy, executor=mock_executor)
    val.register_tool_schema("get_weather", weather_tool_schema)
    return val


def test_register_invalid_schema():
    val = MCPActionToolPreflightValidatorV10()
    invalid_schema = {"type": "invalid_type_name"}
    with pytest.raises(ValueError, match="Invalid JSON Schema"):
        val.register_tool_schema("bad_tool", invalid_schema)


def test_validate_args_unregistered_tool(validator):
    is_valid, err = validator.validate_args("unknown_tool", {"param": 1})
    assert is_valid is False
    assert "not registered" in err


def test_validate_args_success(validator):
    is_valid, err = validator.validate_args("get_weather", {"location": "London", "units": "celsius"})
    assert is_valid is True
    assert err == ""


def test_validate_args_missing_required(validator):
    is_valid, err = validator.validate_args("get_weather", {"units": "celsius"})
    assert is_valid is False
    assert "location" in err


def test_validate_args_invalid_type(validator):
    is_valid, err = validator.validate_args("get_weather", {"location": 12345})
    assert is_valid is False
    assert "12345 is not of type 'string'" in err


def test_validate_args_invalid_enum(validator):
    is_valid, err = validator.validate_args("get_weather", {"location": "Paris", "units": "kelvin"})
    assert is_valid is False
    assert "'kelvin' is not one of" in err


def test_validate_preflight_call_unregistered_tool(validator):
    is_allowed, code, msg = validator.validate_preflight_call("unknown_tool", {})
    assert is_allowed is False
    assert code == -32601
    assert "Method or tool not found" in msg


def test_validate_preflight_call_schema_failure(validator, mock_policy):
    is_allowed, code, msg = validator.validate_preflight_call("get_weather", {"units": "celsius"})
    assert is_allowed is False
    assert code == -32602
    assert "Invalid params" in msg
    mock_policy.evaluate.assert_not_called()


def test_validate_preflight_call_policy_blocked(validator, mock_policy):
    mock_policy.evaluate.return_value = (False, "Location restricted")
    is_allowed, code, msg = validator.validate_preflight_call("get_weather", {"location": "RestrictedCity"})
    assert is_allowed is False
    assert code == -32000
    assert "Policy evaluation blocked execution" in msg


def test_validate_preflight_call_success(validator, mock_policy):
    is_allowed, code, msg = validator.validate_preflight_call("get_weather", {"location": "Tokyo"})
    assert is_allowed is True
    assert code == 0
    assert msg == ""
    mock_policy.evaluate.assert_called_once_with("get_weather", {"location": "Tokyo"})


def test_process_jsonrpc_request_parse_error(validator):
    resp = validator.process_jsonrpc_request("invalid json payload{")
    assert resp["error"]["code"] == -32700


def test_process_jsonrpc_request_invalid_version(validator):
    resp = validator.process_jsonrpc_request({"jsonrpc": "1.0", "method": "get_weather", "id": 1})
    assert resp["error"]["code"] == -32600


def test_process_jsonrpc_request_missing_method(validator):
    resp = validator.process_jsonrpc_request({"jsonrpc": "2.0", "id": 1})
    assert resp["error"]["code"] == -32600


def test_process_jsonrpc_request_tools_call_format_success(validator, mock_executor):
    req = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "get_weather",
            "arguments": {"location": "Berlin", "units": "celsius"}
        },
        "id": 100
    }
    resp = validator.process_jsonrpc_request(req)
    assert resp["id"] == 100
    assert "result" in resp
    assert resp["result"]["location"] == "Berlin"
    mock_executor.assert_called_once_with("get_weather", {"location": "Berlin", "units": "celsius"})


def test_process_jsonrpc_request_call_tool_format_schema_error(validator, mock_executor, mock_policy):
    req = json.dumps({
        "jsonrpc": "2.0",
        "method": "call_tool",
        "params": {
            "name": "get_weather",
            "arguments": {"units": "celsius"}  # missing location
        },
        "id": 101
    })
    resp = validator.process_jsonrpc_request(req)
    assert resp["id"] == 101
    assert resp["error"]["code"] == -32602
    assert "Invalid params" in resp["error"]["message"]
    mock_policy.evaluate.assert_not_called()
    mock_executor.assert_not_called()


def test_process_jsonrpc_request_direct_method_format(validator):
    req = {
        "jsonrpc": "2.0",
        "method": "get_weather",
        "params": {"location": "Madrid"},
        "id": 102
    }
    resp = validator.process_jsonrpc_request(req)
    assert resp["id"] == 102
    assert resp["result"]["location"] == "Madrid"


def test_process_jsonrpc_request_execution_exception(validator):
    validator.register_tool_schema("failing_tool", {"type": "object"})
    req = {
        "jsonrpc": "2.0",
        "method": "failing_tool",
        "params": {},
        "id": 103
    }
    resp = validator.process_jsonrpc_request(req)
    assert resp["id"] == 103
    assert resp["error"]["code"] == -32603
    assert "Execution catastrophic failure" in resp["error"]["message"]
