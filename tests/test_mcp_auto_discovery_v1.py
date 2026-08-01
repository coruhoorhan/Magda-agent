import os
import pytest
from unittest.mock import MagicMock
from magda_agent.skills.mcp_auto_discovery_v1 import MCPAutoDiscoveryV1
from magda_agent.skills.mcp_registry_v7 import MCPRegistryV7

@pytest.fixture
def dummy_directory(tmp_path):
    """Creates a dummy directory with a valid and an invalid Python file."""
    # Create valid python file
    valid_py = tmp_path / "valid_tool.py"
    valid_py.write_text(
        "def mcp_tool_test(param1: int, param2: str = 'default'):\n"
        "    \"\"\"This is a test tool.\"\"\"\n"
        "    pass\n"
        "\n"
        "def helper_func():\n"
        "    pass\n" # no docstring, should be ignored
        "\n"
        "def _private_func():\n"
        "    \"\"\"This should be ignored.\"\"\"\n"
        "    pass\n",
        encoding="utf-8"
    )

    # Create invalid python file
    invalid_py = tmp_path / "invalid.py"
    invalid_py.write_text(
        "def syntax_error(:\n"
        "    pass",
        encoding="utf-8"
    )

    return tmp_path

def test_mcp_auto_discovery(dummy_directory):
    """Test discovering and registering tools from a directory."""
    mock_registry = MagicMock(spec=MCPRegistryV7)

    discovery = MCPAutoDiscoveryV1(registry=mock_registry)
    discovery.discover_and_register(str(dummy_directory))

    # Expect one call to register_tool for mcp_tool_test
    # helper_func should be ignored because it lacks a docstring
    # _private_func should be ignored because it's private
    assert mock_registry.register_tool.call_count == 1

    args, kwargs = mock_registry.register_tool.call_args
    schema = args[0]

    assert schema["name"] == "mcp_tool_test"
    assert schema["description"] == "This is a test tool."
    assert "inputSchema" in schema
    assert "properties" in schema["inputSchema"]

    properties = schema["inputSchema"]["properties"]
    assert "param1" in properties
    assert properties["param1"]["type"] == "integer"
    assert "param2" in properties
    assert properties["param2"]["type"] == "string"

    required = schema["inputSchema"]["required"]
    assert "param1" in required
    assert "param2" not in required

def test_mcp_auto_discovery_invalid_dir():
    """Test discovery on an invalid directory."""
    mock_registry = MagicMock(spec=MCPRegistryV7)

    discovery = MCPAutoDiscoveryV1(registry=mock_registry)
    discovery.discover_and_register("/nonexistent_directory/path")

    assert mock_registry.register_tool.call_count == 0
