import pytest
import jsonschema
from typing import Dict, Any
from magda_agent.skills.mcp_validator import (
    MCPActionToolValidator,
    validate_mcp_tool,
    MCPRegistrationError
)
from magda_agent.skills.mcp_registry import MCPRegistry

def test_validate_schema_valid() -> None:
    schema = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": {
            "type": "object",
            "properties": {
                "arg1": {"type": "string"}
            }
        }
    }
    # Should not raise any exception
    MCPActionToolValidator.validate_schema(schema)

def test_validate_schema_missing_name() -> None:
    schema = {
        "description": "A test tool"
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        MCPActionToolValidator.validate_schema(schema)

def test_validate_schema_invalid_type() -> None:
    schema = {
        "name": 123,  # Invalid type, should be string
        "description": "A test tool"
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        MCPActionToolValidator.validate_schema(schema)

def test_validate_schema_invalid_input_schema_type() -> None:
    schema = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": {
            "type": "array" # Invalid type, should be object
        }
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        MCPActionToolValidator.validate_schema(schema)

def test_validate_mcp_tool_decorator_valid() -> None:
    class MockRegistry:
        @validate_mcp_tool
        def register_tool(self, tool_schema: Dict[str, Any]) -> bool:
            return True

    registry = MockRegistry()
    schema = {
        "name": "test_tool",
        "description": "A test tool"
    }
    assert registry.register_tool(schema) is True

def test_validate_mcp_tool_decorator_invalid() -> None:
    class MockRegistry:
        @validate_mcp_tool
        def register_tool(self, tool_schema: Dict[str, Any]) -> bool:
            return True

    registry = MockRegistry()
    schema = {
        "description": "A test tool" # Missing name
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        registry.register_tool(tool_schema=schema)

def test_validate_mcp_tool_decorator_positional_invalid() -> None:
    class MockRegistry:
        @validate_mcp_tool
        def register_tool(self, tool_schema: Dict[str, Any]) -> bool:
            return True

    registry = MockRegistry()
    schema = {
        "description": "A test tool" # Missing name
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        registry.register_tool(schema) # Call with positional argument


# --- New interceptor and MCPRegistry tests ---

def test_mcp_registry_interceptor_valid() -> None:
    """Test that a valid schema registers successfully with the intercepted load_tool."""
    registry = MCPRegistry()
    schema = {
        "name": "valid_tool",
        "description": "This is a valid tool description.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            }
        }
    }
    assert registry.load_tool(schema) is True
    assert "valid_tool" in registry.list_tools()

def test_mcp_registry_interceptor_invalid() -> None:
    """Test that an invalid schema raises MCPRegistrationError during load_tool."""
    registry = MCPRegistry()
    invalid_schema = {
        "name": "invalid_tool",
        "description": "Missing inputSchema structure.",
        "inputSchema": {
            "type": "array"  # Should be object
        }
    }
    with pytest.raises(MCPRegistrationError) as exc_info:
        registry.load_tool(invalid_schema)

    assert "Schema validation failed" in str(exc_info.value)

def test_mcp_registry_add_custom_interceptor() -> None:
    """Test that we can dynamically add a custom interceptor and it runs during registration."""
    registry = MCPRegistry()
    called = []

    def custom_interceptor(schema: Dict[str, Any]) -> None:
        called.append(schema.get("name"))
        if schema.get("name") == "blocked_tool":
            raise ValueError("This tool is explicitly blocked by custom interceptor.")

    # Add the custom interceptor
    registry.add_interceptor(custom_interceptor)

    # Valid schema for other tools
    schema1 = {
        "name": "allowed_tool",
        "description": "An allowed tool",
        "inputSchema": {
            "type": "object"
        }
    }
    assert registry.load_tool(schema1) is True
    assert called == ["allowed_tool"]

    # Blocked schema
    schema2 = {
        "name": "blocked_tool",
        "description": "A blocked tool",
        "inputSchema": {
            "type": "object"
        }
    }
    with pytest.raises(MCPRegistrationError) as exc_info:
        registry.load_tool(schema2)

    assert "Registration interceptor error" in str(exc_info.value)
    assert "This tool is explicitly blocked" in str(exc_info.value)
    assert "blocked_tool" in called
