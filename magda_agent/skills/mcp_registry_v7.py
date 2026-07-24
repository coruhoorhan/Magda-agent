import logging
from typing import Dict, Any, List

class MCPRegistryV7:
    """
    Registry specialized in handling tools exported via the Model Context Protocol (MCP) version 7.
    Allows for dynamic registration and unregistration of tools at runtime.
    """

    def __init__(self) -> None:
        """Initialize the MCP Registry V7."""
        self.mcp_tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, tool_schema: Dict[str, Any]) -> bool:
        """
        Dynamically registers and verifies an external MCP tool schema.

        Args:
            tool_schema (Dict[str, Any]): The MCP tool schema dictionary.

        Returns:
            bool: True if registered successfully, False otherwise.
        """
        if not self._is_valid_schema(tool_schema):
            logging.error(f"Failed to register MCP tool: Invalid schema {tool_schema}")
            return False

        name: str = tool_schema["name"]
        self.mcp_tools[name] = tool_schema
        logging.info(f"Successfully registered MCP tool: {name}")
        return True

    def _is_valid_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Verifies if the schema complies with MCP tool standards.
        Also validates that 'inputSchema' is a dictionary if provided.

        Args:
            schema (Dict[str, Any]): The MCP tool schema.

        Returns:
            bool: True if valid, False otherwise.
        """
        if not isinstance(schema, dict):
            return False

        required_fields = ["name", "description"]
        for field in required_fields:
            if field not in schema or not isinstance(schema[field], str) or not schema[field]:
                return False

        # Validate inputSchema if it exists
        if "inputSchema" in schema and not isinstance(schema["inputSchema"], dict):
            return False

        return True

    def get_tool(self, name: str) -> Dict[str, Any]:
        """
        Retrieve a registered MCP tool by name.

        Args:
            name (str): The name of the MCP tool.

        Returns:
            Dict[str, Any]: The tool schema, or an empty dictionary if not found.
        """
        return self.mcp_tools.get(name, {})

    def list_tools(self) -> List[str]:
        """
        List all available registered MCP tools.

        Returns:
            List[str]: A list of tool names.
        """
        return list(self.mcp_tools.keys())

    def unregister_tool(self, name: str) -> bool:
        """
        Dynamically unregisters and removes an MCP tool from the registry.

        Args:
            name (str): The name of the MCP tool to unregister.

        Returns:
            bool: True if the tool was successfully unregistered, False if not found.
        """
        if name in self.mcp_tools:
            del self.mcp_tools[name]
            logging.info(f"Successfully unregistered MCP tool: {name}")
            return True
        logging.warning(f"Failed to unregister MCP tool: {name} not found in registry.")
        return False
