import inspect
import asyncio
from typing import Dict, Any, List, Optional
from magda_agent.skills.registry import SkillRegistry

class MCPExporterV2:
    """
    MCP Server adapter for Magda's SkillRegistry.
    Converts Magda skills into standard MCP tools, and provides a standard JSON-RPC 2.0 interface.
    """
    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def _get_json_schema(self, func: Any) -> Dict[str, Any]:
        """
        Extracts JSON schema parameters from the function signature.
        """
        if hasattr(func, "__mcp_schema__"):
            return getattr(func, "__mcp_schema__")

        sig = inspect.signature(func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            if name in ('self', 'kwargs', 'args'):
                continue

            param_type = "string"
            if param.annotation is not inspect.Parameter.empty:
                if param.annotation is int:
                    param_type = "integer"
                elif param.annotation is float:
                    param_type = "number"
                elif param.annotation is bool:
                    param_type = "boolean"
                elif param.annotation is list or param.annotation is List:
                    param_type = "array"
                elif param.annotation is dict or param.annotation is Dict:
                    param_type = "object"

            properties[name] = {"type": param_type}

            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Lists all registered skills as MCP-compatible tool definitions.
        """
        tools = []
        for name, func in self.registry.skills.items():
            description = self.registry.descriptions.get(name, "")
            schema = self._get_json_schema(func)
            tools.append({
                "name": name,
                "description": description,
                "inputSchema": schema
            })
        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invokes a skill via the MCP protocol format. Awaits if the underlying skill is async.
        """
        if not self.registry.has_skill(name):
            return {
                "content": [{"type": "text", "text": f"Error: Tool '{name}' not found."}],
                "isError": True
            }

        try:
            result = self.registry.execute_skill(name, **arguments)
            if inspect.isawaitable(result):
                result = await result
            return {
                "content": [{"type": "text", "text": str(result)}],
                "isError": False
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error executing tool {name}: {e}"}],
                "isError": True
            }

    async def handle_rpc_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles a JSON-RPC 2.0 formatted request mapping to MCP tools.
        Supported methods:
        - "tools/list"
        - "tools/call"
        """
        if "jsonrpc" not in request or request["jsonrpc"] != "2.0":
            return self._build_error(request.get("id"), -32600, "Invalid Request")

        method = request.get("method")
        if not method:
            return self._build_error(request.get("id"), -32600, "Invalid Request: missing method")

        req_id = request.get("id")

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": self.list_tools()
                }
            }
        elif method == "tools/call":
            params = request.get("params", {})
            name = params.get("name")
            if not name:
                return self._build_error(req_id, -32602, "Invalid params: missing tool name")

            arguments = params.get("arguments", {})
            tool_result = await self.call_tool(name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": tool_result
            }
        else:
            return self._build_error(req_id, -32601, f"Method not found: {method}")

    def _build_error(self, req_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": code,
                "message": message
            }
        }
