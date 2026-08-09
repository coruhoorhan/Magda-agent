import pytest
import asyncio
from typing import Dict, Any, List
from magda_agent.skills.registry import SkillRegistry
from magda_agent.integration.mcp_export_v2 import MCPExporterV2

def dummy_sync_skill(a: int, b: str) -> str:
    """A dummy synchronous skill."""
    return f"{b}_{a}"

async def dummy_async_skill(x: int) -> int:
    """A dummy asynchronous skill."""
    await asyncio.sleep(0.01)
    return x * 2

def dummy_error_skill() -> str:
    raise ValueError("Test error")

def test_get_json_schema():
    registry = SkillRegistry()
    exporter = MCPExporterV2(registry)
    schema = exporter._get_json_schema(dummy_sync_skill)
    assert schema["type"] == "object"
    assert "a" in schema["properties"]
    assert "b" in schema["properties"]
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["type"] == "string"
    assert "a" in schema["required"]
    assert "b" in schema["required"]

def test_list_tools():
    registry = SkillRegistry()
    registry.register_skill("sync_skill", dummy_sync_skill, "Sync description")
    exporter = MCPExporterV2(registry)

    tools = exporter.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "sync_skill"
    assert tools[0]["description"] == "Sync description"
    assert tools[0]["inputSchema"]["type"] == "object"

def test_call_tool_sync():
    registry = SkillRegistry()
    registry.register_skill("sync_skill", dummy_sync_skill, "Sync description")
    exporter = MCPExporterV2(registry)

    async def run_test():
        result = await exporter.call_tool("sync_skill", {"a": 42, "b": "test"})
        assert result["isError"] is False
        assert result["content"][0]["text"] == "test_42"

    asyncio.run(run_test())

def test_call_tool_async():
    registry = SkillRegistry()
    registry.register_skill("async_skill", dummy_async_skill, "Async description")
    exporter = MCPExporterV2(registry)

    async def run_test():
        result = await exporter.call_tool("async_skill", {"x": 5})
        assert result["isError"] is False
        assert result["content"][0]["text"] == "10"

    asyncio.run(run_test())

def test_call_tool_not_found():
    registry = SkillRegistry()
    exporter = MCPExporterV2(registry)

    async def run_test():
        result = await exporter.call_tool("missing_skill", {})
        assert result["isError"] is True
        assert "not found" in result["content"][0]["text"]

    asyncio.run(run_test())

def test_handle_rpc_request_list():
    registry = SkillRegistry()
    registry.register_skill("sync_skill", dummy_sync_skill, "Sync description")
    exporter = MCPExporterV2(registry)

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list"
    }

    async def run_test():
        response = await exporter.handle_rpc_request(request)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) == 1
        assert response["result"]["tools"][0]["name"] == "sync_skill"

    asyncio.run(run_test())

def test_handle_rpc_request_call():
    registry = SkillRegistry()
    registry.register_skill("sync_skill", dummy_sync_skill, "Sync description")
    exporter = MCPExporterV2(registry)

    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "sync_skill",
            "arguments": {"a": 10, "b": "val"}
        }
    }

    async def run_test():
        response = await exporter.handle_rpc_request(request)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert response["result"]["isError"] is False
        assert response["result"]["content"][0]["text"] == "val_10"

    asyncio.run(run_test())

def test_handle_rpc_request_invalid():
    registry = SkillRegistry()
    exporter = MCPExporterV2(registry)

    async def run_test():
        # Missing JSON-RPC version
        response = await exporter.handle_rpc_request({"id": 3, "method": "tools/list"})
        assert "error" in response
        assert response["error"]["code"] == -32600

        # Missing method
        response = await exporter.handle_rpc_request({"jsonrpc": "2.0", "id": 4})
        assert "error" in response
        assert response["error"]["code"] == -32600

        # Invalid method
        response = await exporter.handle_rpc_request({"jsonrpc": "2.0", "id": 5, "method": "unknown"})
        assert "error" in response
        assert response["error"]["code"] == -32601

        # Missing tool name in call
        response = await exporter.handle_rpc_request({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {}})
        assert "error" in response
        assert response["error"]["code"] == -32602

    asyncio.run(run_test())
