"""
Unit tests for MCP Action Tools Export Compatibility V2 (V8 Exporter).
"""

import json
import unittest
from typing import Dict, List, Optional

try:
    from magda_agent.skills.mcp_export_v8 import (
        MCPActionToolsExporterV8,
        MCPToolDefinitionV8,
        MCPToolSchemaExtractorV8,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "skills"
        / "mcp_export_v8.py"
    )
    spec = importlib.util.spec_from_file_location("mcp_export_v8", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    MCPActionToolsExporterV8 = module.MCPActionToolsExporterV8
    MCPToolDefinitionV8 = module.MCPToolDefinitionV8
    MCPToolSchemaExtractorV8 = module.MCPToolSchemaExtractorV8


class MockSkillRegistry:
    def __init__(self):
        self.skills = {}
        self.descriptions = {}

    def register(self, name, func, desc=""):
        self.skills[name] = func
        self.descriptions[name] = desc


class TestMCPExportV8(unittest.TestCase):
    def setUp(self):
        self.exporter = MCPActionToolsExporterV8(server_name="magda-test-server", version="2.0.0")

    def test_export_function_schema_generation(self):
        def sample_tool(query: str, limit: int = 10, filters: Optional[List[str]] = None) -> List[str]:
            """
            Search repository records.

            Args:
                query: The search query string.
                limit: Maximum results to return.
                filters: Optional category filters.
            """
            return ["result"]

        tool_def = self.exporter.export_callable(sample_tool, name="search_records")

        self.assertEqual(tool_def.name, "search_records")
        self.assertIn("Search repository records", tool_def.description)

        schema = tool_def.inputSchema
        self.assertEqual(schema["type"], "object")
        self.assertIn("query", schema["properties"])
        self.assertIn("limit", schema["properties"])
        self.assertIn("filters", schema["properties"])

        # Check types
        self.assertEqual(schema["properties"]["query"]["type"], "string")
        self.assertEqual(schema["properties"]["query"]["description"], "The search query string.")
        self.assertEqual(schema["properties"]["limit"]["type"], "integer")
        self.assertEqual(schema["properties"]["limit"]["default"], 10)
        self.assertEqual(schema["properties"]["filters"]["type"], "array")

        # Required fields: query has no default, limit has default 10
        self.assertIn("query", schema["required"])
        self.assertNotIn("limit", schema["required"])

        # Output schema
        self.assertIsNotNone(tool_def.outputSchema)
        self.assertEqual(tool_def.outputSchema["type"], "array")

    def test_export_skill_registry(self):
        registry = MockSkillRegistry()

        def add_nums(a: int, b: int) -> int:
            return a + b

        def fetch_weather(city: str) -> str:
            return f"Weather for {city}"

        registry.register("add_nums", add_nums, "Adds two numbers together")
        registry.register("fetch_weather", fetch_weather, "Fetches city weather")

        exported_tools = self.exporter.export_skill_registry(registry)
        self.assertEqual(len(exported_tools), 2)

        names = {t.name for t in exported_tools}
        self.assertEqual(names, {"add_nums", "fetch_weather"})

    def test_manifest_generation_and_json(self):
        tool1 = MCPToolDefinitionV8(
            name="tool_1",
            description="Tool one",
            inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
        )

        manifest_json = self.exporter.export_to_json([tool1])
        parsed = json.loads(manifest_json)

        self.assertEqual(parsed["server"]["name"], "magda-test-server")
        self.assertEqual(parsed["server"]["version"], "2.0.0")
        self.assertEqual(parsed["protocol_version"], "2024-11-05")
        self.assertEqual(len(parsed["tools"]), 1)
        self.assertEqual(parsed["tools"][0]["name"], "tool_1")


if __name__ == "__main__":
    unittest.main()
