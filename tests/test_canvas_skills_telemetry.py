import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock

from magda_agent.telemetry.canvas_skills_telemetry import CanvasSkillsTelemetry
from magda_agent.skills.registry import SkillRegistry

@pytest.mark.asyncio
async def test_canvas_skills_telemetry_broadcasts():
    websocket = AsyncMock()
    telemetry = CanvasSkillsTelemetry(websocket)

    # Test skill start
    await telemetry.broadcast_skill_start("test_skill", {"arg1": "value1"})
    websocket.send_text.assert_called_once()
    payload = json.loads(websocket.send_text.call_args[0][0])
    assert payload["type"] == "skill_start"
    assert payload["data"]["skill_name"] == "test_skill"
    assert payload["data"]["kwargs"] == {"arg1": "value1"}
    assert payload["data"]["status"] == "start"
    websocket.reset_mock()

    # Test skill success
    await telemetry.broadcast_skill_success("test_skill", {"result": "success"}, 150.0)
    websocket.send_text.assert_called_once()
    payload = json.loads(websocket.send_text.call_args[0][0])
    assert payload["type"] == "skill_success"
    assert payload["data"]["skill_name"] == "test_skill"
    assert payload["data"]["result"] == str({"result": "success"})
    assert payload["data"]["duration_ms"] == 150.0
    assert payload["data"]["status"] == "success"
    websocket.reset_mock()

    # Test skill fail
    await telemetry.broadcast_skill_fail("test_skill", "An error occurred", 50.0)
    websocket.send_text.assert_called_once()
    payload = json.loads(websocket.send_text.call_args[0][0])
    assert payload["type"] == "skill_fail"
    assert payload["data"]["skill_name"] == "test_skill"
    assert payload["data"]["error"] == "An error occurred"
    assert payload["data"]["duration_ms"] == 50.0
    assert payload["data"]["status"] == "fail"

def test_registry_integration():
    telemetry = CanvasSkillsTelemetry(websocket=AsyncMock())
    registry = SkillRegistry(canvas_telemetry=telemetry)

    # Register a simple skill
    def simple_skill(x, y):
        return x + y

    registry.register_skill("add", simple_skill, "Adds two numbers")

    telemetry.broadcast_skill_start = AsyncMock()
    telemetry.broadcast_skill_success = AsyncMock()
    telemetry.broadcast_skill_fail = AsyncMock()

    # Execute skill
    result = registry.execute_skill("add", x=5, y=3)

    assert result == 8
    telemetry.broadcast_skill_start.assert_called_once_with("add", {"x": 5, "y": 3})
    telemetry.broadcast_skill_success.assert_called_once()
    assert telemetry.broadcast_skill_success.call_args[0][0] == "add"
    assert telemetry.broadcast_skill_success.call_args[0][1] == 8

def test_registry_integration_error():
    telemetry = CanvasSkillsTelemetry(websocket=AsyncMock())
    registry = SkillRegistry(canvas_telemetry=telemetry)

    def failing_skill():
        raise ValueError("Intentional fail")

    registry.register_skill("fail_skill", failing_skill, "Fails intentionally")

    telemetry.broadcast_skill_start = AsyncMock()
    telemetry.broadcast_skill_success = AsyncMock()
    telemetry.broadcast_skill_fail = AsyncMock()

    # Execute skill that fails
    try:
        registry.execute_skill("fail_skill")
    except Exception:
        pass

    telemetry.broadcast_skill_start.assert_called_once_with("fail_skill", {})
    telemetry.broadcast_skill_success.assert_not_called()
    telemetry.broadcast_skill_fail.assert_called_once()
    assert telemetry.broadcast_skill_fail.call_args[0][0] == "fail_skill"
    assert "Intentional fail" in telemetry.broadcast_skill_fail.call_args[0][1]
