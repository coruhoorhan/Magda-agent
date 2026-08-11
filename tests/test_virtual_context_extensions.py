import pytest
from magda_agent.architecture.virtual_context_extensions import VirtualContextEngineExtension

def test_create_sandbox() -> None:
    """Tests creation and retrieval of a virtual context sandbox."""
    engine = VirtualContextEngineExtension()
    engine.create_sandbox("agent_1", {"key1": "val1"})

    ctx = engine.get_sandbox("agent_1")
    assert ctx == {"key1": "val1"}

def test_create_sandbox_existing() -> None:
    """Tests creating a sandbox that already exists logs a warning and doesn't overwrite."""
    engine = VirtualContextEngineExtension()
    engine.create_sandbox("agent_1", {"key1": "val1"})
    engine.create_sandbox("agent_1", {"key2": "val2"})

    ctx = engine.get_sandbox("agent_1")
    assert ctx == {"key1": "val1"}

def test_update_sandbox() -> None:
    """Tests updating a value in a virtual context sandbox."""
    engine = VirtualContextEngineExtension()
    engine.create_sandbox("agent_1")
    engine.update_sandbox("agent_1", "key1", "val1")

    ctx = engine.get_sandbox("agent_1")
    assert ctx["key1"] == "val1"

def test_update_sandbox_not_found() -> None:
    """Tests updating a sandbox that doesn't exist raises ValueError."""
    engine = VirtualContextEngineExtension()
    with pytest.raises(ValueError, match="Sandbox not found"):
        engine.update_sandbox("missing_agent", "key1", "val1")

def test_get_sandbox_not_found() -> None:
    """Tests getting a sandbox that doesn't exist raises ValueError."""
    engine = VirtualContextEngineExtension()
    with pytest.raises(ValueError, match="Sandbox not found"):
        engine.get_sandbox("missing_agent")

def test_isolation() -> None:
    """Tests that memory spaces do not bleed across subagents."""
    engine = VirtualContextEngineExtension()
    engine.create_sandbox("agent_1", {"global": True})
    engine.create_sandbox("agent_2", {"global": True})

    engine.update_sandbox("agent_1", "agent_1_only", True)

    ctx1 = engine.get_sandbox("agent_1")
    ctx2 = engine.get_sandbox("agent_2")

    assert "agent_1_only" in ctx1
    assert "agent_1_only" not in ctx2

def test_get_sandbox_returns_copy() -> None:
    """Tests that get_sandbox returns a copy to prevent accidental mutations."""
    engine = VirtualContextEngineExtension()
    engine.create_sandbox("agent_1", {"key1": "val1"})

    ctx = engine.get_sandbox("agent_1")
    ctx["key1"] = "mutated"

    ctx_stored = engine.get_sandbox("agent_1")
    assert ctx_stored["key1"] == "val1"

def test_remove_sandbox() -> None:
    """Tests removing a sandbox."""
    engine = VirtualContextEngineExtension()
    engine.create_sandbox("agent_1")
    engine.remove_sandbox("agent_1")

    with pytest.raises(ValueError):
        engine.get_sandbox("agent_1")

def test_remove_sandbox_not_found() -> None:
    """Tests removing a sandbox that doesn't exist is safe."""
    engine = VirtualContextEngineExtension()
    engine.remove_sandbox("missing_agent")  # Should not raise
