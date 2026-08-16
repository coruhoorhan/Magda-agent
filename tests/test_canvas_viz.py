import pytest
from magda_agent.memory.canvas_viz import MemoryCanvasVisualizer

def test_convert_to_nodes_empty() -> None:
    """Test converting an empty list of events returns an empty list."""
    visualizer = MemoryCanvasVisualizer()
    nodes = visualizer.convert_to_nodes([])
    assert nodes == []

def test_convert_to_nodes_valid_events() -> None:
    """Test converting a list of valid events into canvas nodes."""
    visualizer = MemoryCanvasVisualizer()
    events = [
        {
            "id": "event-1",
            "text": "This is a short memory.",
            "metadata": {"user_id": 123, "decayed": False}
        },
        {
            "id": "event-2",
            "text": "This is a much longer memory that will exceed the fifty character limit for the label and thus should be truncated properly.",
            "metadata": {"user_id": 456, "decayed": True}
        }
    ]

    nodes = visualizer.convert_to_nodes(events)

    assert len(nodes) == 2

    # Check the first node
    assert nodes[0]["id"] == "event-1"
    assert nodes[0]["type"] == "episodic_memory"
    assert nodes[0]["label"] == "This is a short memory."
    assert nodes[0]["data"]["full_text"] == "This is a short memory."
    assert nodes[0]["data"]["metadata"] == {"user_id": 123, "decayed": False}

    # Check the second node (truncation)
    assert nodes[1]["id"] == "event-2"
    assert nodes[1]["type"] == "episodic_memory"
    assert nodes[1]["label"] == "This is a much longer memory that will exceed the ..."
    assert nodes[1]["data"]["full_text"] == events[1]["text"]
    assert nodes[1]["data"]["metadata"] == {"user_id": 456, "decayed": True}

def test_convert_to_nodes_missing_keys() -> None:
    """Test converting events that are missing some expected keys."""
    visualizer = MemoryCanvasVisualizer()
    events = [
        {
            "text": "Missing ID and metadata"
        }
    ]

    nodes = visualizer.convert_to_nodes(events)

    assert len(nodes) == 1
    assert nodes[0]["id"] == "unknown-id"
    assert nodes[0]["type"] == "episodic_memory"
    assert nodes[0]["label"] == "Missing ID and metadata"
    assert nodes[0]["data"]["full_text"] == "Missing ID and metadata"
    assert nodes[0]["data"]["metadata"] == {}
