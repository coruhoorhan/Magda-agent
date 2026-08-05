import pytest
from magda_agent.visualization.procedural_ui_v2 import ProceduralMemoryVisualizerV2

def test_apply_patches_add_replace_remove() -> None:
    """
    Test that patches correctly apply to the internal state of the visualizer.
    """
    visualizer = ProceduralMemoryVisualizerV2()

    # 1. Test Add
    patches_add = [
        {"op": "add", "path": "/skill_1", "value": 0.5},
        {"op": "add", "path": "/skill_2", "value": 0.8}
    ]
    visualizer.apply_patches(patches_add)
    assert visualizer.state == {"skill_1": 0.5, "skill_2": 0.8}

    # 2. Test Replace
    patches_replace = [
        {"op": "replace", "path": "/skill_1", "value": 0.2}
    ]
    visualizer.apply_patches(patches_replace)
    assert visualizer.state == {"skill_1": 0.2, "skill_2": 0.8}

    # 3. Test Remove
    patches_remove = [
        {"op": "remove", "path": "/skill_2"}
    ]
    visualizer.apply_patches(patches_remove)
    assert visualizer.state == {"skill_1": 0.2}

def test_get_ui_structure() -> None:
    """
    Test that get_ui_structure correctly formats the internal state, maps colors based
    on thresholds, and sorts the skills by weight descending.
    """
    visualizer = ProceduralMemoryVisualizerV2()
    visualizer.state = {
        "skill_low": 0.1,    # should be red
        "skill_mid": 0.5,    # should be yellow
        "skill_high": 0.9    # should be green
    }

    ui_struct = visualizer.get_ui_structure()

    assert ui_struct["type"] == "ProceduralMemoryVisualization"
    assert ui_struct["status"] == "live"
    assert ui_struct["total_skills"] == 3

    skills = ui_struct["skills"]
    assert len(skills) == 3

    # Verify Sorting (highest first)
    assert skills[0]["skill_id"] == "skill_high"
    assert skills[1]["skill_id"] == "skill_mid"
    assert skills[2]["skill_id"] == "skill_low"

    # Verify color assignments
    assert skills[0]["color"] == "green"
    assert skills[1]["color"] == "yellow"
    assert skills[2]["color"] == "red"
