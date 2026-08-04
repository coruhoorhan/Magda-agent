import pytest
from magda_agent.visualization.canvas_patching_v5 import CanvasPatchManagerV5

def test_generate_patch_add():
    """Test generating a patch for adding a new key."""
    old_state = {"user1": {"id": 1}}
    new_state = {"user1": {"id": 1, "name": "test"}}
    patches = CanvasPatchManagerV5.generate_patch(old_state, new_state)
    assert len(patches) == 1
    assert patches[0] == {"op": "add", "path": "/user1/name", "value": "test"}

def test_generate_patch_remove():
    """Test generating a patch for removing a key."""
    old_state = {"user1": {"id": 1, "name": "test"}}
    new_state = {"user1": {"id": 1}}
    patches = CanvasPatchManagerV5.generate_patch(old_state, new_state)
    assert len(patches) == 1
    assert patches[0] == {"op": "remove", "path": "/user1/name"}

def test_generate_patch_replace():
    """Test generating a patch for replacing a value."""
    old_state = {"user1": {"id": 1, "name": "old"}}
    new_state = {"user1": {"id": 1, "name": "new"}}
    patches = CanvasPatchManagerV5.generate_patch(old_state, new_state)
    assert len(patches) == 1
    assert patches[0] == {"op": "replace", "path": "/user1/name", "value": "new"}

def test_generate_patch_complex():
    """Test generating a complex patch with multiple operations."""
    old_state = {
        "user1": {"id": 1, "tags": ["a", "b"]},
        "user2": {"id": 2, "name": "old"}
    }
    new_state = {
        "user1": {"id": 1, "tags": ["a", "b", "c"]}, # list changed, should trigger a replace on the list
        "user3": {"id": 3} # added
    }
    patches = CanvasPatchManagerV5.generate_patch(old_state, new_state)

    # We should have:
    # 1. replace /user1/tags
    # 2. add /user3
    # 3. remove /user2

    assert len(patches) == 3

    # Order doesn't strictly matter for the test if we check elements
    expected = [
        {"op": "add", "path": "/user3", "value": {"id": 3}},
        {"op": "replace", "path": "/user1/tags", "value": ["a", "b", "c"]},
        {"op": "remove", "path": "/user2"}
    ]

    for e in expected:
        assert e in patches

def test_apply_patch():
    """Test applying a patch to a state."""
    state = {
        "user1": {"id": 1, "name": "old"},
        "user2": {"id": 2}
    }

    patches = [
        {"op": "replace", "path": "/user1/name", "value": "new"},
        {"op": "add", "path": "/user3", "value": {"id": 3}},
        {"op": "remove", "path": "/user2"}
    ]

    new_state = CanvasPatchManagerV5.apply_patch(state, patches)

    expected = {
        "user1": {"id": 1, "name": "new"},
        "user3": {"id": 3}
    }

    assert new_state == expected

def test_end_to_end_patching():
    """Test generating a patch and applying it to get the exact new state."""
    old_state = {
        "users": [
            {"id": 1, "role": "admin"},
            {"id": 2, "role": "user"}
        ],
        "settings": {
            "theme": "dark",
            "notifications": True
        }
    }

    new_state = {
        "users": [
            {"id": 1, "role": "superadmin"},
            {"id": 3, "role": "guest"}
        ],
        "settings": {
            "theme": "light",
            "volume": 80
        },
        "version": "1.0"
    }

    patches = CanvasPatchManagerV5.generate_patch(old_state, new_state)
    applied_state = CanvasPatchManagerV5.apply_patch(old_state, patches)

    assert applied_state == new_state

def test_apply_patch_root_replace():
    """Test applying a replace operation on the root."""
    old_state = {"a": 1}
    new_state = {"b": 2}

    patches = [{"op": "replace", "path": "/", "value": {"b": 2}}]
    applied_state = CanvasPatchManagerV5.apply_patch(old_state, patches)

    assert applied_state == new_state
