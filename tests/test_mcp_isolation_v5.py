import pytest
import os
import tempfile
import subprocess
from typing import Any, Dict
from magda_agent.teams.mcp_isolation_v5 import MCPIsolationManagerV5
from magda_agent.skills.mcp_registry_v7 import MCPRegistryV7

@pytest.fixture
def test_repo_path(tmp_path: Any) -> str:
    """Fixture to create a dummy git repository for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize a dummy git repo
    subprocess.run(["git", "init"], cwd=str(repo_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_path), check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo_path), check=True)

    # Create an initial commit
    dummy_file = repo_path / "dummy.txt"
    dummy_file.write_text("initial content")
    subprocess.run(["git", "add", "dummy.txt"], cwd=str(repo_path), check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(repo_path), check=True)

    return str(repo_path)

def test_create_and_cleanup_isolated_worktree(test_repo_path: str) -> None:
    """Test creating and cleaning up an isolated git worktree."""
    manager = MCPIsolationManagerV5(base_repo_path=test_repo_path)

    # Test creation
    worktree_path = manager.create_isolated_worktree(tool_name="test_tool")

    assert os.path.exists(worktree_path)
    assert os.path.exists(os.path.join(worktree_path, "dummy.txt"))
    assert len(manager.active_worktrees) == 1

    # Test cleanup
    manager.cleanup_worktree(worktree_path)

    assert not os.path.exists(worktree_path)
    assert len(manager.active_worktrees) == 0

def test_multiple_worktrees_isolation(test_repo_path: str) -> None:
    """Test that multiple isolated git worktrees are truly isolated from each other."""
    manager = MCPIsolationManagerV5(base_repo_path=test_repo_path)

    # Create two worktrees
    wt1 = manager.create_isolated_worktree("tool1")
    wt2 = manager.create_isolated_worktree("tool2")

    assert wt1 != wt2
    assert os.path.exists(wt1)
    assert os.path.exists(wt2)

    # Modify one worktree
    with open(os.path.join(wt1, "dummy.txt"), "w") as f:
        f.write("modified in wt1")

    # Check that the other worktree is unaffected
    with open(os.path.join(wt2, "dummy.txt"), "r") as f:
        content = f.read()
        assert content == "initial content"

    # Cleanup
    manager.cleanup_worktree(wt1)
    manager.cleanup_worktree(wt2)

def test_mcp_registry_isolation_not_required() -> None:
    """Test that isolation is bypassed if not requested by the tool."""
    registry = MCPRegistryV7()

    def dummy_tool(**kwargs: Any) -> Dict[str, Any]:
        """Mock tool that returns its arguments."""
        return {"status": "success", "cwd": kwargs.get("cwd")}

    registry.load_action_tool({
        "name": "no_isolation_tool",
        "description": "A tool that does not require isolation.",
        "requires_isolation": False,
        "func": dummy_tool
    })

    result = registry.execute_tool("no_isolation_tool", {})
    assert result["cwd"] is None

def test_mcp_registry_isolation_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that isolation is triggered and worktree path is passed if requested."""
    registry = MCPRegistryV7()

    # Mock isolation manager
    class MockIsolationManager:
        def __init__(self, base_repo_path: str) -> None:
            """Initialize mock isolation manager."""
            self.base_repo_path = base_repo_path

        def create_isolated_worktree(self, name: str) -> str:
            """Mock creating isolated worktree."""
            return "/tmp/mocked_worktree_path"

        def cleanup_worktree(self, path: str) -> None:
            """Mock cleaning up worktree."""
            pass

    registry.isolation_manager = MockIsolationManager(base_repo_path=".")

    def isolated_tool(**kwargs: Any) -> Dict[str, Any]:
        """Mock isolated tool."""
        return {"status": "success", "cwd": kwargs.get("cwd")}

    registry.load_action_tool({
        "name": "isolated_tool",
        "description": "A tool that requires isolation.",
        "requires_isolation": True,
        "func": isolated_tool
    })

    result = registry.execute_tool("isolated_tool", {})
    assert result["cwd"] == "/tmp/mocked_worktree_path"
