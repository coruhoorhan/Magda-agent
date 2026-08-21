import pytest
import os
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from magda_agent.agents.git_conflict_resolver_v2 import GitConflictResolverV2
from magda_agent.llm_client import LLMClient
from magda_agent.architecture.agent_teams_v4 import AgentWorktreeIsolationV4

@pytest.fixture
def mock_llm_client():
    llm = MagicMock(spec=LLMClient)
    llm.chat_completion = AsyncMock()
    return llm

@pytest.fixture
def mock_isolation_manager():
    manager = MagicMock(spec=AgentWorktreeIsolationV4)
    manager.active_worktrees = {"agent_123": "/tmp/mock_worktree_123"}
    return manager

@pytest.fixture
def conflict_resolver(mock_llm_client, mock_isolation_manager):
    return GitConflictResolverV2(llm=mock_llm_client, isolation_manager=mock_isolation_manager)

@pytest.mark.asyncio
async def test_resolve_conflicts_success(conflict_resolver, mock_llm_client, tmp_path):
    agent_id = "agent_123"

    # Setup mock worktree dir
    worktree_dir = tmp_path / "mock_worktree_123"
    worktree_dir.mkdir()
    conflict_resolver.isolation_manager.active_worktrees[agent_id] = str(worktree_dir)

    # Setup mock file with conflict markers
    file_path = "src/main.py"
    full_path = worktree_dir / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    conflict_content = """def hello():
<<<<<<< HEAD
    print("Hello from HEAD")
=======
    print("Hello from other branch")
>>>>>>> branch-name
"""
    full_path.write_text(conflict_content, encoding="utf-8")

    # Configure mock LLM response
    resolved_content = "def hello():\n    print(\"Hello world\")"
    mock_llm_client.chat_completion.return_value = f"```python\n{resolved_content}\n```"

    # Mock asyncio.create_subprocess_exec for `git diff` and `git add`
    async def mock_create_subprocess_exec(*args, **kwargs):
        mock_process = MagicMock()
        if args[:3] == ("git", "diff", "--name-only"):
            mock_process.communicate = AsyncMock(return_value=(b"src/main.py\n", b""))
            mock_process.returncode = 0
        elif args[:2] == ("git", "add"):
            mock_process.communicate = AsyncMock(return_value=(b"", b""))
            mock_process.returncode = 0
        else:
            mock_process.communicate = AsyncMock(return_value=(b"", b""))
            mock_process.returncode = 0
        return mock_process

    with patch("magda_agent.agents.git_conflict_resolver_v2.asyncio.create_subprocess_exec", side_effect=mock_create_subprocess_exec):
        result = await conflict_resolver.resolve_conflicts(agent_id)

    assert result is True

    # Verify file was updated correctly
    updated_content = full_path.read_text(encoding="utf-8")
    assert updated_content == resolved_content

    # Verify LLM was called
    mock_llm_client.chat_completion.assert_called_once()
    call_args = mock_llm_client.chat_completion.call_args[0][0]
    assert "src/main.py" in call_args[0]["content"]

@pytest.mark.asyncio
async def test_resolve_conflicts_no_active_worktree(conflict_resolver):
    result = await conflict_resolver.resolve_conflicts("non_existent_agent")
    assert result is False

@pytest.mark.asyncio
async def test_resolve_conflicts_no_conflicts(conflict_resolver, tmp_path):
    agent_id = "agent_123"
    worktree_dir = tmp_path / "mock_worktree_123"
    worktree_dir.mkdir()
    conflict_resolver.isolation_manager.active_worktrees[agent_id] = str(worktree_dir)

    async def mock_create_subprocess_exec(*args, **kwargs):
        mock_process = MagicMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.returncode = 0
        return mock_process

    with patch("magda_agent.agents.git_conflict_resolver_v2.asyncio.create_subprocess_exec", side_effect=mock_create_subprocess_exec):
        result = await conflict_resolver.resolve_conflicts(agent_id)

    assert result is True
