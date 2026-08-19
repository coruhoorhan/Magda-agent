import os
import shutil
import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from magda_agent.isolation.workspace_provisioner import WorkspaceProvisioner


@pytest.mark.asyncio
async def test_provision_workspace_async_basic(tmp_path) -> None:
    source_repo = tmp_path / "source_repo"
    source_repo.mkdir()
    (source_repo / "test_file.txt").write_text("initial content")

    base_dir = tmp_path / "workspaces"

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        provisioner = WorkspaceProvisioner(base_dir=str(base_dir), source_repo_path=str(source_repo))
        workspace_path = await provisioner.provision_workspace_async("agent_1")

        assert "agent_1" in provisioner.active_workspaces
        assert provisioner.active_workspaces["agent_1"] == workspace_path
        assert os.path.exists(base_dir)

        await provisioner.remove_workspace_async("agent_1")
        assert "agent_1" not in provisioner.active_workspaces


@pytest.mark.asyncio
async def test_workspace_isolation_and_no_cross_contamination(tmp_path) -> None:
    source_repo = tmp_path / "source_repo"
    source_repo.mkdir()
    (source_repo / "main.py").write_text("print('hello')")

    base_dir = tmp_path / "workspaces"
    provisioner = WorkspaceProvisioner(base_dir=str(base_dir), source_repo_path=str(source_repo))

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 1  # simulate git clone fallback to dir copy

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        path_a = await provisioner.provision_workspace_async("agent_a")
        path_b = await provisioner.provision_workspace_async("agent_b")

        assert path_a != path_b
        assert os.path.exists(path_a)
        assert os.path.exists(path_b)

        # Modify file in agent A's workspace
        file_a = os.path.join(path_a, "main.py")
        with open(file_a, "w") as f:
            f.write("print('agent_a modified')")

        # Verify agent B's workspace and source repo remain unchanged
        file_b = os.path.join(path_b, "main.py")
        file_source = source_repo / "main.py"

        with open(file_b, "r") as f:
            content_b = f.read()

        assert content_b == "print('hello')"
        assert file_source.read_text() == "print('hello')"

        await provisioner.cleanup_all_async()
        assert len(provisioner.active_workspaces) == 0


@pytest.mark.asyncio
async def test_isolated_workspace_context_manager(tmp_path) -> None:
    source_repo = tmp_path / "source_repo"
    source_repo.mkdir()
    base_dir = tmp_path / "workspaces"

    provisioner = WorkspaceProvisioner(base_dir=str(base_dir), source_repo_path=str(source_repo))

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        async with provisioner.isolated_workspace("agent_ctx") as ws_path:
            assert os.path.exists(ws_path)
            assert "agent_ctx" in provisioner.active_workspaces

        # Verify cleanup after exiting context
        assert not os.path.exists(ws_path)
        assert "agent_ctx" not in provisioner.active_workspaces


@pytest.mark.asyncio
async def test_provision_team_workspaces_async(tmp_path) -> None:
    source_repo = tmp_path / "source_repo"
    source_repo.mkdir()
    base_dir = tmp_path / "workspaces"

    provisioner = WorkspaceProvisioner(base_dir=str(base_dir), source_repo_path=str(source_repo))

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    agent_ids = ["agent_1", "agent_2", "agent_3"]

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        workspaces = await provisioner.provision_team_workspaces_async(agent_ids)

        assert len(workspaces) == 3
        for aid in agent_ids:
            assert aid in workspaces
            assert os.path.exists(workspaces[aid])

        await provisioner.cleanup_all_async()
        assert len(provisioner.active_workspaces) == 0


@pytest.mark.asyncio
async def test_execute_in_workspace_async(tmp_path) -> None:
    source_repo = tmp_path / "source_repo"
    source_repo.mkdir()
    base_dir = tmp_path / "workspaces"

    provisioner = WorkspaceProvisioner(base_dir=str(base_dir), source_repo_path=str(source_repo))

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    async def sample_task(ws_path: str) -> str:
        new_file = os.path.join(ws_path, "output.txt")
        with open(new_file, "w") as f:
            f.write("task complete")
        return "SUCCESS"

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        result = await provisioner.execute_in_workspace_async("worker_1", sample_task)
        assert result == "SUCCESS"
        assert "worker_1" not in provisioner.active_workspaces


@pytest.mark.asyncio
async def test_execute_in_workspace_async_timeout(tmp_path) -> None:
    source_repo = tmp_path / "source_repo"
    source_repo.mkdir()
    base_dir = tmp_path / "workspaces"

    provisioner = WorkspaceProvisioner(base_dir=str(base_dir), source_repo_path=str(source_repo))

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    async def slow_task(ws_path: str) -> str:
        await asyncio.sleep(1.0)
        return "DONE"

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(asyncio.TimeoutError):
            await provisioner.execute_in_workspace_async("slow_worker", slow_task, timeout=0.05)

        # Ensure workspace was cleaned up despite timeout
        assert "slow_worker" not in provisioner.active_workspaces
