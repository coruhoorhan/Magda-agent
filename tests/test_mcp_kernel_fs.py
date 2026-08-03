import pytest
from pathlib import Path

from magda_agent.security.mcp_kernel import SecurityError
from magda_agent.security.mcp_kernel_fs import MCPKernelFS


def test_mcp_kernel_fs_allowed_read_write(tmp_path: Path) -> None:
    """Test that allowed paths can be read and written to."""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    fs = MCPKernelFS([allowed_dir])

    target_file = allowed_dir / "test.txt"
    fs.write_text(target_file, "hello sandbox")
    content = fs.read_text(target_file)

    assert content == "hello sandbox"


def test_mcp_kernel_fs_disallowed_read_write(tmp_path: Path) -> None:
    """Test that reading/writing outside allowed paths raises a SecurityError."""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    disallowed_dir = tmp_path / "disallowed"
    disallowed_dir.mkdir()

    fs = MCPKernelFS([allowed_dir])
    target_file = disallowed_dir / "secret.txt"

    # Should raise error when writing
    with pytest.raises(SecurityError, match="is denied by MCPKernelFS"):
        fs.write_text(target_file, "secret data")

    # Write manually to test reading
    target_file.write_text("secret data")

    # Should raise error when reading
    with pytest.raises(SecurityError, match="is denied by MCPKernelFS"):
        fs.read_text(target_file)


def test_mcp_kernel_fs_open_allowed(tmp_path: Path) -> None:
    """Test that opening a file in allowed directory works."""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    fs = MCPKernelFS([allowed_dir])

    target_file = allowed_dir / "test_open.txt"

    with fs.open(target_file, "w") as f:
        f.write("open test")

    with fs.open(target_file, "r") as f:
        content = f.read()

    assert content == "open test"


def test_mcp_kernel_fs_open_disallowed(tmp_path: Path) -> None:
    """Test that opening a file outside allowed directory raises SecurityError."""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    disallowed_dir = tmp_path / "disallowed"
    disallowed_dir.mkdir()

    fs = MCPKernelFS([allowed_dir])
    target_file = disallowed_dir / "secret_open.txt"
    target_file.write_text("secret data")

    with pytest.raises(SecurityError, match="is denied by MCPKernelFS"):
        fs.open(target_file, "r")


def test_mcp_kernel_fs_relative_path_escape(tmp_path: Path) -> None:
    """Test that path traversal attempts using .. are caught."""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    disallowed_dir = tmp_path / "disallowed"
    disallowed_dir.mkdir()

    fs = MCPKernelFS([allowed_dir])

    # Path inside allowed dir pointing outside using ..
    target_file = allowed_dir / ".." / "disallowed" / "secret.txt"

    with pytest.raises(SecurityError, match="is denied by MCPKernelFS"):
        fs.write_text(target_file, "escape attempt")
