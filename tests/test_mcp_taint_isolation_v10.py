"""Tests for MCPKernel Tool Taint Isolation V10."""

import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from magda_agent.security.mcp_kernel_taint import mark_tainted
from magda_agent.security.mcp_taint_isolation_v10 import IsolatedSubprocessProxy, IsolationError


@patch("subprocess.run")
def test_isolated_subprocess_proxy_clean_args(mock_run: MagicMock) -> None:
    """Test executing a command with untainted arguments."""
    mock_run.return_value = subprocess.CompletedProcess(args=["ls", "-la"], returncode=0)

    result = IsolatedSubprocessProxy.run(["ls", "-la"])

    assert result.returncode == 0
    mock_run.assert_called_once()

    # Verify that the isolation flag is not set
    call_kwargs = mock_run.call_args.kwargs
    assert "__MCP_ISOLATED_NETWORK" not in call_kwargs.get("env", {})
    assert "preexec_fn" not in call_kwargs or call_kwargs["preexec_fn"] is None


@patch("subprocess.run")
@patch("magda_agent.security.mcp_taint_isolation_v10._drop_network")
def test_isolated_subprocess_proxy_tainted_args(mock_drop_network: MagicMock, mock_run: MagicMock) -> None:
    """Test executing a command with tainted arguments."""
    mock_run.return_value = subprocess.CompletedProcess(args=["curl", "http://example.com"], returncode=0)

    tainted_args = mark_tainted(["curl", "http://example.com"])
    result = IsolatedSubprocessProxy.run(tainted_args)

    assert result.returncode == 0
    mock_run.assert_called_once()

    # Verify the arguments were sanitized
    call_args = mock_run.call_args.args[0]
    assert type(call_args[0]) is str
    assert type(call_args[1]) is str

    # Verify isolation mechanisms were engaged
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["env"]["__MCP_ISOLATED_NETWORK"] == "1"
    assert call_kwargs["preexec_fn"] is not None

    # Call the preexec_fn to ensure it attempts to drop the network
    call_kwargs["preexec_fn"]()
    mock_drop_network.assert_called_once()


@patch("subprocess.run")
@patch("magda_agent.security.mcp_taint_isolation_v10._drop_network")
def test_isolated_subprocess_proxy_tainted_env(mock_drop_network: MagicMock, mock_run: MagicMock) -> None:
    """Test executing a command with tainted environment variables."""
    mock_run.return_value = subprocess.CompletedProcess(args=["env"], returncode=0)

    tainted_env = mark_tainted({"SECRET": "value"})
    result = IsolatedSubprocessProxy.run(["env"], env=tainted_env)

    assert result.returncode == 0
    mock_run.assert_called_once()

    # Verify the environment variables were sanitized
    call_kwargs = mock_run.call_args.kwargs
    assert type(call_kwargs["env"]["SECRET"]) is str
    assert call_kwargs["env"]["SECRET"] == "value"

    # Verify isolation mechanisms were engaged
    assert call_kwargs["env"]["__MCP_ISOLATED_NETWORK"] == "1"
    assert call_kwargs["preexec_fn"] is not None

    # Call the preexec_fn to ensure it attempts to drop the network
    call_kwargs["preexec_fn"]()
    mock_drop_network.assert_called_once()


@patch("subprocess.run")
@patch("magda_agent.security.mcp_taint_isolation_v10._drop_network")
def test_isolated_subprocess_proxy_existing_preexec(mock_drop_network: MagicMock, mock_run: MagicMock) -> None:
    """Test executing a tainted command with an existing preexec_fn."""
    mock_run.return_value = subprocess.CompletedProcess(args=["ls"], returncode=0)

    mock_custom_preexec = MagicMock()

    tainted_args = mark_tainted(["ls"])
    result = IsolatedSubprocessProxy.run(tainted_args, preexec_fn=mock_custom_preexec)

    assert result.returncode == 0

    # Call the wrapped preexec_fn
    call_kwargs = mock_run.call_args.kwargs
    call_kwargs["preexec_fn"]()

    # Both drop_network and the custom preexec_fn should be called
    mock_drop_network.assert_called_once()
    mock_custom_preexec.assert_called_once()


@patch("subprocess.run")
@patch("magda_agent.security.mcp_taint_isolation_v10._drop_network", side_effect=IsolationError("Failed"))
def test_isolated_subprocess_proxy_drop_network_fails(mock_drop_network: MagicMock, mock_run: MagicMock) -> None:
    """Test that execution fails securely if drop_network raises an IsolationError."""
    tainted_args = mark_tainted(["curl", "http://example.com"])

    # The run method returns, but the preexec_fn would throw when called
    IsolatedSubprocessProxy.run(tainted_args)

    call_kwargs = mock_run.call_args.kwargs

    with pytest.raises(IsolationError, match="Failed"):
        call_kwargs["preexec_fn"]()
