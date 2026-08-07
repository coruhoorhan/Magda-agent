import pytest
from magda_agent.security.mcp_taint import mcp_action_taint_sandbox, TaintSandboxError
from magda_agent.security.mcp_kernel_taint import mark_tainted, is_tainted

def test_mcp_action_taint_sandbox_blocks_critical_param() -> None:
    """Tests that the sandbox blocks a critical parameter if it is tainted."""
    @mcp_action_taint_sandbox(critical_params=["command"])
    def execute_command(command: str, background: bool = False) -> str:
        return f"Executed {command}"

    tainted_command = mark_tainted("rm -rf /")

    with pytest.raises(TaintSandboxError, match="Critical parameter 'command' in 'execute_command' received tainted data."):
        execute_command(tainted_command)

def test_mcp_action_taint_sandbox_allows_clean_param() -> None:
    """Tests that the sandbox allows a clean critical parameter."""
    @mcp_action_taint_sandbox(critical_params=["command"])
    def execute_command(command: str, background: bool = False) -> str:
        return f"Executed {command}"

    clean_command = "ls -l"
    result = execute_command(clean_command)
    assert result == "Executed ls -l"

def test_mcp_action_taint_sandbox_taints_output() -> None:
    """Tests that the sandbox automatically taints the output of the wrapped function."""
    @mcp_action_taint_sandbox(critical_params=["query"])
    def search_web(query: str) -> dict:
        return {"results": [f"Result for {query}"]}

    clean_query = "weather"
    result = search_web(clean_query)

    assert is_tainted(result) is True
    assert result == {"results": ["Result for weather"]}

def test_mcp_action_taint_sandbox_allows_taint_on_non_critical_param() -> None:
    """Tests that the sandbox allows tainted data on non-critical parameters."""
    @mcp_action_taint_sandbox(critical_params=["query"])
    def search_web(query: str, extra_info: str = "") -> dict:
        return {"results": [f"Result for {query} with {extra_info}"]}

    clean_query = "weather"
    tainted_extra = mark_tainted("bad_data")

    result = search_web(clean_query, extra_info=tainted_extra)
    assert is_tainted(result) is True
    assert result == {"results": [f"Result for {clean_query} with {tainted_extra}"]}

def test_mcp_action_taint_sandbox_v4_param_type_hints() -> None:
    """Tests that the v4 sandbox handles type hints correctly."""
    @mcp_action_taint_sandbox(critical_params=["count"])
    def process_data(count: int, msg: str = "ok") -> str:
        return f"{msg} {count}"

    result = process_data(10)
    assert is_tainted(result) is True
    assert result == "ok 10"


from magda_agent.security.mcp_taint import advanced_mcp_taint_tracker

def test_advanced_mcp_taint_tracker_tainted_arg() -> None:
    """Tests that tainted arg results in tainted output."""
    @advanced_mcp_taint_tracker()
    def process(data: str) -> str:
        return f"Processed {data}"

    tainted_data = mark_tainted("secret")
    result = process(tainted_data)
    assert is_tainted(result) is True
    assert result == "Processed secret"

def test_advanced_mcp_taint_tracker_tainted_kwarg() -> None:
    """Tests that tainted kwarg results in tainted output."""
    @advanced_mcp_taint_tracker()
    def process(data: str, meta: str = "") -> str:
        return f"Processed {data} with {meta}"

    clean_data = "public"
    tainted_meta = mark_tainted("hidden")
    result = process(clean_data, meta=tainted_meta)
    assert is_tainted(result) is True
    assert result == "Processed public with hidden"

def test_advanced_mcp_taint_tracker_clean_inputs() -> None:
    """Tests that clean inputs result in clean output."""
    @advanced_mcp_taint_tracker()
    def process(data: str) -> str:
        return f"Processed {data}"

    clean_data = "public"
    result = process(clean_data)
    assert is_tainted(result) is False
    assert result == "Processed public"

def test_mcp_action_taint_sandbox_blocks_nested_dict() -> None:
    """Tests that the sandbox blocks a critical parameter if a nested dict value is tainted."""
    @mcp_action_taint_sandbox(critical_params=["config"])
    def execute(config: dict) -> str:
        return "ok"

    with pytest.raises(TaintSandboxError, match="Critical parameter 'config' in 'execute' received tainted data."):
        execute({"cmd": {"run": mark_tainted("rm -rf")}})

def test_mcp_action_taint_sandbox_allows_clean_nested_dict() -> None:
    """Tests that the sandbox allows a clean nested dict."""
    @mcp_action_taint_sandbox(critical_params=["config"])
    def execute(config: dict) -> str:
        return "ok"

    result = execute({"cmd": {"run": "ls"}})
    assert result == "ok"

def test_mcp_action_taint_sandbox_dot_notation() -> None:
    """Tests that the sandbox checks specific dot-notation paths."""
    @mcp_action_taint_sandbox(critical_params=["config.cmd.run"])
    def execute(config: dict) -> str:
        return "ok"

    # Should block if the specific path is tainted
    with pytest.raises(TaintSandboxError, match="Critical parameter 'config.cmd.run' in 'execute' received tainted data."):
        execute({"cmd": {"run": mark_tainted("rm -rf")}})

    # Should allow if a different path is tainted
    result = execute({"cmd": {"run": "ls"}, "other": mark_tainted("bad")})
    assert result == "ok"

    # Should allow if the path doesn't exist
    result2 = execute({"other": "safe"})
    assert result2 == "ok"
