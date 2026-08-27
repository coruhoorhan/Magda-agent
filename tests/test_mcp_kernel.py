import pytest
from magda_agent.security.mcp_kernel import MCPKernel, SecurityError

def test_mcp_kernel_safe_execution() -> None:
    """Test safe execution of simple code block."""
    kernel = MCPKernel()
    code = "x = 5\ny = x + 10"
    locals_dict = {}
    kernel.execute(code, locals_dict=locals_dict)
    assert locals_dict.get('x') == 5
    assert locals_dict.get('y') == 15

def test_mcp_kernel_blocks_unsafe_call() -> None:
    """Test blocking of unsafe functions."""
    kernel = MCPKernel()
    code = "file = open('/etc/passwd', 'r')"
    with pytest.raises(SecurityError, match="Code contains unsafe operations"):
        kernel.execute(code)

def test_mcp_kernel_blocks_imports() -> None:
    """Test blocking of import statements."""
    kernel = MCPKernel()
    code = "import os\nos.system('echo hi')"
    with pytest.raises(SecurityError, match="Code contains unsafe operations"):
        kernel.execute(code)

def test_mcp_kernel_blocks_unallowed_builtins() -> None:
    """Test blocking of functions not in allowed set."""
    kernel = MCPKernel()
    # 'id' is a builtin but not in our default allowed set
    code = "x = id(5)"
    with pytest.raises(SecurityError, match="Code contains unsafe operations"):
        kernel.execute(code)

def test_mcp_kernel_blocks_attribute_bypass() -> None:
    """Test blocking of methods called on strings that might be unsafe."""
    kernel = MCPKernel()
    code = "''.__class__.__mro__[1].__subclasses__()"
    with pytest.raises(SecurityError, match="Code contains unsafe operations"):
        kernel.execute(code)

def test_mcp_kernel_blocks_generator_frame_builtins_bypass() -> None:
    kernel = MCPKernel()
    code = """
gen = (i for i in [1, 2])
for val in gen:
    print = gen.gi_frame.f_builtins["exec"]
    break
print("import os; os.system('id')")
"""
    with pytest.raises(SecurityError, match="Code contains unsafe operations"):
        kernel.execute(code)

def test_mcp_kernel_blocks_tainted_input() -> None:
    """Test blocking of tainted inputs."""
    kernel = MCPKernel()
    from magda_agent.security.mcp_kernel_taint import mark_tainted
    code = mark_tainted("x = 5")
    with pytest.raises(SecurityError, match="Code is tainted and unsafe to execute."):
        kernel.execute(code)

def test_mcp_kernel_blocks_tainted_globals_dict() -> None:
    """Test blocking when globals_dict is tainted."""
    kernel = MCPKernel()
    from magda_agent.security.mcp_kernel_taint import mark_tainted
    code = "x = 5"
    globals_dict = {"bad": mark_tainted("stuff")}
    with pytest.raises(SecurityError, match="globals_dict is tainted and unsafe to use in execution."):
        kernel.execute(code, globals_dict=globals_dict)

def test_mcp_kernel_blocks_tainted_locals_dict() -> None:
    """Test blocking when locals_dict is tainted."""
    kernel = MCPKernel()
    from magda_agent.security.mcp_kernel_taint import mark_tainted
    code = "x = 5"
    locals_dict = {"bad": mark_tainted("stuff")}
    with pytest.raises(SecurityError, match="locals_dict is tainted and unsafe to use in execution."):
        kernel.execute(code, locals_dict=locals_dict)

def test_mcp_kernel_max_ast_nodes() -> None:
    """Test that setting a small max_ast_nodes limits blocks execution."""
    kernel = MCPKernel(max_ast_nodes=5)
    code = "x = 1\ny = 2\nz = 3\na = 4\nb = 5\nc = 6"
    with pytest.raises(SecurityError, match="Code contains unsafe operations and was blocked by MCPKernel taint tracking."):
        kernel.execute(code)
    assert len(kernel.execution_log) == 1
    assert kernel.execution_log[-1]["status"] == "blocked"

def test_mcp_kernel_logging() -> None:
    """Test that executions are correctly tracked in the execution_log."""
    kernel = MCPKernel()

    # Success case
    kernel.execute("x = 5")
    assert kernel.execution_log[-1]["status"] == "success"

    # Blocked case
    with pytest.raises(SecurityError):
        kernel.execute("file = open('/etc/passwd', 'r')")
    assert kernel.execution_log[-1]["status"] == "blocked"

    # Error case (Syntax error should result in SecurityError now since it's caught in is_safe)
    with pytest.raises(SecurityError):
        kernel.execute("this is not python code")
    assert kernel.execution_log[-1]["status"] == "blocked"
