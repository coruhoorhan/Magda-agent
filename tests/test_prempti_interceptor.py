import pytest
import time
from magda_agent.safety.audit_trail import AuditTrail
from magda_agent.safety.prempti_interceptor import PremptiInterceptor
from magda_agent.safety.taint import TaintedString, mark_tainted

@pytest.fixture
def audit_trail():
    return AuditTrail(max_capacity=100, db_path=None)

@pytest.fixture
def interceptor(audit_trail):
    return PremptiInterceptor(audit_trail=audit_trail)

def test_interceptor_normal_call(interceptor, audit_trail):
    @interceptor.intercept(tool_name="my_tool", why="just testing")
    def dummy_tool(a: str, b: int) -> str:
        return f"{a}_{b}"

    res = dummy_tool("hello", 42)
    assert res == "hello_42"

    logs = audit_trail.get_all()
    assert len(logs) == 1
    log = logs[0]

    assert log["tool_name"] == "my_tool"
    assert log["why"] == "just testing"
    assert log["result"] == "hello_42"
    assert log["kwargs"]["a"] == "hello"
    assert log["kwargs"]["b"] == 42
    assert log["kwargs"]["_tainted_boundary_crossover"] is False

def test_interceptor_tainted_argument(interceptor, audit_trail):
    @interceptor.intercept(why="testing taint")
    def dummy_tool_tainted(a: str) -> str:
        return f"safe_{a}"

    res = dummy_tool_tainted(mark_tainted("evil_data"))

    logs = audit_trail.get_all()
    assert len(logs) == 1
    log = logs[0]

    assert log["tool_name"] == "dummy_tool_tainted"
    assert log["kwargs"]["_tainted_boundary_crossover"] is True

def test_interceptor_tainted_result(interceptor, audit_trail):
    @interceptor.intercept()
    def dummy_tool_output() -> str:
        return mark_tainted("evil_output")

    res = dummy_tool_output()

    logs = audit_trail.get_all()
    assert len(logs) == 1
    log = logs[0]

    assert log["kwargs"]["_tainted_boundary_crossover"] is True

@pytest.mark.asyncio
async def test_interceptor_async_call(interceptor, audit_trail):
    @interceptor.intercept(tool_name="async_tool")
    async def dummy_async_tool(x: int):
        return x * 2

    res = await dummy_async_tool(5)
    assert res == 10

    logs = audit_trail.get_all()
    assert len(logs) == 1
    assert logs[0]["tool_name"] == "async_tool"
    assert logs[0]["kwargs"]["_tainted_boundary_crossover"] is False
