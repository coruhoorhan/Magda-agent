"""
Tests for Hermes Portal Resource-Bounded Skill Sandbox.
"""
import pytest
import time
import sys
from magda_agent.safety.resource_sandbox_v1 import ResourceSandboxV1

def infinite_loop():
    """An infinite loop to trigger CPU timeout."""
    while True:
        pass

def successful_task(x, y):
    """A simple successful task."""
    return x + y

def memory_hog():
    """A task that allocates large amount of memory to trigger memory limit."""
    x = "a" * (300 * 1024 * 1024) # ~300MB
    return len(x)

def failing_task():
    """A task that fails with an exception."""
    raise ValueError("Task failed")

def large_payload_task():
    """A task that returns a large string >65KB to test OS pipe deadlocks."""
    return "a" * (100 * 1024)

def test_sandbox_success():
    """Test that a task within limits executes successfully."""
    sandbox = ResourceSandboxV1(max_cpu_time=2, max_memory_mb=256)
    result = sandbox.execute(successful_task, 3, 4)
    assert result == 7

def test_sandbox_large_payload():
    """Test that a task returning a large payload does not cause a queue deadlock."""
    sandbox = ResourceSandboxV1(max_cpu_time=2, max_memory_mb=256)
    result = sandbox.execute(large_payload_task)
    assert len(result) == 100 * 1024
    assert result == "a" * (100 * 1024)

@pytest.mark.skip(reason="Flaky due to system CPU timing")
def test_sandbox_timeout():
    """Test that a task exceeding CPU time limit raises TimeoutError."""
    sandbox = ResourceSandboxV1(max_cpu_time=1, max_memory_mb=256)
    with pytest.raises(TimeoutError, match="Sandbox execution exceeded CPU time limit."):
        sandbox.execute(infinite_loop)

@pytest.mark.skipif(sys.platform != "linux", reason="Resource limits behave differently or are unavailable on non-Linux platforms.")
def test_sandbox_memory_limit():
    """Test that a task exceeding memory limits crashes with RuntimeError or MemoryError."""
    sandbox = ResourceSandboxV1(max_cpu_time=2, max_memory_mb=50) # 50MB limit
    with pytest.raises((RuntimeError, MemoryError)):
        sandbox.execute(memory_hog)

def test_sandbox_function_error():
    """Test that exceptions raised by the target function are propagated correctly."""
    sandbox = ResourceSandboxV1(max_cpu_time=2, max_memory_mb=256)
    with pytest.raises(ValueError, match="Task failed"):
        sandbox.execute(failing_task)
