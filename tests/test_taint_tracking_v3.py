"""Tests for MCPKernel Taint Tracking V3 and TaintedEpisodicMemory."""
import pytest

from magda_agent.safety.taint_tracking_v2 import PolicyViolationError
from magda_agent.safety.taint_tracking_v3 import MCPKernelV3, TaintedEpisodicMemory, TaintTrackerV3


def test_taint_tracker_v3_initialization() -> None:
    """Test that TaintTrackerV3 can be successfully initialized and behaves like TaintTrackerV2."""
    tracker = TaintTrackerV3()
    assert tracker is not None

    # Verify basic inheritance functionality
    tainted = tracker.taint("test", "user_prompt")
    assert tracker.is_tainted(tainted)
    assert tracker.get_origins(tainted) == {"user_prompt"}


def test_tainted_episodic_memory_store_and_recall() -> None:
    """Test storing and recalling tainted events in TaintedEpisodicMemory preserves taint tracks."""
    tracker = TaintTrackerV3()
    # Initialize with ephemeral ChromaDB client (via ":memory:") for speed and isolation
    memory = TaintedEpisodicMemory(persist_directory=":memory:", tracker=tracker)

    # 1. Store clean event
    memory.store_event("Clean event content", metadata={"importance": 0.9}, user_id=42)

    # 2. Store tainted event
    tainted_text = tracker.taint("Tainted event command", "untrusted_user")
    memory.store_event(tainted_text, metadata={"importance": 1.0}, user_id=42)

    # 3. Retrieve all events and verify taint propagation
    events = memory.get_all_events(user_id=42)
    assert len(events) == 2

    # Map by content
    clean_retrieved = next(e for e in events if "Clean" in e["text"])
    tainted_retrieved = next(e for e in events if "Tainted" in e["text"])

    # Verify clean event has no taint
    assert not tracker.is_tainted(clean_retrieved["text"])

    # Verify tainted event has reconstructed taint tracks
    assert tracker.is_tainted(tainted_retrieved["text"])
    assert tracker.get_origins(tainted_retrieved["text"]) == {"untrusted_user"}

    # 4. Recall events based on query and verify reconstructed taints
    recalled_clean = memory.recall_events(query="Clean", top_k=1, user_id=42)
    assert len(recalled_clean) == 1
    assert not tracker.is_tainted(recalled_clean[0])

    recalled_tainted = memory.recall_events(query="Tainted", top_k=1, user_id=42)
    assert len(recalled_tainted) == 1
    assert tracker.is_tainted(recalled_tainted[0])
    assert tracker.get_origins(recalled_tainted[0]) == {"untrusted_user"}


def test_mcp_kernel_v3_clean_execution() -> None:
    """Test that MCPKernelV3 successfully executes a clean tool."""
    kernel = MCPKernelV3()

    def my_clean_tool(data: str) -> str:
        return f"Processed: {data}"

    result = kernel.execute_tool(my_clean_tool, {"data": "clean_input"}, is_sensitive=True)
    assert result == "Processed: clean_input"
    assert not kernel.tracker.is_tainted(result)


def test_mcp_kernel_v3_taint_propagation() -> None:
    """Test that MCPKernelV3 correctly propagates taints when is_sensitive=False."""
    kernel = MCPKernelV3()

    def my_tool(data: str) -> str:
        return f"Processed: {data}"

    tainted_input = kernel.tracker.taint("tainted_string", "user_xyz")
    result = kernel.execute_tool(my_tool, {"data": tainted_input}, is_sensitive=False)

    assert result == "Processed: tainted_string"
    assert kernel.tracker.is_tainted(result)
    assert kernel.tracker.get_origins(result) == {"user_xyz"}


def test_mcp_kernel_v3_blocks_sensitive() -> None:
    """Test that MCPKernelV3 blocks sensitive tools when inputs are tainted."""
    kernel = MCPKernelV3()

    def my_sensitive_tool(data: str) -> str:
        return f"Executed sensitive action with: {data}"

    tainted_input = kernel.tracker.taint("tainted_command", "malicious_user")

    with pytest.raises(PolicyViolationError) as exc_info:
        kernel.execute_tool(my_sensitive_tool, {"data": tainted_input}, is_sensitive=True)

    assert "Tainted input to sensitive tool call from origins" in str(exc_info.value)
    assert "malicious_user" in str(exc_info.value)
