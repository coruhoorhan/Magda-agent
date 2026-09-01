"""
Tests for MemGPTVirtualContextManagerV2.
"""

from magda_agent.memory.memgpt_virtual_context_v2 import MemGPTVirtualContextManagerV2

def test_initialization() -> None:
    """Test that the manager initializes correctly."""
    manager = MemGPTVirtualContextManagerV2(max_tokens=500)
    assert manager.max_tokens == 500
    assert manager.get_active_context() == []
    assert manager.get_episodic_memory() == []

def test_enforces_token_limits() -> None:
    """Test that memories within limits remain in active context."""
    # max_tokens = 10. len // 4 -> tokens
    # "hello world!" -> 12 // 4 = 3 tokens
    # "test test" -> 9 // 4 = 2 tokens
    manager = MemGPTVirtualContextManagerV2(max_tokens=10)

    manager.add_memory("hello world!")
    manager.add_memory("test test")

    active = manager.get_active_context()
    assert len(active) == 2
    assert "hello world!" in active
    assert "test test" in active
    assert len(manager.get_episodic_memory()) == 0

def test_episodic_memory_swapping() -> None:
    """Test that adding memories exceeding the limit swaps oldest out."""
    # max_tokens = 5.
    # "memory 1 length 16" -> 18 // 4 = 4 tokens (Wait, length of "memory 1 length 16" is 18)
    # Let's use exact strings.
    # "aaaabbbbccccdddd" -> 16 // 4 = 4 tokens
    # "eeeeffff" -> 8 // 4 = 2 tokens
    # Total = 6 tokens. Limit = 5. Oldest should swap.

    manager = MemGPTVirtualContextManagerV2(max_tokens=5)

    manager.add_memory("aaaabbbbccccdddd") # 4 tokens
    manager.add_memory("eeeeffff") # 2 tokens

    # After adding the second memory, total tokens = 6.
    # The oldest memory ("aaaabbbbccccdddd") is swapped out.
    # Remaining active: "eeeeffff" (2 tokens).

    active = manager.get_active_context()
    episodic = manager.get_episodic_memory()

    assert len(active) == 1
    assert active[0] == "eeeeffff"

    assert len(episodic) == 1
    assert episodic[0] == "aaaabbbbccccdddd"

def test_large_memory_addition() -> None:
    """Test adding a memory that alone exceeds the limit."""
    manager = MemGPTVirtualContextManagerV2(max_tokens=3)

    # Add a memory that takes 2 tokens
    manager.add_memory("12345678")

    # Add a memory that takes 4 tokens.
    # Total tokens before removal = 6.
    # "12345678" will be removed. "1234567890123456" stays as the only element.
    # (Since we check len(active_context) > 1, the single large item remains).
    manager.add_memory("1234567890123456")

    active = manager.get_active_context()
    episodic = manager.get_episodic_memory()

    assert len(active) == 1
    assert active[0] == "1234567890123456"
    assert len(episodic) == 1
    assert episodic[0] == "12345678"
