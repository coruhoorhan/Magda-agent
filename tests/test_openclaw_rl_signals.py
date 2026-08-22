"""
Tests for the OpenClawRLSignals module.
"""

import pytest
from magda_agent.learning.openclaw_rl_signals import OpenClawRLSignals

def test_process_signal() -> None:
    """Test that a signal is correctly processed and stored."""
    processor = OpenClawRLSignals()

    # Process a positive user reply
    signal1 = processor.process_signal(
        source="user_reply",
        content="That worked perfectly, thanks!",
        sentiment_score=0.8
    )

    assert signal1["source"] == "user_reply"
    assert signal1["reward"] == 0.8
    assert signal1["is_positive"] is True

    # Process a negative tool output
    signal2 = processor.process_signal(
        source="tool_output",
        content="Error: file not found.",
        sentiment_score=-0.9
    )

    assert signal2["source"] == "tool_output"
    assert signal2["reward"] == -0.9
    assert signal2["is_positive"] is False

def test_implicit_reward_calculation() -> None:
    """Test that implicit rewards are calculated correctly when sentiment is 0.0."""
    processor = OpenClawRLSignals()

    # Implicit positive
    signal_pos = processor.process_signal(
        source="user_reply",
        content="That's great!",
    )
    assert signal_pos["reward"] == 0.5
    assert signal_pos["is_positive"] is True

    # Implicit negative
    signal_neg = processor.process_signal(
        source="tool_output",
        content="Error: file not found",
    )
    assert signal_neg["reward"] == -0.5
    assert signal_neg["is_positive"] is False

    # Neutral
    signal_neu = processor.process_signal(
        source="user_reply",
        content="Okay.",
    )
    assert signal_neu["reward"] == 0.0
    assert signal_neu["is_positive"] is False

def test_clear_history() -> None:
    """Test that the signal history can be cleared."""
    processor = OpenClawRLSignals()
    processor.process_signal(source="user", content="hello", sentiment_score=0.1)

    assert len(processor.signal_history) == 1
    processor.clear_history()
    assert len(processor.signal_history) == 0

def test_get_recent_signals() -> None:
    """Test retrieving recent signals respects the limit."""
    processor = OpenClawRLSignals()

    for i in range(15):
        processor.process_signal(
            source="user_reply",
            content=f"Message {i}",
            sentiment_score=0.1
        )

    recent = processor.get_recent_signals(limit=5)
    assert len(recent) == 5
    assert recent[-1]["content"] == "Message 14"
