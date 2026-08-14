import pytest
from magda_agent.learning.openclaw_rl_pipeline_v9 import OpenClawRLNextStatePipelineV9

def test_parse_text_signal_positive():
    pipeline = OpenClawRLNextStatePipelineV9()
    assert pipeline.parse_text_signal("This is great!") == 1.0
    assert pipeline.parse_text_signal("Thanks for the help") == 1.0

def test_parse_text_signal_negative():
    pipeline = OpenClawRLNextStatePipelineV9()
    assert pipeline.parse_text_signal("This is bad.") == -1.0
    assert pipeline.parse_text_signal("No, that's wrong") == -1.0

def test_parse_text_signal_neutral():
    pipeline = OpenClawRLNextStatePipelineV9()
    assert pipeline.parse_text_signal("Okay.") == 0.0

def test_calculate_reward_with_tool_output():
    pipeline = OpenClawRLNextStatePipelineV9()

    # Positive text, positive tool output
    reward = pipeline.calculate_reward("great", "Command executed with success")
    assert reward == 1.0  # Max is 1.0

    # Neutral text, negative tool output
    reward = pipeline.calculate_reward("okay", "An error occurred")
    assert reward == -0.5

    # Negative text, negative tool output
    reward = pipeline.calculate_reward("wrong", "Error: failed")
    assert reward == -1.0  # Min is -1.0
