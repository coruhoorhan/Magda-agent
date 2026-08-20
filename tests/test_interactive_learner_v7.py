import pytest
from unittest.mock import MagicMock
from magda_agent.learning.interactive_learner_v7 import OpenClawInteractiveLearnerV7

@pytest.mark.asyncio
async def test_process_conversation_trace_positive():
    habit_tracker = MagicMock()
    mirror_neurons = MagicMock()
    user_model = MagicMock()

    user_model.get_model.return_value = {"communication_style": "friendly"}
    mirror_neurons.empathize.return_value = (0.3, 0.1, 0.0)

    learner = OpenClawInteractiveLearnerV7(
        habit_tracker=habit_tracker,
        mirror_neurons=mirror_neurons,
        user_model=user_model,
    )

    trace = [
        {
            "user_reply": "That was great and amazing!",
            "action_context": "code_search",
            "skills_used": ["search_code_skill"],
            "tool_output": "Success",
        },
        {
            "user_reply": "Awesome work, excellent results!",
            "action_context": "file_write",
            "skills_used": ["write_file_skill"],
            "tool_output": "Written 100 bytes",
        },
    ]

    result = await learner.process_conversation_trace(trace=trace, user_id=101)

    assert result["total_turns"] == 2
    assert result["reward"] > 5.0
    assert "search_code_skill" in result["skills_updated"]
    assert "write_file_skill" in result["skills_updated"]

    assert habit_tracker.record_usage.call_count == 2
    user_model.save_model.assert_called_once()
    saved_model = user_model.save_model.call_args[0][1]
    assert "(trace_validated)" in saved_model["communication_style"]

@pytest.mark.asyncio
async def test_process_conversation_trace_negative():
    habit_tracker = MagicMock()
    mirror_neurons = MagicMock()
    user_model = MagicMock()

    user_model.get_model.return_value = {"communication_style": "neutral"}
    mirror_neurons.empathize.return_value = (-0.4, 0.1, 0.0)

    learner = OpenClawInteractiveLearnerV7(
        habit_tracker=habit_tracker,
        mirror_neurons=mirror_neurons,
        user_model=user_model,
    )

    trace = [
        {
            "user_reply": "This is terrible and bad error",
            "action_context": "execute_code",
            "skills_used": ["run_code_skill"],
            "tool_output": "Error 500",
        }
    ]

    result = await learner.process_conversation_trace(trace=trace, user_id=202)

    assert result["total_turns"] == 1
    assert result["reward"] < 5.0
    assert result["skills_updated"] == []

    habit_tracker.record_usage.assert_not_called()
    user_model.save_model.assert_called_once()
    saved_model = user_model.save_model.call_args[0][1]
    assert "(trace_cautious)" in saved_model["communication_style"]

@pytest.mark.asyncio
async def test_process_empty_trace():
    habit_tracker = MagicMock()
    mirror_neurons = MagicMock()
    user_model = MagicMock()

    learner = OpenClawInteractiveLearnerV7(
        habit_tracker=habit_tracker,
        mirror_neurons=mirror_neurons,
        user_model=user_model,
    )

    result = await learner.process_conversation_trace(trace=[], user_id=303)

    assert result["total_turns"] == 0
    assert result["reward"] == 0.0
    habit_tracker.record_usage.assert_not_called()
    user_model.save_model.assert_not_called()
