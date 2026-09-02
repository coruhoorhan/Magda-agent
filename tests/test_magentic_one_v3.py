import pytest
import json
from unittest.mock import AsyncMock
from magda_agent.llm_client import LLMClient
from magda_agent.architecture.magentic_one_v3 import MagenticOneOrchestratorV3, MagenticOneWorkerV3

@pytest.mark.asyncio
async def test_worker_state_parsing():
    mock_llm = AsyncMock(spec=LLMClient)
    worker = MagenticOneWorkerV3("TestWorker", "Test", mock_llm)

    mock_llm.chat_completion.return_value = "Done work. STATE_UPDATE: {\"key1\": \"value1\"}"
    outcome, state = await worker.execute_subtask("do this", [])

    assert outcome == "Done work."
    assert state == {"key1": "value1"}
    assert worker.state == {"key1": "value1"}

@pytest.mark.asyncio
async def test_orchestrate_state_merging():
    mock_llm = AsyncMock(spec=LLMClient)

    # Mock planning to return JSON
    plan_json = '[{"id": "t1", "description": "do step 1"}]'
    mock_llm.chat_completion.side_effect = [
        plan_json, # plan
        "result 1 STATE_UPDATE: {\"merged_list\": [1], \"k1\": \"v1\"}", # execute
        "YES, task is complete" # review
    ]

    orchestrator = MagenticOneOrchestratorV3(llm=mock_llm)

    result = await orchestrator.orchestrate("short task")

    assert "YES, task is complete" in result
    assert orchestrator.global_state == {"merged_list": [1], "k1": "v1"}
    assert len(orchestrator.active_workers) == 0

@pytest.mark.asyncio
async def test_merge_state_logic():
    mock_llm = AsyncMock(spec=LLMClient)
    orchestrator = MagenticOneOrchestratorV3(llm=mock_llm)

    orchestrator.global_state = {"list_key": [1], "dict_key": {"a": 1}, "str_key": "old"}

    updates = [
        {"list_key": [2], "dict_key": {"b": 2}, "str_key": "new"},
        {"new_key": "added"}
    ]

    orchestrator._merge_state(updates)

    assert orchestrator.global_state["list_key"] == [1, 2]
    assert orchestrator.global_state["dict_key"] == {"a": 1, "b": 2}
    assert orchestrator.global_state["str_key"] == "new"
    assert orchestrator.global_state["new_key"] == "added"

@pytest.mark.asyncio
async def test_magentic_one_v3_orchestrator_success():
    mock_llm = AsyncMock(spec=LLMClient)

    # Mock behavior:
    # Evaluate difficulty: Local heuristic returns 2 (length < 20) -> team size 1
    # Plan call: returns JSON string (team size 1)
    # Execute call: returns "Subtask done"
    # Review call: returns "YES Result complete"
    mock_llm.chat_completion.side_effect = [
        '[{"id": "test_1", "description": "Execute first part of task"}]',
        "Subtask done",
        "YES Result complete"
    ]

    orchestrator = MagenticOneOrchestratorV3(llm=mock_llm)
    result = await orchestrator.orchestrate("Do the task")

    assert "YES Result complete" in result
    assert mock_llm.chat_completion.call_count == 3

@pytest.mark.asyncio
async def test_magentic_one_v3_round_robin_execution():
    mock_llm = AsyncMock(spec=LLMClient)

    # Task > 100 characters gives difficulty 10 -> team size 5
    # Generate 5 tasks without explicit worker assignment
    plan_json = json.dumps([
        {"id": "1", "description": "Task 1"},
        {"id": "2", "description": "Task 2"},
        {"id": "3", "description": "Task 3"},
        {"id": "4", "description": "Task 4"},
        {"id": "5", "description": "Task 5"}
    ])

    mock_llm.chat_completion.side_effect = [
        plan_json,
        "Worker 1 done", # Exec 1 -> WebSurfer
        "Worker 2 done", # Exec 2 -> FileSurfer
        "Worker 3 done", # Exec 3 -> Coder
        "Worker 4 done", # Exec 4 -> Executor
        "Worker 5 done", # Exec 5 -> WebSurfer (round robin cycle)
        "YES Complete"   # Review
    ]

    orchestrator = MagenticOneOrchestratorV3(llm=mock_llm)

    # Task > 100 characters
    task_string = "Execute tasks " * 20
    await orchestrator.orchestrate(task_string)

    # 1 plan + 5 exec + 1 review = 7 calls
    assert mock_llm.chat_completion.call_count == 7

    # Let's inspect the execution calls to ensure they hit the right workers based on their description
    exec_calls = mock_llm.chat_completion.call_args_list[1:6]
    assert "WebSurfer" in exec_calls[0][0][0][0]["content"]
    assert "FileSurfer" in exec_calls[1][0][0][0]["content"]
    assert "Coder" in exec_calls[2][0][0][0]["content"]
    assert "Executor" in exec_calls[3][0][0][0]["content"]
    assert "DynamicWorker" in exec_calls[4][0][0][0]["content"]


@pytest.mark.asyncio
async def test_magentic_one_v3_explicit_worker():
    mock_llm = AsyncMock(spec=LLMClient)

    # Evaluate difficulty: Local heuristic returns 2 -> team size 1
    plan_json = json.dumps([
        {"id": "test_1", "description": "Execute first part of task", "worker": "Coder"}
    ])

    mock_llm.chat_completion.side_effect = [
        plan_json,
        "Coder task done",
        "YES Result complete"
    ]

    orchestrator = MagenticOneOrchestratorV3(llm=mock_llm)
    result = await orchestrator.orchestrate("Do the task")

    assert "YES Result complete" in result
    assert mock_llm.chat_completion.call_count == 3

    exec_call = mock_llm.chat_completion.call_args_list[1]
    assert "WebSurfer" in exec_call[0][0][0]["content"]

@pytest.mark.asyncio
async def test_magentic_one_v3_orchestrator_max_iterations():
    mock_llm = AsyncMock(spec=LLMClient)

    # 3 iterations * 3 LLM calls each = 9 calls total.
    mock_llm.chat_completion.side_effect = [
        '[{"id": "test_1", "description": "Execute"}]', "Execute", "NO",
        '[{"id": "test_1", "description": "Execute"}]', "Execute", "NO",
        '[{"id": "test_1", "description": "Execute"}]', "Execute", "NO"
    ]

    orchestrator = MagenticOneOrchestratorV3(llm=mock_llm)
    result = await orchestrator.orchestrate("Do the task", max_iterations=3)

    assert "Task incomplete after 3 iterations" in result
    assert mock_llm.chat_completion.call_count == 9

@pytest.mark.asyncio
async def test_magentic_one_v3_orchestrator_invalid_json():
    mock_llm = AsyncMock(spec=LLMClient)

    mock_llm.chat_completion.side_effect = [
        'Invalid JSON',
        "Fallback task executed",
        "YES Result complete"
    ]

    orchestrator = MagenticOneOrchestratorV3(llm=mock_llm)
    result = await orchestrator.orchestrate("Do the task")

    assert "YES Result complete" in result
    assert mock_llm.chat_completion.call_count == 3


@pytest.mark.asyncio
async def test_magentic_one_v3_orchestrator_hierarchical_delegation():
    mock_llm = AsyncMock(spec=LLMClient)

    mock_llm.chat_completion.side_effect = [
        '[{"id": "parent_1", "description": "Parent task", "subtasks": [{"id": "child_1", "description": "Child task 1"}, {"id": "child_2", "description": "Child task 2"}]}]',
        "Child 1 done",
        "YES Complete"
    ]

    orchestrator = MagenticOneOrchestratorV3(llm=mock_llm)
    result = await orchestrator.orchestrate("Complex hierarchical task")

    assert "YES Complete" in result

    assert mock_llm.chat_completion.call_count == 3

    exec_call_1 = mock_llm.chat_completion.call_args_list[1]

    assert "WebSurfer" in exec_call_1[0][0][0]["content"]
