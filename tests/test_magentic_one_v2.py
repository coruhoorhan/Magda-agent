import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from magda_agent.llm_client import LLMClient
from magda_agent.architecture.magentic_one_v2 import MagenticOneOrchestratorV2

class MockLLMClient(LLMClient):
    def __init__(self):
        self.chat_completion = AsyncMock()

def test_evaluate_difficulty():
    llm = MockLLMClient()
    orchestrator = MagenticOneOrchestratorV2(llm)

    assert orchestrator._evaluate_difficulty("short task") == 2
    assert orchestrator._evaluate_difficulty("this is a slightly longer task for testing") == 5
    assert orchestrator._evaluate_difficulty("this is an even longer task that should evaluate to a higher difficulty score for dynamic scaling.") == 8
    assert orchestrator._evaluate_difficulty("this is a very long task " * 10) == 10

def test_calculate_team_size():
    llm = MockLLMClient()
    orchestrator = MagenticOneOrchestratorV2(llm)

    assert orchestrator._calculate_team_size(2) == 1
    assert orchestrator._calculate_team_size(5) == 3
    assert orchestrator._calculate_team_size(8) == 4
    assert orchestrator._calculate_team_size(10) == 5

def test_spawn_and_kill_workers():
    llm = MockLLMClient()
    orchestrator = MagenticOneOrchestratorV2(llm)

    assert len(orchestrator.active_workers) == 0

    orchestrator._spawn_workers(3)
    assert len(orchestrator.active_workers) == 3
    assert orchestrator.active_workers[0].name == "WebSurfer"
    assert orchestrator.active_workers[1].name == "FileSurfer"
    assert orchestrator.active_workers[2].name == "Coder"

    orchestrator._spawn_workers(5)
    assert len(orchestrator.active_workers) == 5
    assert orchestrator.active_workers[4].name == "DynamicWorker_4"

    orchestrator._kill_workers()
    assert len(orchestrator.active_workers) == 0

def test_orchestrate():
    llm = MockLLMClient()

    # Mock planning to return JSON
    plan_json = '[{"id": "t1", "description": "do step 1"}]'
    llm.chat_completion.side_effect = [
        plan_json, # plan
        "result 1", # execute
        "YES, task is complete" # review
    ]

    orchestrator = MagenticOneOrchestratorV2(llm)

    # Use asyncio.run since it's an async test without pytest-asyncio wrapper
    result = asyncio.run(orchestrator.orchestrate("short task"))

    assert "YES, task is complete" in result
    assert len(orchestrator.active_workers) == 0 # verify cleanup

def test_orchestrate_incomplete():
    llm = MockLLMClient()

    plan_json = '[{"id": "t1", "description": "do step 1"}]'
    # Return "NO" for review for 3 iterations
    llm.chat_completion.side_effect = [
        plan_json, "r1", "NO",
        plan_json, "r2", "NO",
        plan_json, "r3", "NO"
    ]

    orchestrator = MagenticOneOrchestratorV2(llm)

    result = asyncio.run(orchestrator.orchestrate("short task", max_iterations=3))

    assert "Task incomplete" in result
    assert len(orchestrator.active_workers) == 0
