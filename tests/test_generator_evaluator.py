"""
Tests for the Generator/Evaluator Subagent Spawning module.
"""

import pytest
from unittest.mock import AsyncMock

from magda_agent.architecture.generator_evaluator import PairedSubagentSpawner

class AsyncExecutor:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    async def execute(self, context):
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

@pytest.mark.asyncio
async def test_spawn_paired_agents_success_first_attempt():
    spawner = PairedSubagentSpawner()

    generator_executor = AsyncExecutor(["Proposed Code v1"])
    evaluator_executor = AsyncExecutor([{"approved": True, "feedback": "Looks good."}])

    result = await spawner.spawn_paired_agents(
        task_description="Write a hello world function",
        full_context=[{"role": "system", "content": "System"}],
        generator_executor=generator_executor,
        evaluator_executor=evaluator_executor,
        max_retries=3
    )

    assert result["status"] == "success"
    assert result["attempts"] == 1
    assert result["proposal"] == "Proposed Code v1"

    assert generator_executor.call_count == 1
    assert evaluator_executor.call_count == 1

@pytest.mark.asyncio
async def test_spawn_paired_agents_success_after_retries():
    spawner = PairedSubagentSpawner()

    # Generator returns different proposals on each call
    generator_executor = AsyncExecutor(["Proposed Code v1", "Proposed Code v2"])

    # Evaluator rejects first, approves second
    evaluator_executor = AsyncExecutor([
        {"approved": False, "feedback": "Missing type hints."},
        {"approved": True, "feedback": "Perfect."}
    ])

    result = await spawner.spawn_paired_agents(
        task_description="Write a hello world function",
        full_context=[{"role": "system", "content": "System"}],
        generator_executor=generator_executor,
        evaluator_executor=evaluator_executor,
        max_retries=3
    )

    assert result["status"] == "success"
    assert result["attempts"] == 2
    assert result["proposal"] == "Proposed Code v2"

    assert generator_executor.call_count == 2
    assert evaluator_executor.call_count == 2

@pytest.mark.asyncio
async def test_spawn_paired_agents_failure_exceeds_retries():
    spawner = PairedSubagentSpawner()

    generator_executor = AsyncExecutor(["Flawed Code", "Flawed Code", "Flawed Code"])

    # Evaluator always rejects
    evaluator_executor = AsyncExecutor([
        {"approved": False, "feedback": "Still wrong."},
        {"approved": False, "feedback": "Still wrong."},
        {"approved": False, "feedback": "Still wrong."}
    ])

    result = await spawner.spawn_paired_agents(
        task_description="Write a hello world function",
        full_context=[{"role": "system", "content": "System"}],
        generator_executor=generator_executor,
        evaluator_executor=evaluator_executor,
        max_retries=2
    )

    assert result["status"] == "failed"
    assert result["attempts"] == 2
    assert result["last_feedback"] == "Still wrong."

    assert generator_executor.call_count == 2
    assert evaluator_executor.call_count == 2

@pytest.mark.asyncio
async def test_spawn_paired_agents_invalid_executors():
    spawner = PairedSubagentSpawner()

    generator_executor = "not a callable"
    evaluator_executor = AsyncMock()

    with pytest.raises(TypeError):
        await spawner.spawn_paired_agents(
            task_description="Write a hello world function",
            full_context=[],
            generator_executor=generator_executor,
            evaluator_executor=evaluator_executor
        )
