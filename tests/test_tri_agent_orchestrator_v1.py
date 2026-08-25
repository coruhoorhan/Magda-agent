import pytest
from unittest.mock import AsyncMock, MagicMock

from magda_agent.architecture.tri_agent_orchestrator_v1 import (
    PlannerComponent,
    GeneratorComponent,
    EvaluatorComponent,
    TriAgentOrchestratorV1,
)


@pytest.mark.asyncio
async def test_default_components_execution():
    orchestrator = TriAgentOrchestratorV1()
    res = await orchestrator.execute_task("Simple task")

    assert res["status"] == "success"
    assert res["completed_steps"] == ["step_1"]
    assert "step_1" in res["step_outputs"]
    assert res["step_outputs"]["step_1"] == "Completed step: Simple task"


@pytest.mark.asyncio
async def test_multi_step_dag_execution():
    async def mock_plan(task_description, context):
        return [
            {"id": "step_1", "description": "Fetch data", "dependencies": []},
            {"id": "step_2", "description": "Process data", "dependencies": ["step_1"]},
            {"id": "step_3", "description": "Format output", "dependencies": ["step_2"]},
        ]

    executed_steps = []

    async def mock_execute(step, context, feedback=None):
        executed_steps.append(step["id"])
        # Check context propagation
        if step["id"] == "step_2":
            assert "step_1" in context["completed_outputs"]
        return f"Output of {step['id']}"

    async def mock_evaluate(step, proposal, context):
        return {"approved": True, "feedback": "Good job"}

    planner = PlannerComponent(plan_fn=mock_plan)
    generator = GeneratorComponent(execute_fn=mock_execute)
    evaluator = EvaluatorComponent(evaluate_fn=mock_evaluate)

    orchestrator = TriAgentOrchestratorV1(planner=planner, generator=generator, evaluator=evaluator)
    res = await orchestrator.execute_task("Pipeline task")

    assert res["status"] == "success"
    assert res["completed_steps"] == ["step_1", "step_2", "step_3"]
    assert executed_steps == ["step_1", "step_2", "step_3"]
    assert res["step_outputs"]["step_3"] == "Output of step_3"


@pytest.mark.asyncio
async def test_evaluator_rejection_and_retry():
    attempts = {"count": 0}
    received_feedbacks = []

    async def mock_execute(step, context, feedback=None):
        attempts["count"] += 1
        received_feedbacks.append(feedback)
        if attempts["count"] == 1:
            return "Initial faulty output"
        return "Fixed output"

    async def mock_evaluate(step, proposal, context):
        if proposal == "Initial faulty output":
            return {"approved": False, "feedback": "Needs fix: syntax error"}
        return {"approved": True, "feedback": "Approved"}

    generator = GeneratorComponent(execute_fn=mock_execute)
    evaluator = EvaluatorComponent(evaluate_fn=mock_evaluate)

    orchestrator = TriAgentOrchestratorV1(generator=generator, evaluator=evaluator, max_retries=3)
    res = await orchestrator.execute_task("Task needing revision")

    assert res["status"] == "success"
    assert attempts["count"] == 2
    assert received_feedbacks == [None, "Needs fix: syntax error"]
    assert res["step_outputs"]["step_1"] == "Fixed output"


@pytest.mark.asyncio
async def test_evaluator_max_retries_exceeded():
    async def mock_evaluate(step, proposal, context):
        return {"approved": False, "feedback": "Always rejected"}

    evaluator = EvaluatorComponent(evaluate_fn=mock_evaluate)
    orchestrator = TriAgentOrchestratorV1(evaluator=evaluator, max_retries=2)
    res = await orchestrator.execute_task("Failing task")

    assert res["status"] == "failed"
    assert "failed evaluation after 2 attempts" in res["error"]
    assert res["failed_step"] == "step_1"
    assert res["last_feedback"] == "Always rejected"


@pytest.mark.asyncio
async def test_cyclic_plan_detection():
    async def mock_plan(task_description, context):
        return [
            {"id": "step_A", "description": "Step A", "dependencies": ["step_B"]},
            {"id": "step_B", "description": "Step B", "dependencies": ["step_A"]},
        ]

    planner = PlannerComponent(plan_fn=mock_plan)
    orchestrator = TriAgentOrchestratorV1(planner=planner)
    res = await orchestrator.execute_task("Cyclic task")

    assert res["status"] == "failed"
    assert "Cycle detected" in res["error"]


@pytest.mark.asyncio
async def test_empty_plan_handling():
    async def mock_plan(task_description, context):
        return []

    planner = PlannerComponent(plan_fn=mock_plan)
    orchestrator = TriAgentOrchestratorV1(planner=planner)
    res = await orchestrator.execute_task("Empty task")

    assert res["status"] == "failed"
    assert "empty plan" in res["error"]
