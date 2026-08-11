import asyncio
import pytest
from unittest.mock import AsyncMock

from magda_agent.planning.planner import PlanStep, TypedPlan
from magda_agent.execution.parallel_planner import ParallelPlanner

@pytest.fixture
def mock_executor():
    """Mock execution backend to trace which skills are called and simulate delays."""
    async def executor(skill_name, kwargs):
        if skill_name == "fast_skill":
            return kwargs.get("val", "fast")
        elif skill_name == "slow_skill":
            await asyncio.sleep(0.1)
            return kwargs.get("val", "slow")
        elif skill_name == "error_skill":
            raise ValueError(f"Error triggered for {kwargs.get('val')}")
        return "default"
    return executor

@pytest.fixture
def planner(mock_executor):
    return ParallelPlanner(skill_executor_func=mock_executor)

@pytest.mark.asyncio
async def test_parse_intent(planner):
    """Test that parse_intent returns the intended execution plan, cleaning up invalid dependencies."""
    plan = TypedPlan(
        goal="Test parsing",
        risk="low",
        steps=[
            PlanStep(id="1", description="Step 1", skill="fast_skill", dependencies=[]),
            PlanStep(id="2", description="Step 2", skill="fast_skill", dependencies=["1", "invalid_dep"]),
        ]
    )

    result = planner.parse_intent(plan)

    assert len(result) == 2
    assert result[0].id == "1"
    assert result[1].id == "2"
    # invalid_dep should be stripped
    assert result[1].dependencies == ["1"]

@pytest.mark.asyncio
async def test_execute_plan_concurrent(planner):
    """Test that execute_plan runs independent tasks concurrently."""
    steps = [
        PlanStep(id="s1", description="Slow 1", skill="slow_skill", skill_kwargs={"val": 1}),
        PlanStep(id="s2", description="Slow 2", skill="slow_skill", skill_kwargs={"val": 2}),
        PlanStep(id="s3", description="Slow 3", skill="slow_skill", skill_kwargs={"val": 3}),
    ]

    start_time = asyncio.get_event_loop().time()
    results = await planner.execute_plan(steps)
    end_time = asyncio.get_event_loop().time()

    duration = end_time - start_time

    # Should take around 0.1s instead of 0.3s
    assert duration < 0.2
    assert results == {"s1": 1, "s2": 2, "s3": 3}

@pytest.mark.asyncio
async def test_execute_plan_dependencies(planner):
    """Test that execute_plan respects dependencies."""
    # We will trace the execution order by appending to a list
    execution_order = []

    async def tracking_executor(skill_name, kwargs):
        val = kwargs.get("val")
        if skill_name == "slow_skill":
            await asyncio.sleep(0.05)
        execution_order.append(val)
        return val

    tracking_planner = ParallelPlanner(skill_executor_func=tracking_executor)

    # s1 is slow but has no dependencies.
    # s2 is fast but depends on s1.
    # s3 is fast, no dependencies, should run alongside s1 and finish before s2.
    steps = [
        PlanStep(id="s1", description="Slow dep", skill="slow_skill", skill_kwargs={"val": "s1"}),
        PlanStep(id="s2", description="Fast dep on s1", skill="fast_skill", skill_kwargs={"val": "s2"}, dependencies=["s1"]),
        PlanStep(id="s3", description="Fast no dep", skill="fast_skill", skill_kwargs={"val": "s3"}),
    ]

    results = await tracking_planner.execute_plan(steps)

    assert results == {"s1": "s1", "s2": "s2", "s3": "s3"}
    # s3 should finish before s1 and s2 because s1 is slow and s2 waits for s1.
    assert execution_order.index("s3") < execution_order.index("s1")
    # s2 must finish after s1
    assert execution_order.index("s1") < execution_order.index("s2")

@pytest.mark.asyncio
async def test_execute_plan_with_exception(planner):
    """Test that execute_plan handles exceptions and cascades failures to dependents."""
    steps = [
        PlanStep(id="ok_step", description="OK", skill="fast_skill", skill_kwargs={"val": "ok"}),
        PlanStep(id="err_step", description="Error", skill="error_skill", skill_kwargs={"val": "err"}),
        PlanStep(id="dep_step", description="Depends on Error", skill="fast_skill", skill_kwargs={"val": "dep"}, dependencies=["err_step"]),
    ]

    results = await planner.execute_plan(steps)

    assert results["ok_step"] == "ok"
    assert isinstance(results["err_step"], ValueError)
    assert "Error triggered for err" in str(results["err_step"])

    # dep_step should also fail because its dependency failed
    assert isinstance(results["dep_step"], Exception)
    assert "Dependency err_step failed" in str(results["dep_step"])
