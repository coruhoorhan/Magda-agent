import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from magda_agent.architecture.triad_coordinator import TriadCoordinator

@pytest.mark.asyncio
async def test_coordinate_flow():
    # Setup mocks
    planner_agent = MagicMock()
    planner_agent.plan = AsyncMock()

    generator_agent = MagicMock()
    generator_agent.execute_plan = AsyncMock(return_value="executed_plan")
    generator_agent.generate_response = AsyncMock(return_value="response")

    evaluator_agent = MagicMock()
    evaluator_agent.evaluate = AsyncMock()

    coordinator = TriadCoordinator(planner_agent, generator_agent, evaluator_agent)

    # Call the method
    user_input = "test input"
    user_id = "test_user"
    policies = ["policy1"]
    mental_state = {"state": "happy"}
    behavior_weights = {"b1": 1.0}
    skill_weights = {"s1": 0.5}

    response = await coordinator.coordinate(
        user_input,
        user_id=user_id,
        policies=policies,
        mental_state=mental_state,
        behavior_weights=behavior_weights,
        skill_weights=skill_weights
    )

    assert response == "response"

    # Assert sequence of calls
    planner_agent.plan.assert_called_once_with(
        user_input,
        user_id=user_id,
        mental_state=mental_state,
        behavior_weights=behavior_weights,
        skill_weights=skill_weights
    )
    generator_agent.execute_plan.assert_called_once_with(user_input, user_id=user_id)
    generator_agent.generate_response.assert_called_once_with([])
    evaluator_agent.evaluate.assert_called_once_with(user_input, "response", user_id=user_id, policies=policies)


@pytest.mark.asyncio
async def test_coordinate_with_hooks():
    planner_agent = MagicMock()
    planner_agent.plan = AsyncMock()

    generator_agent = MagicMock()
    generator_agent.execute_plan = AsyncMock(return_value="executed_plan_string")
    generator_agent.generate_response = AsyncMock(return_value="response_string")

    evaluator_agent = MagicMock()
    evaluator_agent.evaluate = AsyncMock()

    coordinator = TriadCoordinator(planner_agent, generator_agent, evaluator_agent)

    message_builder = MagicMock(return_value=[{"role": "user", "content": "hello"}])
    pre_generation_hook = MagicMock()

    response = await coordinator.coordinate(
        "input",
        message_builder=message_builder,
        pre_generation_hook=pre_generation_hook
    )

    assert response == "response_string"
    message_builder.assert_called_once_with("executed_plan_string")
    pre_generation_hook.assert_called_once()
    generator_agent.generate_response.assert_called_once_with([{"role": "user", "content": "hello"}])
