import pytest
from unittest.mock import MagicMock
from magda_agent.agents.subagent_factory import SubagentFactory
from magda_agent.agents.sub_agent import SubAgent
from magda_agent.llm_client import LLMClient

def test_spawn_planner_isolated():
    """
    Test spawning a Planner subagent with isolated context properly configured.
    """
    mock_llm = MagicMock(spec=LLMClient)

    agent = SubagentFactory.spawn_subagent(
        role="Planner",
        llm=mock_llm
    )

    assert isinstance(agent, SubAgent)
    assert agent.llm == mock_llm
    assert agent.use_isolation is True
    assert "Planner Sub-Agent" in agent.system_prompt
    assert agent.worktree_manager is not None

def test_spawn_evaluator_isolated():
    """
    Test spawning an Evaluator subagent with isolated context properly configured.
    """
    mock_llm = MagicMock(spec=LLMClient)

    agent = SubagentFactory.spawn_subagent(
        role="Evaluator",
        llm=mock_llm
    )

    assert isinstance(agent, SubAgent)
    assert agent.llm == mock_llm
    assert agent.use_isolation is True
    assert "Evaluator Sub-Agent" in agent.system_prompt
    assert agent.worktree_manager is not None

def test_spawn_custom_role():
    """
    Test spawning a subagent with a custom role and custom system prompt.
    """
    mock_llm = MagicMock(spec=LLMClient)
    custom_prompt = "You are a custom CodeReviewer Sub-Agent."

    agent = SubagentFactory.spawn_subagent(
        role="CodeReviewer",
        llm=mock_llm,
        system_prompt=custom_prompt,
        use_isolation=False
    )

    assert isinstance(agent, SubAgent)
    assert agent.system_prompt == custom_prompt
    assert agent.use_isolation is False
    assert agent.worktree_manager is None
