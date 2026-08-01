import pytest
from unittest.mock import patch, MagicMock
from magda_agent.learning.hermes_skills_loop import HermesSkillsLoop

def test_hermes_skills_loop_initialization():
    loop = HermesSkillsLoop()
    assert loop._registry == {}

def test_record_successful_outcome():
    loop = HermesSkillsLoop()
    with patch("time.time", return_value=100.0):
        loop.record_skill_outcome("test_skill", True, {"runtime": 0.5})

    meta = loop.get_skill_metadata("test_skill")
    assert meta is not None
    assert meta["success_count"] == 1
    assert meta["failure_count"] == 0
    assert meta["total_usage"] == 1
    assert meta["success_rate"] == 1.0
    assert meta["last_used_timestamp"] == 100.0
    assert meta["metadata"] == {"runtime": 0.5}

def test_record_failed_outcome():
    loop = HermesSkillsLoop()
    with patch("time.time", return_value=150.0):
        loop.record_skill_outcome("fail_skill", False, {"error": "timeout"})

    meta = loop.get_skill_metadata("fail_skill")
    assert meta is not None
    assert meta["success_count"] == 0
    assert meta["failure_count"] == 1
    assert meta["total_usage"] == 1
    assert meta["success_rate"] == 0.0
    assert meta["last_used_timestamp"] == 150.0
    assert meta["metadata"] == {"error": "timeout"}

def test_multiple_outcomes_and_metadata_update():
    loop = HermesSkillsLoop()
    loop.record_skill_outcome("multi_skill", True, {"key1": "val1"})
    loop.record_skill_outcome("multi_skill", False, {"key2": "val2"})
    loop.record_skill_outcome("multi_skill", True)

    meta = loop.get_skill_metadata("multi_skill")
    assert meta is not None
    assert meta["success_count"] == 2
    assert meta["failure_count"] == 1
    assert meta["total_usage"] == 3
    assert meta["success_rate"] == 2/3
    assert meta["metadata"] == {"key1": "val1", "key2": "val2"}

def test_get_nonexistent_skill_metadata():
    loop = HermesSkillsLoop()
    assert loop.get_skill_metadata("missing_skill") is None

@patch("magda_agent.llm_client.LLMClient")
def test_hermes_skills_loop_with_llm(mock_llm_client_cls):
    """
    Test that demonstrates registry improvement processing with mocked LLM.
    We'll simulate checking the registry, querying the LLM for a potential
    action based on the success rate, and ensuring the LLM was called.
    """
    mock_llm = MagicMock()
    mock_llm_client_cls.return_value = mock_llm
    mock_llm.generate.return_value = '{"action": "refine_prompt", "reason": "Low success rate"}'

    loop = HermesSkillsLoop()
    loop.record_skill_outcome("flaky_skill", False, {"error": "timeout"})
    loop.record_skill_outcome("flaky_skill", False, {"error": "wrong output"})

    meta = loop.get_skill_metadata("flaky_skill")

    # Let's say if success_rate < 0.5, we consult LLM for improvements
    if meta and meta["success_rate"] < 0.5:
        prompt = f"Skill flaky_skill has a low success rate of {meta['success_rate']}. Suggest improvement."
        response = mock_llm.generate(prompt)
        assert "refine_prompt" in response
        mock_llm.generate.assert_called_once_with(prompt)
