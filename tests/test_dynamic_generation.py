import pytest
from unittest.mock import MagicMock
from magda_agent.skills.dynamic_generation import DynamicSkillGenerator, TrajectoryStep

def test_dynamic_skill_generator_valid_code():
    mock_llm_client = MagicMock()
    mock_llm_client.generate.return_value = """```python
def example_skill():
    print("Success")
    return True
```"""

    generator = DynamicSkillGenerator(llm_client=mock_llm_client)
    logs = [TrajectoryStep(action="test_action", result="test_result")]

    code = generator.generate_skill_from_logs(logs)

    assert code is not None
    assert "def example_skill():" in code
    assert generator.is_valid_python(code)
    mock_llm_client.generate.assert_called_once()

def test_dynamic_skill_generator_invalid_code():
    mock_llm_client = MagicMock()
    mock_llm_client.generate.return_value = """```python
def example_skill()
    print("Success")
```""" # Missing colon

    generator = DynamicSkillGenerator(llm_client=mock_llm_client)
    logs = [TrajectoryStep(action="test_action", result="test_result")]

    code = generator.generate_skill_from_logs(logs)

    assert code is None
    mock_llm_client.generate.assert_called_once()

def test_dynamic_skill_generator_empty_logs():
    mock_llm_client = MagicMock()

    generator = DynamicSkillGenerator(llm_client=mock_llm_client)

    code = generator.generate_skill_from_logs([])

    assert code is None
    mock_llm_client.generate.assert_not_called()

def test_dynamic_skill_generator_no_llm_client():
    generator = DynamicSkillGenerator()
    logs = [TrajectoryStep(action="test_action", result="test_result")]

    code = generator.generate_skill_from_logs(logs)

    assert code is None

def test_extract_code_without_markdown():
    mock_llm_client = MagicMock()
    mock_llm_client.generate.return_value = """
def raw_code():
    pass
"""
    generator = DynamicSkillGenerator(llm_client=mock_llm_client)
    logs = [TrajectoryStep(action="test_action", result="test_result")]

    code = generator.generate_skill_from_logs(logs)
    assert code is not None
    assert "def raw_code():" in code

def test_load_skill_to_registry():
    generator = DynamicSkillGenerator()

    code = """
def generated_skill():
    '''This is a generated skill'''
    return "OK"
"""

    mock_registry = MagicMock()

    result = generator.load_skill_to_registry(code, "generated_skill", mock_registry)

    assert result is True
    mock_registry.register_skill.assert_called_once()

    call_kwargs = mock_registry.register_skill.call_args.kwargs
    assert call_kwargs["name"] == "generated_skill"
    assert call_kwargs["description"] == "This is a generated skill"
    assert callable(call_kwargs["func"])
    assert call_kwargs["func"]() == "OK"

def test_load_skill_to_registry_invalid_name():
    generator = DynamicSkillGenerator()

    code = """
def other_name():
    return "OK"
"""

    mock_registry = MagicMock()

    result = generator.load_skill_to_registry(code, "generated_skill", mock_registry)

    assert result is False
    mock_registry.register_skill.assert_not_called()

def test_load_skill_to_registry_execution_error():
    generator = DynamicSkillGenerator()

    code = """
def generated_skill():
    return "OK"

1 / 0  # This will raise ZeroDivisionError during exec
"""

    mock_registry = MagicMock()

    result = generator.load_skill_to_registry(code, "generated_skill", mock_registry)

    assert result is False
    mock_registry.register_skill.assert_not_called()
