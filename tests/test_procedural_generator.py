import json
import pytest
from magda_agent.skills.procedural_generator import ProceduralMemoryGenerator

def test_generate_skill_template_empty_logs():
    """Test generating a skill template with empty logs."""
    generator = ProceduralMemoryGenerator()
    result = generator.generate_skill_template([])

    expected = json.dumps({"skill_name": "unknown_skill", "steps": []}, indent=2)
    assert result == expected

def test_generate_skill_template_with_logs():
    """Test generating a skill template with valid logs."""
    generator = ProceduralMemoryGenerator()
    logs = [
        {"action": "read_file", "parameters": {"filepath": "test.txt"}, "result": "content"},
        {"action": "write_file", "parameters": {"filepath": "out.txt", "content": "content"}}
    ]

    result = generator.generate_skill_template(logs)

    expected_dict = {
        "skill_name": "auto_generated_skill",
        "description": "Automatically generated skill from success logs.",
        "steps": [
            {
                "step_index": 1,
                "action": "read_file",
                "parameters": {"filepath": "test.txt"}
            },
            {
                "step_index": 2,
                "action": "write_file",
                "parameters": {"filepath": "out.txt", "content": "content"}
            }
        ]
    }

    assert json.loads(result) == expected_dict

def test_generate_skill_template_missing_keys():
    """Test generating a skill template with logs missing standard keys."""
    generator = ProceduralMemoryGenerator()
    logs = [
        {"other_key": "value"}
    ]

    result = generator.generate_skill_template(logs)

    expected_dict = {
        "skill_name": "auto_generated_skill",
        "description": "Automatically generated skill from success logs.",
        "steps": [
            {
                "step_index": 1,
                "action": "unknown_action",
                "parameters": {}
            }
        ]
    }

    assert json.loads(result) == expected_dict
