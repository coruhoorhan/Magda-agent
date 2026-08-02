import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.skills.dynamic_creator import DynamicSkillCreator
from magda_agent.memory.procedural import ProceduralMemory
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_llm_client() -> LLMClient:
    client = MagicMock(spec=LLMClient)
    client.chat_completion = AsyncMock()
    return client

@pytest.fixture
def mock_procedural_memory() -> ProceduralMemory:
    mem = MagicMock(spec=ProceduralMemory)
    mem.store_procedure = MagicMock()
    return mem

@pytest.mark.asyncio
async def test_parse_and_create_skill_success(mock_llm_client: LLMClient, mock_procedural_memory: ProceduralMemory) -> None:
    mock_response = """
Here is a repeatable skill.
```python
def format_phone_number(number):
    return f"+1-{number[:3]}-{number[3:6]}-{number[6:]}"
```

Here is the JSON schema:
```json
{
    "name": "format_phone_number",
    "description": "Formats a 10-digit number into US format",
    "parameters": {
        "type": "object",
        "properties": {
            "number": {"type": "string"}
        },
        "required": ["number"]
    }
}
```
"""
    mock_llm_client.chat_completion.return_value = mock_response

    creator = DynamicSkillCreator(llm_client=mock_llm_client, procedural_memory=mock_procedural_memory)
    result = await creator.parse_and_create_skill("Can you create a skill to format phone numbers?", user_id=456)

    assert result is not None
    assert result["name"] == "format_phone_number"
    assert "def format_phone_number" in result["code"]
    assert result["description"] == "Formats a 10-digit number into US format"
    assert "format_phone_number" in creator.created_skills

    mock_procedural_memory.store_procedure.assert_called_once_with(
        name="format_phone_number",
        procedure="def format_phone_number(number):\n    return f\"+1-{number[:3]}-{number[3:6]}-{number[6:]}\"",
        user_id=456,
        metadata={
            "source_request": "Can you create a skill to format phone numbers?",
            "type": "dynamic_conversational_skill_v1",
            "schema": result["schema"],
            "description": "Formats a 10-digit number into US format"
        }
    )

@pytest.mark.asyncio
async def test_parse_and_create_skill_no_pattern(mock_llm_client: LLMClient, mock_procedural_memory: ProceduralMemory) -> None:
    mock_llm_client.chat_completion.return_value = "NO_PATTERN"

    creator = DynamicSkillCreator(llm_client=mock_llm_client, procedural_memory=mock_procedural_memory)
    result = await creator.parse_and_create_skill("Hello, just saying hi!", user_id=456)

    assert result is None
    mock_procedural_memory.store_procedure.assert_not_called()

@pytest.mark.asyncio
async def test_parse_and_create_skill_fallback_name(mock_llm_client: LLMClient, mock_procedural_memory: ProceduralMemory) -> None:
    # No JSON, only Python code
    mock_response = """
```python
def cleanup_data(data):
    return data.strip()
```
"""
    mock_llm_client.chat_completion.return_value = mock_response

    creator = DynamicSkillCreator(llm_client=mock_llm_client, procedural_memory=mock_procedural_memory)
    result = await creator.parse_and_create_skill("Clean up my data", user_id=123)

    assert result is not None
    assert result["name"] == "cleanup_data"
    assert "def cleanup_data" in result["code"]
    assert "cleanup_data" in creator.created_skills

    mock_procedural_memory.store_procedure.assert_called_once_with(
        name="cleanup_data",
        procedure="def cleanup_data(data):\n    return data.strip()",
        user_id=123,
        metadata={
            "source_request": "Clean up my data",
            "type": "dynamic_conversational_skill_v1",
            "schema": {},
            "description": "Dynamically created skill from user request."
        }
    )
