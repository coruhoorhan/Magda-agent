import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.learning.skills_creation_v5 import ExperienceSkillCreatorV5
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
async def test_generate_skill_success(mock_llm_client: LLMClient, mock_procedural_memory: ProceduralMemory) -> None:
    mock_response = r"""
Here is the extracted skill:
```python
def extract_ips_from_logs(logs):
    import re
    return re.findall(r'[0-9]+(?:\.[0-9]+){3}', logs)
```
And the schema:
```json
{
    "name": "extract_ips_from_logs",
    "description": "Extracts IP addresses from logs",
    "inputSchema": {
        "type": "object",
        "properties": {
            "logs": {"type": "string"}
        }
    }
}
```
"""
    mock_llm_client.chat_completion.return_value = mock_response

    creator = ExperienceSkillCreatorV5(llm_client=mock_llm_client, procedural_memory=mock_procedural_memory)

    experience_data = "2026-06-15 10:00:00 [INFO] Connection from 192.168.1.1\n2026-06-15 10:05:00 [INFO] Connection from 10.0.0.5"
    result = await creator.generate_skill_from_experience("Extract IPs from log text", experience_data)

    assert result is not None
    assert "def extract_ips_from_logs" in result["code"]
    assert result["schema"]["name"] == "extract_ips_from_logs"
    assert "extract_ips_from_logs" in creator.created_skills

    mock_procedural_memory.store_procedure.assert_called_once_with(
        name="extract_ips_from_logs",
        procedure="def extract_ips_from_logs(logs):\n    import re\n    return re.findall(r'[0-9]+(?:\\.[0-9]+){3}', logs)",
        metadata={
            "source_problem": "Extract IPs from log text",
            "type": "hermes_experience_skill_v5",
            "schema": result["schema"]
        }
    )

@pytest.mark.asyncio
async def test_generate_skill_missing_code(mock_llm_client: LLMClient, mock_procedural_memory: ProceduralMemory) -> None:
    mock_response = r"""
    I generated a schema for you:
    ```json
    {
        "name": "do_something",
        "description": "Does something"
    }
    ```
    """
    mock_llm_client.chat_completion.return_value = mock_response

    creator = ExperienceSkillCreatorV5(llm_client=mock_llm_client, procedural_memory=mock_procedural_memory)
    result = await creator.generate_skill_from_experience("Task", "experience data")

    assert result is None
    mock_procedural_memory.store_procedure.assert_not_called()

@pytest.mark.asyncio
async def test_generate_skill_invalid_schema(mock_llm_client: LLMClient, mock_procedural_memory: ProceduralMemory) -> None:
    mock_response = r"""
    ```python
    def foo(): pass
    ```
    ```json
    {
        "name": "foo",
        "inputSchema": {
            "type": "object"
    ```
    """
    mock_llm_client.chat_completion.return_value = mock_response

    creator = ExperienceSkillCreatorV5(llm_client=mock_llm_client, procedural_memory=mock_procedural_memory)
    result = await creator.generate_skill_from_experience("Task", "experience data")

    assert result is None
    mock_procedural_memory.store_procedure.assert_not_called()

@pytest.mark.asyncio
async def test_generate_skill_api_failure(mock_llm_client: LLMClient, mock_procedural_memory: ProceduralMemory) -> None:
    mock_llm_client.chat_completion.side_effect = Exception("API Error")

    creator = ExperienceSkillCreatorV5(llm_client=mock_llm_client, procedural_memory=mock_procedural_memory)
    result = await creator.generate_skill_from_experience("Task", "experience data")

    assert result is None
    mock_procedural_memory.store_procedure.assert_not_called()
