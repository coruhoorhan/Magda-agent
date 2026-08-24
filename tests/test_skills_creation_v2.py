import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.learning.skills_creation_v2 import ExperienceSkillCreatorV2
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
async def test_generate_skill_from_logs_success(mock_llm_client: LLMClient, mock_procedural_memory: ProceduralMemory) -> None:
    mock_response = """
Here is the skill:
```python
def extract_ips(logs):
    import re
    return re.findall(r'[0-9]+(?:\.[0-9]+){3}', logs)
```
And the schema:
```json
{
    "name": "extract_ips",
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

    creator = ExperienceSkillCreatorV2(llm_client=mock_llm_client, procedural_memory=mock_procedural_memory)

    raw_logs = "2026-06-15 10:00:00 [INFO] Connection from 192.168.1.1\n2026-06-15 10:05:00 [INFO] Connection from 10.0.0.5"
    result = await creator.generate_skill_from_logs("Extract IPs from log text", raw_logs)

    assert result is not None
    assert "def extract_ips" in result["code"]
    assert result["schema"]["name"] == "extract_ips"
    assert "extract_ips" in creator.created_skills

    mock_procedural_memory.store_procedure.assert_called_once_with(
        name="extract_ips",
        procedure="def extract_ips(logs):\n    import re\n    return re.findall(r'[0-9]+(?:\\.[0-9]+){3}', logs)",
        metadata={
            "source_problem": "Extract IPs from log text",
            "type": "hermes_experience_skill_v4",
            "schema": result["schema"]
        }
    )

@pytest.mark.asyncio
async def test_generate_skill_from_logs_missing_code(mock_llm_client: LLMClient, mock_procedural_memory: ProceduralMemory) -> None:
    mock_response = """
    I generated a schema for you:
    ```json
    {
        "name": "do_something",
        "description": "Does something"
    }
    ```
    """
    mock_llm_client.chat_completion.return_value = mock_response

    creator = ExperienceSkillCreatorV2(llm_client=mock_llm_client, procedural_memory=mock_procedural_memory)
    result = await creator.generate_skill_from_logs("Task", "raw logs")

    assert result is None
    mock_procedural_memory.store_procedure.assert_not_called()

@pytest.mark.asyncio
async def test_generate_skill_from_logs_invalid_json(mock_llm_client: LLMClient, mock_procedural_memory: ProceduralMemory) -> None:
    mock_response = """
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

    creator = ExperienceSkillCreatorV2(llm_client=mock_llm_client, procedural_memory=mock_procedural_memory)
    result = await creator.generate_skill_from_logs("Task", "raw logs")

    assert result is None
    mock_procedural_memory.store_procedure.assert_not_called()

@pytest.mark.asyncio
async def test_generate_skill_from_logs_api_failure(mock_llm_client: LLMClient, mock_procedural_memory: ProceduralMemory) -> None:
    mock_llm_client.chat_completion.side_effect = Exception("API Error")

    creator = ExperienceSkillCreatorV2(llm_client=mock_llm_client, procedural_memory=mock_procedural_memory)
    result = await creator.generate_skill_from_logs("Task", "raw logs")

    assert result is None
    mock_procedural_memory.store_procedure.assert_not_called()
