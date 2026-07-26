import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from magda_agent.skills.experience_generator_v2 import ExperienceGeneratorV2

class MockLLMClient:
    async def generate_text(self, prompt: str) -> str:
        """Mock text generation."""
        pass

@pytest.fixture
def mock_llm_client():
    """Fixture to provide a mock LLM client."""
    client = MockLLMClient()
    client.generate_text = AsyncMock(return_value="""
class WeatherSkill:
    def __init__(self):
        pass
    def execute(self, location: str) -> str:
        return f"Weather in {location} is sunny."
""")
    return client

@pytest.fixture
def experience_generator(mock_llm_client):
    """Fixture to provide an ExperienceGeneratorV2 instance."""
    return ExperienceGeneratorV2(llm_client=mock_llm_client)

@pytest.mark.asyncio
async def test_generate_skill_from_traces_success(experience_generator, mock_llm_client):
    """Test successful generation of a skill from traces."""
    traces = [
        {"step_name": "parse_input", "input": "get weather for London", "output": "London", "status": "success"},
        {"step_name": "fetch_weather", "input": "London", "output": "sunny", "status": "success"},
        {"step_name": "format_output", "input": "sunny", "output": "Weather in London is sunny.", "status": "success"}
    ]

    skill_code = await experience_generator.generate_skill_from_traces(
        traces=traces,
        skill_name="WeatherSkill",
        description="Fetches weather for a location."
    )

    assert skill_code is not None
    assert "class WeatherSkill:" in skill_code
    assert "def execute(" in skill_code
    mock_llm_client.generate_text.assert_called_once()

    # Verify prompt formatting
    call_args = mock_llm_client.generate_text.call_args[0][0]
    assert "Skill Name: WeatherSkill" in call_args
    assert "Execution Traces:" in call_args
    assert "Step 1: parse_input" in call_args

@pytest.mark.asyncio
async def test_generate_skill_from_traces_empty_traces(experience_generator, mock_llm_client):
    """Test generation fails with empty traces."""
    skill_code = await experience_generator.generate_skill_from_traces(
        traces=[],
        skill_name="EmptySkill",
        description="Test empty"
    )

    assert skill_code is None
    mock_llm_client.generate_text.assert_not_called()

@pytest.mark.asyncio
async def test_generate_skill_from_traces_unsuccessful_outcome(experience_generator, mock_llm_client):
    """Test generation fails if traces indicate failure."""
    traces = [
        {"step_name": "parse_input", "input": "bad input", "output": "error", "status": "failed"}
    ]

    skill_code = await experience_generator.generate_skill_from_traces(
        traces=traces,
        skill_name="FailedSkill",
        description="Test failed"
    )

    assert skill_code is None
    mock_llm_client.generate_text.assert_not_called()

@pytest.mark.asyncio
async def test_generate_skill_from_traces_handles_markdown_blocks(experience_generator, mock_llm_client):
    """Test generation correctly strips markdown formatting."""
    mock_llm_client.generate_text = AsyncMock(return_value="""```python
class CleanSkill:
    def execute(self):
        return True
```""")

    traces = [{"step_name": "test", "input": "test", "output": "test", "status": "success"}]
    skill_code = await experience_generator.generate_skill_from_traces(
        traces=traces,
        skill_name="CleanSkill",
        description="Test cleanup"
    )

    assert skill_code is not None
    assert skill_code.startswith("class CleanSkill:")
    assert not skill_code.endswith("```")
