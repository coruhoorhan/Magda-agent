import pytest
from unittest.mock import AsyncMock, MagicMock

from magda_agent.learning.skill_refinement_v8 import HermesSkillRefinerV8
from magda_agent.memory.procedural import ProceduralMemory


@pytest.fixture
def procedural_memory():
    return ProceduralMemory(persist_directory=":memory:")


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="""```python
def process_data(data):
    \"\"\"Optimized process_data function.\"\"\"
    return [x * 2 for x in data if x > 0]
```""")
    return llm


def test_record_usage_and_candidates(procedural_memory):
    refiner = HermesSkillRefinerV8(
        procedural_memory=procedural_memory,
        min_usage_threshold=3,
        min_success_rate=0.6
    )

    # Initial state
    assert len(refiner.get_refinement_candidates()) == 0

    # Record 2 usages (below threshold)
    refiner.record_usage("data_pipeline", success=True)
    refiner.record_usage("data_pipeline", success=True)
    assert len(refiner.get_refinement_candidates()) == 0

    # 3rd usage -> total 3 usages, 100% success rate
    refiner.record_usage("data_pipeline", success=True)
    candidates = refiner.get_refinement_candidates()
    assert len(candidates) == 1
    assert candidates[0][0] == "data_pipeline"
    assert candidates[0][1]["usage_count"] == 3
    assert candidates[0][1]["success_rate"] == 1.0

    # Record failure skill
    refiner.record_usage("failing_skill", success=False)
    refiner.record_usage("failing_skill", success=False)
    refiner.record_usage("failing_skill", success=False)
    candidates = refiner.get_refinement_candidates()
    assert len(candidates) == 1  # failing_skill has 0% success rate < 0.6 threshold


@pytest.mark.asyncio
async def test_propose_optimized_skill_with_llm(procedural_memory, mock_llm):
    refiner = HermesSkillRefinerV8(
        procedural_memory=procedural_memory,
        llm_client=mock_llm
    )

    original_code = "def process_data(data):\n    res = []\n    for x in data:\n        res.append(x * 2)\n    return res"
    proposed = await refiner.propose_optimized_skill("process_data", original_code)

    assert proposed.startswith("def process_data(data):")
    assert "[x * 2 for x in data if x > 0]" in proposed
    mock_llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_propose_optimized_skill_fallback(procedural_memory):
    refiner = HermesSkillRefinerV8(procedural_memory=procedural_memory, llm_client=None)

    original_code = "def calculate_total(items):\n    return sum(items)"
    proposed = await refiner.propose_optimized_skill("calculate_total", original_code)

    assert "def calculate_total(items):" in proposed
    assert "Hermes Refine v8: Optimized" in proposed


@pytest.mark.asyncio
async def test_refine_skill_integration(procedural_memory, mock_llm):
    # Store initial skill procedure
    procedural_memory.store_procedure(
        name="process_data",
        procedure="def process_data(data):\n    return [x for x in data]",
        metadata={"version": 1}
    )

    refiner = HermesSkillRefinerV8(
        procedural_memory=procedural_memory,
        llm_client=mock_llm,
        min_usage_threshold=2
    )

    refiner.record_usage("process_data", success=True)
    refiner.record_usage("process_data", success=True)

    refined = await refiner.refine_skill("process_data")
    assert refined is not None
    assert "def process_data" in refined

    # Check updated procedure versions in ProceduralMemory
    versions = procedural_memory.get_procedure_versions("process_data")
    assert len(versions["documents"]) == 2
    assert versions["metadatas"][1]["version"] == 2
    assert versions["metadatas"][1]["type"] == "hermes_skill_refinement_v8"
