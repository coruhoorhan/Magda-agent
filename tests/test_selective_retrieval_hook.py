import pytest
from unittest.mock import MagicMock
from magda_agent.memory.selective_retrieval_hook import SelectiveRetrievalHook
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState

def test_selective_retrieval_hook_after_retrieval():
    """
    Tests that SelectiveRetrievalHook's after_retrieval method correctly calls
    SelectiveMemoryRetriever and appends the returned episodes as MemoryEntry objects.
    """
    # Mock the retriever
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_episodes.return_value = ["episode 1", "episode 2"]

    # Initialize the hook
    hook = SelectiveRetrievalHook(retriever=mock_retriever, top_k=3)

    # Initial context (e.g., from working memory)
    pad_state = PADState(0.0, 0.0, 0.0)
    initial_context = [
        MemoryEntry(content="working memory item 1", importance=0.8, emotional_state=pad_state, tags=["wm"], user_id=42)
    ]

    # Call the hook
    query = "test task query"
    user_id = 42
    updated_context = hook.after_retrieval(context=initial_context, query=query, user_id=user_id)

    # Assert retriever was called correctly
    mock_retriever.get_relevant_episodes.assert_called_once_with(current_task=query, user_id=user_id, top_k=3)

    # Assert context is properly updated
    assert len(updated_context) == 3
    assert updated_context[0].content == "working memory item 1"

    # Assert episodes were appended as MemoryEntry objects
    assert updated_context[1].content == "episode 1"
    assert "episodic_recall" in updated_context[1].tags
    assert updated_context[1].user_id == user_id

    assert updated_context[2].content == "episode 2"
    assert "episodic_recall" in updated_context[2].tags
    assert updated_context[2].user_id == user_id
