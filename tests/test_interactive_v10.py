import pytest
from unittest.mock import MagicMock
from magda_agent.learning.interactive_v10 import InteractiveLearnerV10

@pytest.fixture
def mock_mirror_neurons():
    neurons = MagicMock()
    # default to neutral empathy
    neurons.empathize.return_value = (0.0, 0.0, 0.0)
    return neurons

@pytest.fixture
def mock_user_model():
    model = MagicMock()
    # default simple user model data
    model.get_model.return_value = {
        "interaction_weights": {
            "exploration": 1.0,
            "verbosity": 1.0,
            "directness": 1.0,
            "empathy": 1.0
        }
    }
    return model

@pytest.mark.asyncio
async def test_process_interaction_empty_reply(mock_mirror_neurons, mock_user_model):
    learner = InteractiveLearnerV10(mirror_neurons=mock_mirror_neurons, user_model=mock_user_model)
    await learner.process_interaction("", "user_123")

    mock_mirror_neurons.empathize.assert_not_called()
    mock_user_model.get_model.assert_not_called()
    mock_user_model.save_model.assert_not_called()

@pytest.mark.asyncio
async def test_process_interaction_positive_feedback(mock_mirror_neurons, mock_user_model):
    learner = InteractiveLearnerV10(mirror_neurons=mock_mirror_neurons, user_model=mock_user_model)

    # Mock a positive feedback shift
    mock_mirror_neurons.empathize.return_value = (0.5, 0.2, 0.1) # p_shift, a_shift, d_shift

    user_data = {
        "interaction_weights": {
            "exploration": 1.0,
            "verbosity": 1.0,
            "directness": 1.0,
            "empathy": 1.0
        }
    }
    mock_user_model.get_model.return_value = user_data

    await learner.process_interaction("Great job!", "user_123")

    mock_mirror_neurons.empathize.assert_called_with("Great job!")
    mock_user_model.get_model.assert_called_with("user_123")

    # verify save model was called
    assert mock_user_model.save_model.call_count == 1

    # check that values were updated correctly
    saved_data = mock_user_model.save_model.call_args[0][1]

    # (friendly) should be added
    assert "(friendly)" in saved_data["interaction_style"]

    weights = saved_data["interaction_weights"]
    assert weights["exploration"] == 1.0 + (0.5 * 0.3)
    assert weights["verbosity"] == 1.0 + (0.2 * 0.2)
    assert weights["directness"] == 1.0 + (0.1 * 0.2)
    assert weights["empathy"] == 1.0 + (abs(0.5) * 0.1)

    interactions = saved_data["interactions"]
    assert interactions["last_p_shift"] == 0.5
    assert interactions["last_a_shift"] == 0.2
    assert interactions["last_d_shift"] == 0.1

@pytest.mark.asyncio
async def test_process_interaction_negative_feedback(mock_mirror_neurons, mock_user_model):
    learner = InteractiveLearnerV10(mirror_neurons=mock_mirror_neurons, user_model=mock_user_model)

    # Mock a negative feedback shift
    mock_mirror_neurons.empathize.return_value = (-0.4, -0.1, -0.3) # p_shift, a_shift, d_shift

    user_data = {
        "interaction_style": "default",
        "interaction_weights": {
            "exploration": 1.0,
            "verbosity": 1.0,
            "directness": 1.0,
            "empathy": 1.0
        }
    }
    mock_user_model.get_model.return_value = user_data

    await learner.process_interaction("That was wrong.", "user_456")

    mock_mirror_neurons.empathize.assert_called_with("That was wrong.")
    mock_user_model.get_model.assert_called_with("user_456")

    assert mock_user_model.save_model.call_count == 1
    saved_data = mock_user_model.save_model.call_args[0][1]

    # (cautious) should be added
    assert "(cautious)" in saved_data["interaction_style"]

    weights = saved_data["interaction_weights"]
    assert weights["exploration"] == 1.0 + (-0.4 * 0.3)
    assert weights["verbosity"] == 1.0 + (-0.1 * 0.2)
    assert weights["directness"] == 1.0 + (-0.3 * 0.2)
    assert weights["empathy"] == 1.0 + (abs(-0.4) * 0.1)

@pytest.mark.asyncio
async def test_process_interaction_clipping(mock_mirror_neurons, mock_user_model):
    learner = InteractiveLearnerV10(mirror_neurons=mock_mirror_neurons, user_model=mock_user_model)

    # Mock massive feedback shift to test max/min caps (e.g. max is 2.0, min is 0.5)
    mock_mirror_neurons.empathize.return_value = (5.0, -5.0, 5.0)

    user_data = {
        "interaction_weights": {
            "exploration": 1.8,
            "verbosity": 0.6,
            "directness": 1.9,
            "empathy": 1.9
        }
    }
    mock_user_model.get_model.return_value = user_data

    await learner.process_interaction("Extreme!", "user_789")

    saved_data = mock_user_model.save_model.call_args[0][1]
    weights = saved_data["interaction_weights"]

    assert weights["exploration"] == 2.0  # clipped
    assert weights["verbosity"] == 0.5    # clipped
    assert weights["directness"] == 2.0   # clipped
    assert weights["empathy"] == 2.0      # clipped
