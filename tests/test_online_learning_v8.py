import pytest
from unittest.mock import MagicMock
from magda_agent.learning.online_learning_v8 import OpenClawOnlineLearningV8
from magda_agent.emotions.mirror_neurons import MirrorNeurons

@pytest.fixture
def mock_mirror_neurons():
    return MagicMock(spec=MirrorNeurons)

def test_initialization():
    learner = OpenClawOnlineLearningV8(initial_weights={"skill_a": 1.2})
    assert learner.get_skill_weight("skill_a") == 1.2
    assert learner.get_skill_weight("skill_b") == 1.0  # default
    assert learner.get_all_weights() == {"skill_a": 1.2}

def test_extract_reward_empty_and_neutral(mock_mirror_neurons):
    learner = OpenClawOnlineLearningV8(mirror_neurons=mock_mirror_neurons)

    # Test empty user reply
    assert learner.extract_reward("") == 0.0

    # Test neutral
    mock_mirror_neurons.empathize.return_value = (0.0, 0.0, 0.0)
    assert learner.extract_reward("hello world") == 0.0

def test_extract_reward_positive_and_negative(mock_mirror_neurons):
    learner = OpenClawOnlineLearningV8(mirror_neurons=mock_mirror_neurons)

    # Positive emotion shift
    mock_mirror_neurons.empathize.return_value = (0.3, 0.1, 0.0)
    reward = learner.extract_reward("I am happy")
    assert reward == 0.3

    # Negative override matching word
    mock_mirror_neurons.empathize.return_value = (0.0, 0.0, 0.0)
    reward = learner.extract_reward("That is wrong and terrible")
    assert reward == -1.0

    # Positive override matching word
    mock_mirror_neurons.empathize.return_value = (0.0, 0.0, 0.0)
    reward = learner.extract_reward("thanks, this is perfect")
    assert reward == 1.0

def test_process_feedback_positive(mock_mirror_neurons):
    learner = OpenClawOnlineLearningV8(mirror_neurons=mock_mirror_neurons, learning_rate=0.2)
    mock_mirror_neurons.empathize.return_value = (0.4, 0.0, 0.0)

    new_weight = learner.process_feedback("skill_1", "Yes, thank you")

    # reward extracted is max(p_shift, 0.5) due to positive word 'yes' or 'thank' -> 0.5
    # new_weight = 1.0 + 0.2 * 0.5 = 1.1
    assert new_weight == 1.1
    assert learner.get_skill_weight("skill_1") == 1.1
    assert len(learner.trajectory_history) == 1
    assert learner.trajectory_history[0]["skill_id"] == "skill_1"
    assert learner.trajectory_history[0]["reward"] == 0.5

def test_process_feedback_negative(mock_mirror_neurons):
    learner = OpenClawOnlineLearningV8(mirror_neurons=mock_mirror_neurons, learning_rate=0.1)
    mock_mirror_neurons.empathize.return_value = (-0.3, 0.0, 0.0)

    new_weight = learner.process_feedback("skill_2", "no, it is broken")

    # reward is min(p_shift, -0.5) due to 'no' -> -0.5
    # new_weight = 1.0 + 0.1 * -0.5 = 0.95
    assert abs(new_weight - 0.95) < 1e-6
    assert abs(learner.get_skill_weight("skill_2") - 0.95) < 1e-6

def test_weight_clamping(mock_mirror_neurons):
    learner = OpenClawOnlineLearningV8(
        mirror_neurons=mock_mirror_neurons,
        learning_rate=0.5,
        min_weight=0.5,
        max_weight=1.5
    )

    # Push weight above max
    mock_mirror_neurons.empathize.return_value = (0.9, 0.0, 0.0)
    # perfect word -> reward 1.0
    learner.process_feedback("skill_x", "perfect!")
    learner.process_feedback("skill_x", "perfect!")
    assert learner.get_skill_weight("skill_x") == 1.5

    # Push weight below min
    mock_mirror_neurons.empathize.return_value = (-0.9, 0.0, 0.0)
    # terrible word -> reward -1.0
    learner.process_feedback("skill_x", "terrible")
    learner.process_feedback("skill_x", "terrible")
    learner.process_feedback("skill_x", "terrible")
    assert learner.get_skill_weight("skill_x") == 0.5

@pytest.mark.asyncio
async def test_process_feedback_async(mock_mirror_neurons):
    learner = OpenClawOnlineLearningV8(mirror_neurons=mock_mirror_neurons, learning_rate=0.1)
    mock_mirror_neurons.empathize.return_value = (0.2, 0.0, 0.0)

    new_weight = await learner.process_feedback_async("skill_3", "it is good")

    # good word -> reward is 0.5
    # weight = 1.0 + 0.1 * 0.5 = 1.05
    assert abs(new_weight - 1.05) < 1e-6
    assert abs(learner.get_skill_weight("skill_3") - 1.05) < 1e-6
