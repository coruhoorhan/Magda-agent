import pytest
import threading
from magda_agent.learning.reward_heuristic_v1 import OpenClawRewardHeuristicV1

def test_parse_rating_valid_commands():
    heuristic = OpenClawRewardHeuristicV1()

    assert heuristic.parse_rating("This was great! /rate 5") == 1.0
    assert heuristic.parse_rating("/rate 1 Terrible") == -1.0
    assert heuristic.parse_rating("Average response /rate 3.") == 0.0
    assert heuristic.parse_rating("/rate 4") == 0.5
    assert heuristic.parse_rating("/rate 2") == -0.5

def test_parse_rating_out_of_bounds():
    heuristic = OpenClawRewardHeuristicV1()

    # Should clamp to 1.0
    assert heuristic.parse_rating("/rate 10") == 1.0
    # Should clamp to -1.0
    assert heuristic.parse_rating("/rate -2") == -1.0 # Wait, regex doesn't match negative.
    assert heuristic.parse_rating("/rate 0") == -1.0
    assert heuristic.parse_rating("/rate 0.5") == -1.0

def test_parse_rating_invalid_commands():
    heuristic = OpenClawRewardHeuristicV1()

    assert heuristic.parse_rating("rate 5") is None
    assert heuristic.parse_rating("/rate ") is None
    assert heuristic.parse_rating("No rating here") is None

def test_process_user_reply_updates_weight():
    heuristic = OpenClawRewardHeuristicV1()
    skill_id = "test_skill"

    new_weight = heuristic.process_user_reply("Good job /rate 4", skill_id)
    assert new_weight == 0.5
    assert heuristic.weights[skill_id] == 0.5

    new_weight = heuristic.process_user_reply("/rate 5", skill_id)
    assert new_weight == 1.5
    assert heuristic.weights[skill_id] == 1.5

    new_weight = heuristic.process_user_reply("No rating", skill_id)
    assert new_weight is None
    assert heuristic.weights[skill_id] == 1.5

def test_apply_reward_thread_safety():
    heuristic = OpenClawRewardHeuristicV1()
    skill_id = "concurrent_skill"

    def apply_concurrently():
        for _ in range(1000):
            heuristic.apply_reward(skill_id, 0.01)

    threads = []
    for _ in range(10):
        t = threading.Thread(target=apply_concurrently)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert pytest.approx(heuristic.weights[skill_id], 0.001) == 100.0
