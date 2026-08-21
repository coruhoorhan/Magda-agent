import pytest
from magda_agent.planning.dependency_graph_v9 import DependencyGraphV9

def test_get_executable_steps() -> None:
    """Tests that executable steps are correctly identified based on completed dependencies."""
    plan_steps = [
        {"id": "task1", "dependencies": []},
        {"id": "task2", "dependencies": ["task1"]},
        {"id": "task3", "dependencies": ["task1"]},
        {"id": "task4", "dependencies": ["task2", "task3"]}
    ]

    # Initially only task1 is executable
    completed: set[str] = set()
    executable = DependencyGraphV9.get_executable_steps(plan_steps, completed)
    assert len(executable) == 1
    assert executable[0]["id"] == "task1"

    # After task1 is completed, task2 and task3 are executable
    completed = {"task1"}
    executable = DependencyGraphV9.get_executable_steps(plan_steps, completed)
    assert len(executable) == 2
    ids = {step["id"] for step in executable}
    assert ids == {"task2", "task3"}

    # After task1 and task2, task3 is executable
    completed = {"task1", "task2"}
    executable = DependencyGraphV9.get_executable_steps(plan_steps, completed)
    assert len(executable) == 1
    assert executable[0]["id"] == "task3"

    # After task1, task2, and task3, task4 is executable
    completed = {"task1", "task2", "task3"}
    executable = DependencyGraphV9.get_executable_steps(plan_steps, completed)
    assert len(executable) == 1
    assert executable[0]["id"] == "task4"

def test_topological_sort() -> None:
    """Tests the topological sorting of plan steps."""
    plan_steps = [
        {"id": "task4", "dependencies": ["task2", "task3"]},
        {"id": "task2", "dependencies": ["task1"]},
        {"id": "task3", "dependencies": ["task1"]},
        {"id": "task1", "dependencies": []}
    ]

    sorted_steps = DependencyGraphV9.topological_sort(plan_steps)
    assert len(sorted_steps) == 4
    assert sorted_steps[0]["id"] == "task1"
    assert sorted_steps[3]["id"] == "task4"

    # Check cycle detection
    cyclic_steps = [
        {"id": "task1", "dependencies": ["task2"]},
        {"id": "task2", "dependencies": ["task1"]}
    ]
    with pytest.raises(ValueError, match="Cycle detected"):
        DependencyGraphV9.topological_sort(cyclic_steps)

def test_get_execution_layers() -> None:
    """Tests the calculation of execution layers for parallel execution boundaries."""
    plan_steps = [
        {"id": "task1", "dependencies": []},
        {"id": "task2", "dependencies": ["task1"]},
        {"id": "task3", "dependencies": ["task1"]},
        {"id": "task4", "dependencies": ["task2", "task3"]}
    ]

    layers = DependencyGraphV9.get_execution_layers(plan_steps)
    assert len(layers) == 3
    assert len(layers[0]) == 1 and layers[0][0]["id"] == "task1"
    assert len(layers[1]) == 2
    assert {"task2", "task3"} == {step["id"] for step in layers[1]}
    assert len(layers[2]) == 1 and layers[2][0]["id"] == "task4"

    # Cycle detection
    cyclic_steps = [
        {"id": "task1", "dependencies": ["task2"]},
        {"id": "task2", "dependencies": ["task1"]}
    ]
    with pytest.raises(ValueError, match="Cycle detected"):
        DependencyGraphV9.get_execution_layers(cyclic_steps)
