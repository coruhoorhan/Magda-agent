import pytest
from magda_agent.planning.dependency_graph import DependencyGraph

def test_topological_sort():
    plan_steps = [
        {"id": "task_3", "dependencies": ["task_2"]},
        {"id": "task_1", "dependencies": []},
        {"id": "task_2", "dependencies": ["task_1"]},
    ]
    sorted_steps = DependencyGraph.topological_sort(plan_steps)
    sorted_ids = [step["id"] for step in sorted_steps]
    assert sorted_ids == ["task_1", "task_2", "task_3"]

def test_topological_sort_cycle():
    plan_steps = [
        {"id": "task_1", "dependencies": ["task_3"]},
        {"id": "task_2", "dependencies": ["task_1"]},
        {"id": "task_3", "dependencies": ["task_2"]},
    ]
    with pytest.raises(ValueError, match="Cycle detected"):
        DependencyGraph.topological_sort(plan_steps)

def test_get_execution_layers():
    plan_steps = [
        {"id": "task_5", "dependencies": ["task_3", "task_4"]},
        {"id": "task_3", "dependencies": ["task_1", "task_2"]},
        {"id": "task_1", "dependencies": []},
        {"id": "task_2", "dependencies": []},
        {"id": "task_4", "dependencies": ["task_1"]},
    ]
    layers = DependencyGraph.get_execution_layers(plan_steps)
    assert len(layers) == 3
    layer_1 = {step["id"] for step in layers[0]}
    layer_2 = {step["id"] for step in layers[1]}
    layer_3 = {step["id"] for step in layers[2]}

    assert layer_1 == {"task_1", "task_2"}
    assert layer_2 == {"task_3", "task_4"}
    assert layer_3 == {"task_5"}

def test_get_executable_steps():
    plan_steps = [
        {"id": "task_5", "dependencies": ["task_3", "task_4"]},
        {"id": "task_3", "dependencies": ["task_1", "task_2"]},
        {"id": "task_1", "dependencies": []},
        {"id": "task_2", "dependencies": []},
        {"id": "task_4", "dependencies": ["task_1"]},
    ]
    completed_steps = {"task_1"}

    executable = DependencyGraph.get_executable_steps(plan_steps, completed_steps)
    executable_ids = {step["id"] for step in executable}
    assert executable_ids == {"task_2", "task_4"}
