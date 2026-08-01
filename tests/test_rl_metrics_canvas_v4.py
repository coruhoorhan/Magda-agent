import json
from unittest.mock import MagicMock
from magda_agent.visualization.rl_metrics_canvas_v4 import RLMetricsCanvasV4
from magda_agent.learning.canvas_metrics import RLCanvasMetricsExporter


def test_get_metrics_payload() -> None:
    """
    Tests that get_metrics_payload correctly delegates to the exporter.
    """
    mock_exporter = MagicMock(spec=RLCanvasMetricsExporter)

    mock_payload = {
        "status": "active",
        "total_rewards_received": 10,
        "global_average_q": 0.5,
        "moving_average_reward_5": 0.8,
        "trajectory_trend": "improving",
        "skill_distribution_entropy": 0.1,
        "skills_coverage": {},
        "raw_rewards_trajectory": []
    }

    mock_exporter.export_canvas_payload.return_value = mock_payload

    visualizer = RLMetricsCanvasV4(exporter=mock_exporter)

    payload = visualizer.get_metrics_payload()

    assert payload == mock_payload
    mock_exporter.export_canvas_payload.assert_called_once()


def test_get_metrics_json() -> None:
    """
    Tests that get_metrics_json correctly formats the payload as a JSON string.
    """
    mock_exporter = MagicMock(spec=RLCanvasMetricsExporter)

    mock_payload = {
        "status": "active",
        "total_rewards_received": 5
    }

    mock_exporter.export_canvas_payload.return_value = mock_payload

    visualizer = RLMetricsCanvasV4(exporter=mock_exporter)

    json_string = visualizer.get_metrics_json()

    assert json_string == json.dumps(mock_payload)
    mock_exporter.export_canvas_payload.assert_called_once()
