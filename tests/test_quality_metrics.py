import pytest
import sqlite3
import json
from magda_agent.telemetry.quality_metrics import LongitudinalQualityMetricsTracker


@pytest.fixture
def memory_tracker():
    """Provides a LongitudinalQualityMetricsTracker using an in-memory SQLite database."""
    tracker = LongitudinalQualityMetricsTracker(db_path=":memory:")
    return tracker


def test_initialization(memory_tracker):
    """Test that the tracker initializes without errors and creates the table."""
    # Check if table exists. Use the memory_tracker's connection to see it!
    conn = memory_tracker._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metrics';")
    assert cursor.fetchone() is not None


def test_record_and_get_metrics(memory_tracker):
    """Test recording metrics and retrieving their history."""
    memory_tracker.record_metric("success_rate", 0.95, {"task_id": "t1"})
    memory_tracker.record_metric("success_rate", 0.85, {"task_id": "t2"})
    memory_tracker.record_metric("latency_ms", 120.5)

    # Retrieve history for success_rate
    history = memory_tracker.get_metrics_history("success_rate", limit=10)
    assert len(history) == 2

    values = {record["metric_value"] for record in history}
    assert 0.95 in values
    assert 0.85 in values

    # Retrieve history for latency_ms
    latency_history = memory_tracker.get_metrics_history("latency_ms", limit=10)
    assert len(latency_history) == 1
    assert latency_history[0]["metric_value"] == 120.5
    assert latency_history[0]["metadata"] is None


def test_get_average_metric(memory_tracker):
    """Test calculating the average of a metric."""
    memory_tracker.record_metric("score", 10.0)
    memory_tracker.record_metric("score", 20.0)
    memory_tracker.record_metric("score", 30.0)
    memory_tracker.record_metric("other_score", 100.0)

    avg = memory_tracker.get_average_metric("score")
    assert avg == 20.0

    avg_other = memory_tracker.get_average_metric("other_score")
    assert avg_other == 100.0

    avg_empty = memory_tracker.get_average_metric("empty_metric")
    assert avg_empty is None


def test_metadata_json(memory_tracker):
    """Test that metadata is correctly stored and parsed as JSON."""
    complex_metadata = {"agent": "Jules", "version": "1.0.0", "flags": [True, False, 1]}
    memory_tracker.record_metric("eval_score", 4.5, complex_metadata)

    history = memory_tracker.get_metrics_history("eval_score")
    assert len(history) == 1
    retrieved_metadata = history[0]["metadata"]

    assert retrieved_metadata == complex_metadata
    assert retrieved_metadata["agent"] == "Jules"
    assert retrieved_metadata["flags"] == [True, False, 1]
