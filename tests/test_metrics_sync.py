import pytest
import asyncio
import json
from magda_agent.operations.metrics_sync import MetricsSyncService
from magda_agent.telemetry.quality_metrics import LongitudinalQualityMetricsTracker
from magda_agent.channels.hub import ChannelHub
from magda_agent.channels.base import ChannelAdapter

class MockMetricsEndpointAdapter(ChannelAdapter):
    def __init__(self, channel_id="metrics_endpoint"):
        self.channel_id = channel_id
        self.sent_messages = []

    async def send(self, recipient_id: str, text: str, metadata=None):
        self.sent_messages.append({"recipient_id": recipient_id, "text": text, "metadata": metadata})
        return "mock_success"

    async def receive(self, raw_data):
        return None

@pytest.fixture
def mock_tracker() -> LongitudinalQualityMetricsTracker:
    return LongitudinalQualityMetricsTracker(db_path=":memory:")

@pytest.fixture
def mock_hub() -> ChannelHub:
    hub = ChannelHub()
    adapter = MockMetricsEndpointAdapter()
    hub.register_adapter(adapter)
    return hub

@pytest.mark.asyncio
async def test_metrics_sync_service_flushes_correctly(mock_tracker: LongitudinalQualityMetricsTracker, mock_hub: ChannelHub) -> None:
    # Record some metrics
    mock_tracker.record_metric("success_rate", 0.95, {"task": "test"})
    mock_tracker.record_metric("latency_ms", 120.5)

    service = MetricsSyncService(mock_tracker, mock_hub, sync_interval=0) # 0 for immediate cycle exit if testing start loop

    await service.flush_metrics()

    adapter = mock_hub.get_adapter("metrics_endpoint")
    assert len(adapter.sent_messages) == 1

    payload = json.loads(adapter.sent_messages[0]["text"])
    assert payload["type"] == "quality_metrics_sync"
    assert payload["count"] == 2
    assert len(payload["metrics"]) == 2
    assert adapter.sent_messages[0]["metadata"]["source"] == "MetricsSyncService"

@pytest.mark.asyncio
async def test_metrics_sync_loop_zero_interval(mock_tracker: LongitudinalQualityMetricsTracker, mock_hub: ChannelHub) -> None:
    service = MetricsSyncService(mock_tracker, mock_hub, sync_interval=0.0)
    mock_tracker.record_metric("success_rate", 0.95, {"task": "test"})

    await service.start()
    await asyncio.sleep(0.01) # let the loop run
    await service.stop()

    adapter = mock_hub.get_adapter("metrics_endpoint")
    assert len(adapter.sent_messages) >= 1
