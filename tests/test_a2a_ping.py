import pytest
import httpx
from unittest.mock import AsyncMock, patch

from magda_agent.integration.a2a_ping import A2APingClient, A2APingError

@pytest.mark.asyncio
async def test_ping_task_status_success():
    client = A2APingClient(timeout=2.0)
    mock_request = httpx.Request("GET", "http://peer-agent:8000/a2a/tasks/task-123/status")
    mock_response = httpx.Response(200, json={"status": "in_progress", "progress": 50}, request=mock_request)

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        result = await client.ping_task_status("http://peer-agent:8000", "task-123")

        mock_get.assert_called_once_with("http://peer-agent:8000/a2a/tasks/task-123/status")
        assert result == {"status": "in_progress", "progress": 50}

@pytest.mark.asyncio
async def test_ping_task_status_http_error():
    client = A2APingClient(timeout=2.0)

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_request = httpx.Request("GET", "http://peer-agent:8000/a2a/tasks/task-404/status")
        mock_response = httpx.Response(404, request=mock_request)
        mock_get.side_effect = httpx.HTTPStatusError("Not Found", request=mock_request, response=mock_response)

        with pytest.raises(A2APingError) as exc_info:
            await client.ping_task_status("http://peer-agent:8000", "task-404")

        assert "HTTP 404" in str(exc_info.value)

@pytest.mark.asyncio
async def test_ping_task_status_timeout():
    client = A2APingClient(timeout=2.0)

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_request = httpx.Request("GET", "http://peer-agent:8000/a2a/tasks/task-timeout/status")
        mock_get.side_effect = httpx.RequestError("Timeout", request=mock_request)

        with pytest.raises(A2APingError) as exc_info:
            await client.ping_task_status("http://peer-agent:8000", "task-timeout")

        assert "Network error while pinging task" in str(exc_info.value)
