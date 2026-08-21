import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from magda_agent.llm_client import LLMClient
from magda_agent.architecture.magentic_one_v2 import MagenticOneWorkerV2
from magda_agent.architecture.magentic_one_router_v2 import MagenticOneOrchestrationRouterV2


class MockLLMClient(LLMClient):
    def __init__(self):
        self.chat_completion = AsyncMock()


def test_route_successful():
    llm = MockLLMClient()
    llm.chat_completion.side_effect = [
        "WorkerB",  # First call selects worker
        "WorkerB executed the task"  # Call from inside WorkerB's execute_subtask
    ]

    worker_a = MagenticOneWorkerV2("WorkerA", "Does A things", llm)
    worker_b = MagenticOneWorkerV2("WorkerB", "Does B things", llm)
    workers = [worker_a, worker_b]

    router = MagenticOneOrchestrationRouterV2(llm, workers)

    result = asyncio.run(router.route("Do a B thing", []))

    assert "WorkerB executed the task" in result
    assert llm.chat_completion.call_count == 2
    # First call is router selecting worker
    assert "WorkerB" in llm.chat_completion.call_args_list[0][0][0][0]["content"] or "Available workers" in llm.chat_completion.call_args_list[0][0][0][0]["content"]
    # Second call is the worker executing
    assert "WorkerB" in llm.chat_completion.call_args_list[1][0][0][0]["content"]


def test_route_fallback_when_worker_not_found():
    llm = MockLLMClient()
    llm.chat_completion.side_effect = [
        "NonExistentWorker",  # Router selects unknown worker
        "WorkerA executed the task"  # Fallback to WorkerA
    ]

    worker_a = MagenticOneWorkerV2("WorkerA", "Does A things", llm)
    workers = [worker_a]

    router = MagenticOneOrchestrationRouterV2(llm, workers)

    result = asyncio.run(router.route("Do something", []))

    assert "WorkerA executed the task" in result


def test_route_no_workers():
    llm = MockLLMClient()
    router = MagenticOneOrchestrationRouterV2(llm, [])

    result = asyncio.run(router.route("Do something", []))

    assert "Error: No workers available." in result


def test_route_exception_handling():
    llm = MockLLMClient()
    llm.chat_completion.side_effect = Exception("LLM connection failed")

    worker_a = MagenticOneWorkerV2("WorkerA", "Does A things", llm)
    router = MagenticOneOrchestrationRouterV2(llm, [worker_a])

    result = asyncio.run(router.route("Do something", []))

    assert "Routing error" in result
    assert "LLM connection failed" in result
