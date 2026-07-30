import asyncio
import json
import time
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from aether_deep_agent_service.app import build_application, resolve_run_timeout
from aether_deep_agent_service.executor import ExecutionResult
from aether_deep_agent_service.schemas import DeepRunRequest
from aether_deep_agent_service.security import build_signature
from aether_deep_agent_service.settings import Settings


def test_health() -> None:
    app = build_application(Settings(shared_secret="test-secret", database_url="sqlite+aiosqlite://"))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_create_run_requires_valid_signature() -> None:
    settings = Settings(shared_secret="test-secret", database_url="sqlite+aiosqlite://")
    app = build_application(settings)
    payload = {
        "run_id": "run-1", "user_id": "user-1", "agent_id": "agent-1",
        "conversation_id": "conversation-1", "task": "Summarize the supplied evidence.",
        "delegation_token": "delegation-token",
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    timestamp = str(int(time.time()))
    headers = {
        "X-Aether-Key-Id": settings.key_id,
        "X-Aether-Timestamp": timestamp,
        "X-Aether-Signature": build_signature(settings.shared_secret, timestamp, body),
        "Content-Type": "application/json",
    }
    with TestClient(app) as client:
        response = client.post("/v1/runs", content=body, headers=headers)
    assert response.status_code == 202
    assert response.json()["run_id"] == "run-1"
    assert response.json()["created"] is True


def test_submission_uses_configured_timeout_when_request_omits_it() -> None:
    request = DeepRunRequest(
        run_id="configured-timeout", user_id="user-1", agent_id="agent-1",
        conversation_id="conversation-1", task="Summarize evidence.", delegation_token="token",
    )

    assert resolve_run_timeout(request, Settings(run_timeout_seconds=321)) == 321


def test_submission_preserves_explicit_timeout_over_configured_default() -> None:
    request = DeepRunRequest(
        run_id="explicit-timeout", user_id="user-1", agent_id="agent-1",
        conversation_id="conversation-1", task="Summarize evidence.", delegation_token="token", timeout_seconds=123,
    )

    assert resolve_run_timeout(request, Settings(run_timeout_seconds=321)) == 123


def test_run_started_callback_failure_does_not_prevent_successful_execution(monkeypatch) -> None:
    settings = Settings(shared_secret="test-secret", database_url="sqlite+aiosqlite://")
    execute = AsyncMock(return_value=ExecutionResult(
        content="Result", citations=[], model="test-model", tools=[], prompt_tokens=None, completion_tokens=None,
    ))
    send = AsyncMock(side_effect=lambda _run_id, event_type, _data: (
        (_ for _ in ()).throw(RuntimeError("callback unavailable")) if event_type == "run.started" else None
    ))
    stores = []

    class FakeStore:
        def __init__(self, _database_url) -> None:
            self.updates = []
            stores.append(self)

        async def initialize(self) -> None:
            pass

        async def create_if_absent(self, request):
            return type("Record", (), {"run_id": request.run_id, "status": "QUEUED"})(), True

        async def update(self, run_id, status, result=None, error=None) -> None:
            self.updates.append((run_id, status, result, error))

    monkeypatch.setattr("aether_deep_agent_service.app.DeepAgentExecutor.execute", execute)
    monkeypatch.setattr("aether_deep_agent_service.app.CallbackClient.send", send)
    monkeypatch.setattr("aether_deep_agent_service.app.RunStore", FakeStore)

    app = build_application(settings)
    payload = {
        "run_id": "run-started-failure", "user_id": "user-1", "agent_id": "agent-1",
        "conversation_id": "conversation-1", "task": "Summarize evidence.", "delegation_token": "token",
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    timestamp = str(int(time.time()))
    with TestClient(app) as client:
        response = client.post("/v1/runs", content=body, headers={
            "X-Aether-Key-Id": settings.key_id,
            "X-Aether-Timestamp": timestamp,
            "X-Aether-Signature": build_signature(settings.shared_secret, timestamp, body),
            "Content-Type": "application/json",
        })
        assert response.status_code == 202
        client.portal.call(asyncio.sleep, 0)
        assert execute.await_count == 1
    assert stores[0].updates[-1] == ("run-started-failure", "SUCCEEDED", "Result", None)


def test_run_completed_callback_failure_does_not_change_successful_status(monkeypatch) -> None:
    settings = Settings(shared_secret="test-secret", database_url="sqlite+aiosqlite://")
    execute = AsyncMock(return_value=ExecutionResult(
        content="Result", citations=[], model="test-model", tools=[], prompt_tokens=None, completion_tokens=None,
    ))
    send = AsyncMock(side_effect=lambda _run_id, event_type, _data: (
        (_ for _ in ()).throw(RuntimeError("callback unavailable")) if event_type == "run.completed" else None
    ))
    stores = []

    class FakeStore:
        def __init__(self, _database_url) -> None:
            self.updates = []
            stores.append(self)

        async def initialize(self) -> None:
            pass

        async def create_if_absent(self, request):
            return type("Record", (), {"run_id": request.run_id, "status": "QUEUED"})(), True

        async def update(self, run_id, status, result=None, error=None) -> None:
            self.updates.append((run_id, status, result, error))

    monkeypatch.setattr("aether_deep_agent_service.app.DeepAgentExecutor.execute", execute)
    monkeypatch.setattr("aether_deep_agent_service.app.CallbackClient.send", send)
    monkeypatch.setattr("aether_deep_agent_service.app.RunStore", FakeStore)

    app = build_application(settings)
    payload = {
        "run_id": "run-completed-failure", "user_id": "user-1", "agent_id": "agent-1",
        "conversation_id": "conversation-1", "task": "Summarize evidence.", "delegation_token": "token",
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    timestamp = str(int(time.time()))
    with TestClient(app) as client:
        response = client.post("/v1/runs", content=body, headers={
            "X-Aether-Key-Id": settings.key_id,
            "X-Aether-Timestamp": timestamp,
            "X-Aether-Signature": build_signature(settings.shared_secret, timestamp, body),
            "Content-Type": "application/json",
        })
        assert response.status_code == 202
        client.portal.call(asyncio.sleep, 0)
        assert execute.await_count == 1
    assert stores[0].updates[-1] == ("run-completed-failure", "SUCCEEDED", "Result", None)


def test_run_started_callback_cancellation_persists_cancelled_status(monkeypatch) -> None:
    settings = Settings(shared_secret="test-secret", database_url="sqlite+aiosqlite://")
    stores = []

    class FakeStore:
        def __init__(self, _database_url) -> None:
            self.updates = []
            stores.append(self)

        async def initialize(self) -> None:
            pass

        async def create_if_absent(self, request):
            return type("Record", (), {"run_id": request.run_id, "status": "QUEUED"})(), True

        async def update(self, run_id, status, result=None, error=None) -> None:
            self.updates.append((run_id, status, result, error))

    monkeypatch.setattr("aether_deep_agent_service.app.CallbackClient.send", AsyncMock(
        side_effect=asyncio.CancelledError,
    ))
    monkeypatch.setattr("aether_deep_agent_service.app.RunStore", FakeStore)

    app = build_application(settings)
    payload = {
        "run_id": "run-started-cancelled", "user_id": "user-1", "agent_id": "agent-1",
        "conversation_id": "conversation-1", "task": "Summarize evidence.", "delegation_token": "token",
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    timestamp = str(int(time.time()))
    with TestClient(app) as client:
        response = client.post("/v1/runs", content=body, headers={
            "X-Aether-Key-Id": settings.key_id,
            "X-Aether-Timestamp": timestamp,
            "X-Aether-Signature": build_signature(settings.shared_secret, timestamp, body),
            "Content-Type": "application/json",
        })
        assert response.status_code == 202
        client.portal.call(asyncio.sleep, 0)
    assert stores[0].updates[-1] == ("run-started-cancelled", "CANCELLED", None, None)
