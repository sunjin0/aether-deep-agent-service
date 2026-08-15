import asyncio
import json
import time
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from aether_deep_agent_service.app import (
    build_application,
    build_ask_user_response_decisions,
    normalize_ask_user_payload,
    resolve_run_timeout,
)
from aether_deep_agent_service.executor import ExecutionResult
from aether_deep_agent_service.schemas import DeepRunRequest
from aether_deep_agent_service.security import build_signature
from aether_deep_agent_service.settings import Settings


def test_health() -> None:
    app = build_application(Settings(shared_secret="test-secret", database_url="sqlite+aiosqlite://"))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_create_run_requires_valid_signature(monkeypatch) -> None:
    # 真实后台运行会做回调投递；空 callback_base_url 下 httpx 在 Windows 的
    # TestClient 退出取消时可能挂起。mock 投递层使后台任务立即结束，不依赖网络。
    settings = Settings(shared_secret="test-secret", database_url="sqlite+aiosqlite://")
    monkeypatch.setattr("aether_deep_agent_service.app.CallbackClient.send_event", AsyncMock())
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
        # 让后台 run 任务在 TestClient 退出前跑完，避免 Windows 下取消 sqlite 操作挂起。
        client.portal.call(asyncio.sleep, 0.2)
    assert response.status_code == 202
    assert response.json()["run_id"] == "run-1"
    assert response.json()["created"] is True


def test_session_task_alias_persists_session_and_exposes_latest_status(monkeypatch) -> None:
    settings = Settings(shared_secret="test-secret", database_url="sqlite+aiosqlite://")
    monkeypatch.setattr("aether_deep_agent_service.app.DeepAgentExecutor.plan", AsyncMock(return_value=[{"title": "准备证据"}]))
    monkeypatch.setattr("aether_deep_agent_service.app.DeepAgentExecutor.execute", AsyncMock(return_value=ExecutionResult(
        content="Result", citations=[], model="test-model", tools=[], prompt_tokens=None, completion_tokens=None,
    )))
    # 真实 RunStore 下 safe_callback 走 send_event；mock 投递层避免真实网络请求。
    monkeypatch.setattr("aether_deep_agent_service.app.CallbackClient.send_event", AsyncMock())
    payload = {
        "run_id": "session-run-1", "user_id": "user-1", "agent_id": "agent-1",
        "conversation_id": "conversation-1", "task_id": "task-1",
        "task": "Summarize the supplied evidence.", "delegation_token": "delegation-token",
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    timestamp = str(int(time.time()))
    headers = {
        "X-Aether-Key-Id": settings.key_id,
        "X-Aether-Timestamp": timestamp,
        "X-Aether-Signature": build_signature(settings.shared_secret, timestamp, body),
        "Content-Type": "application/json",
    }
    with TestClient(build_application(settings)) as client:
        created = client.post("/v1/sessions/session-1/tasks", content=body, headers=headers)
        assert created.status_code == 202
        get_timestamp = str(int(time.time()))
        queried = client.get("/v1/sessions/session-1", headers={
            "X-Aether-Key-Id": settings.key_id,
            "X-Aether-Timestamp": get_timestamp,
            "X-Aether-Signature": build_signature(settings.shared_secret, get_timestamp, b""),
        })
        # 让后台 run 任务在 TestClient 退出前跑完，避免 Windows 下取消挂起。
        client.portal.call(asyncio.sleep, 0.3)
    assert queried.status_code == 200
    assert queried.json()["run_id"] == "session-run-1"
    assert queried.json()["task_id"] == "task-1"


def test_session_task_alias_rejects_mismatched_session_id() -> None:
    settings = Settings(shared_secret="test-secret", database_url="sqlite+aiosqlite://")
    payload = {
        "run_id": "session-mismatch", "user_id": "user-1", "agent_id": "agent-1",
        "conversation_id": "conversation-1", "session_id": "another-session",
        "task": "Summarize the supplied evidence.", "delegation_token": "delegation-token",
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    timestamp = str(int(time.time()))
    with TestClient(build_application(settings)) as client:
        response = client.post("/v1/sessions/session-1/tasks", content=body, headers={
            "X-Aether-Key-Id": settings.key_id,
            "X-Aether-Timestamp": timestamp,
            "X-Aether-Signature": build_signature(settings.shared_secret, timestamp, body),
            "Content-Type": "application/json",
        })
    assert response.status_code == 422


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


def test_ask_user_normalizes_missing_options_to_choice_with_custom_input() -> None:
    payload = normalize_ask_user_payload({
        "questions": [{"id": "target", "question": "请提供目标文档或用户 ID"}],
    })

    question = payload["questions"][0]
    assert question["type"] == "choice"
    assert question["allowCustomInput"] is True
    assert len(question["options"]) == 2


def test_ask_user_answer_is_sent_as_human_response() -> None:
    decisions = build_ask_user_response_decisions({"target": {"selected": "用户 123"}})

    assert decisions[0]["type"] == "respond"
    assert '"用户 123"' in decisions[0]["message"]


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


def test_run_emits_task_plan_updates(monkeypatch) -> None:
    settings = Settings(shared_secret="test-secret", database_url="sqlite+aiosqlite://")
    execute = AsyncMock(return_value=ExecutionResult(
        content="Result", citations=[], model="test-model", tools=[], prompt_tokens=None, completion_tokens=None,
    ))
    send = AsyncMock()

    class FakeStore:
        def __init__(self, _database_url) -> None:
            pass

        async def initialize(self) -> None:
            pass

        async def create_if_absent(self, request):
            return type("Record", (), {"run_id": request.run_id, "status": "QUEUED"})(), True

        async def update(self, _run_id, _status, result=None, error=None) -> None:
            pass

    monkeypatch.setattr("aether_deep_agent_service.app.DeepAgentExecutor.execute", execute)
    monkeypatch.setattr("aether_deep_agent_service.app.CallbackClient.send", send)
    monkeypatch.setattr("aether_deep_agent_service.app.RunStore", FakeStore)

    app = build_application(settings)
    payload = {
        "run_id": "task-plan", "user_id": "user-1", "agent_id": "agent-1",
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
        client.portal.call(asyncio.sleep, 0.05)

    plans = [call.args[2] for call in send.await_args_list if call.args[1] == "plan.updated"]
    assert len(plans) == 2
    assert plans[0]["tasks"][0]["status"] == "running"
    assert plans[-1]["tasks"][-1]["status"] == "completed"


def test_tool_failure_emits_replanned_task_plan(monkeypatch) -> None:
    settings = Settings(shared_secret="test-secret", database_url="sqlite+aiosqlite://")
    send = AsyncMock()

    async def execute_with_failure(_executor, _request, emit, _checkpointer):
        await emit("tool.failed", {"toolName": "read_document", "error": "timeout"})
        return ExecutionResult(content="Result", citations=[], model="test-model", tools=[], prompt_tokens=None, completion_tokens=None)

    class FakeStore:
        def __init__(self, _database_url) -> None:
            pass

        async def initialize(self) -> None:
            pass

        async def create_if_absent(self, request):
            return type("Record", (), {"run_id": request.run_id, "status": "QUEUED"})(), True

        async def update(self, _run_id, _status, result=None, error=None) -> None:
            pass

    monkeypatch.setattr("aether_deep_agent_service.app.DeepAgentExecutor.execute", execute_with_failure)
    monkeypatch.setattr("aether_deep_agent_service.app.CallbackClient.send", send)
    monkeypatch.setattr("aether_deep_agent_service.app.RunStore", FakeStore)
    app = build_application(settings)
    payload = {
        "run_id": "replan-on-failure", "user_id": "user-1", "agent_id": "agent-1",
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
        client.portal.call(asyncio.sleep, 0.05)

    plans = [call.args[2] for call in send.await_args_list if call.args[1] == "plan.updated"]
    assert [plan["reason"] for plan in plans] == ["INITIAL", "STEP_FAILED", "COMPLETED"]


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
