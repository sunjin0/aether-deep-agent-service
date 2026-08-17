"""服务重启恢复与幂等重投验证。

这些测试通过一个 ``RunStore`` 实例预置持久化状态（运行、发件箱、待处理交互和
检查点），再基于同一 SQLite 文件重建应用，以模拟 Deep Agent 服务重启。投递层与
图执行器均被模拟，因此断言只验证恢复流程衔接，而不依赖模型或网络。

每个场景均在单个 ``asyncio.run`` 内运行，避免 pytest-asyncio 测试循环或
TestClient 门户线程跨测试泄漏（现有套件使用同步测试与 TestClient，必须保持不变）。
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock

import httpx

from aether_deep_agent_service.app import build_application
from aether_deep_agent_service.callbacks import CallbackClient
from aether_deep_agent_service.executor import DeepAgentExecutor, ExecutionResult
from aether_deep_agent_service.schemas import DeepRunRequest, RunStatus
from aether_deep_agent_service.security import build_signature
from aether_deep_agent_service.settings import Settings
from aether_deep_agent_service.store import RunStore


def _db_url(tmp_path, name="agent.db") -> str:
    return "sqlite+aiosqlite:///" + str(tmp_path / name).replace("\\", "/")


def _signed_headers(settings: Settings, body: bytes) -> dict[str, str]:
    timestamp = str(int(time.time()))
    return {
        "X-Aether-Key-Id": settings.key_id,
        "X-Aether-Timestamp": timestamp,
        "X-Aether-Signature": build_signature(settings.shared_secret, timestamp, body),
        "Content-Type": "application/json",
    }


def _deep_request(**overrides) -> DeepRunRequest:
    base = {
        "run_id": "run-1", "user_id": "user-1", "agent_id": "agent-1",
        "conversation_id": "conversation-1", "session_id": "session-1", "task_id": "task-1",
        "task": "Summarize evidence.", "delegation_token": "token",
    }
    base.update(overrides)
    return DeepRunRequest(**base)


def _result() -> ExecutionResult:
    return ExecutionResult(content="ok", citations=[], model="test-model", tools=[],
                           prompt_tokens=None, completion_tokens=None)


def _settings(url: str) -> Settings:
    return Settings(shared_secret="test-secret", database_url=url)


async def _post(app, settings: Settings, path: str, payload: dict):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, content=body, headers=_signed_headers(settings, body))


def test_pause_incomplete_runs_on_restart(tmp_path) -> None:
    async def scenario() -> None:
        url = _db_url(tmp_path)
        store = RunStore(url)
        await store.initialize()
        await store.create_if_absent(_deep_request(run_id="queued-run"))
        await store.create_if_absent(_deep_request(run_id="succeeded-run"))
        await store.update("succeeded-run", RunStatus.SUCCEEDED, result="done")
        await store.engine.dispose()

        # 新建应用等同于服务重启；生命周期会将中断的运行标记为 PAUSED。
        app = build_application(_settings(url))
        async with app.router.lifespan_context(app):
            pass

        restarted = RunStore(url)
        await restarted.initialize()
        queued = await restarted.get("queued-run")
        succeeded = await restarted.get("succeeded-run")
        await restarted.engine.dispose()
        assert queued.status == RunStatus.PAUSED
        assert succeeded.status == RunStatus.SUCCEEDED

    asyncio.run(scenario())


def test_undelivered_outbox_replayed_and_delivered_skipped_on_restart(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        url = _db_url(tmp_path)
        store = RunStore(url)
        await store.initialize()
        await store.create_if_absent(_deep_request(run_id="run-1"))
        await store.enqueue_callback("evt-undelivered", "run-1", "plan.updated",
                                     {"reason": "INITIAL", "tasks": []}, 100)
        await store.enqueue_callback("evt-delivered", "run-1", "tool.started",
                                     {"toolName": "read_document"}, 200)
        await store.mark_callback_delivered("evt-delivered")
        await store.engine.dispose()

        send = AsyncMock()
        monkeypatch.setattr(CallbackClient, "send_event", send)
        app = build_application(_settings(url))
        async with app.router.lifespan_context(app):
            pass

        sent = [call.args[0] for call in send.await_args_list]
        assert [event.event_id for event in sent] == ["evt-undelivered"]
        assert [event.event_type for event in sent] == ["plan.updated"]

        restarted = RunStore(url)
        await restarted.initialize()
        pending = await restarted.pending_callbacks()
        await restarted.engine.dispose()
        # 重放事件会被标记为已投递，第二次重启时不会再次发送。
        assert pending == []

    asyncio.run(scenario())


def test_enqueue_callback_is_idempotent_by_event_id(tmp_path) -> None:
    async def scenario() -> None:
        url = _db_url(tmp_path)
        store = RunStore(url)
        await store.initialize()
        await store.enqueue_callback("evt-same", "run-1", "plan.updated", {"reason": "INITIAL"}, 1)
        await store.enqueue_callback("evt-same", "run-1", "plan.updated", {"reason": "CHANGED"}, 2)
        pending = await store.pending_callbacks()
        await store.engine.dispose()
        assert len(pending) == 1
        assert pending[0].data == {"reason": "INITIAL"}

    asyncio.run(scenario())


def test_tool_approval_interaction_restored_after_restart(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        url = _db_url(tmp_path)
        store = RunStore(url)
        await store.initialize()
        await store.create_if_absent(_deep_request(run_id="run-1"))
        await store.save_interaction("run-1", "tool_approval", {
            "actions": [{"name": "http_request", "args": {"url": "https://example.com"}}],
        })
        await store.engine.dispose()

        send = AsyncMock()
        resume = AsyncMock(return_value=_result())
        monkeypatch.setattr(CallbackClient, "send_event", send)
        monkeypatch.setattr(DeepAgentExecutor, "resume", resume)

        settings = _settings(url)
        app = build_application(settings)
        async with app.router.lifespan_context(app):
            response = await _post(app, settings, "/v1/runs/run-1/resume",
                                   {"decisions": [{"type": "approve"}]})
            assert response.status_code == 202
            await asyncio.sleep(0.3)

        assert response.json() == {"runId": "run-1", "status": "RUNNING"}
        assert resume.await_count == 1
        assert resume.await_args.args[1] == [{"type": "approve"}]

        restarted = RunStore(url)
        await restarted.initialize()
        interaction = await restarted.take_interaction("run-1")
        await restarted.engine.dispose()
        assert interaction is None  # 交互记录只能被消费一次

    asyncio.run(scenario())


def test_ask_user_interaction_restored_after_restart(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        url = _db_url(tmp_path)
        store = RunStore(url)
        await store.initialize()
        await store.create_if_absent(_deep_request(run_id="run-1"))
        await store.save_interaction("run-1", "ask_user", {
            "actions": [{"name": "ask_user", "args": {
                "question": "请选择目标文档", "questions": [{"id": "target", "question": "目标文档？"}],
            }}],
        })
        await store.engine.dispose()

        send = AsyncMock()
        resume = AsyncMock(return_value=_result())
        monkeypatch.setattr(CallbackClient, "send_event", send)
        monkeypatch.setattr(DeepAgentExecutor, "resume", resume)

        settings = _settings(url)
        app = build_application(settings)
        async with app.router.lifespan_context(app):
            response = await _post(app, settings, "/v1/runs/run-1/resume",
                                   {"answers": {"target": {"selected": "用户 123"}}})
            assert response.status_code == 202
            await asyncio.sleep(0.3)

        assert response.json() == {"runId": "run-1", "status": "RUNNING"}
        assert resume.await_count == 1
        assert resume.await_args.args[1][0]["type"] == "respond"

    asyncio.run(scenario())


def test_plan_projection_restored_from_checkpoint_on_resume(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        url = _db_url(tmp_path)
        store = RunStore(url)
        await store.initialize()
        await store.create_if_absent(_deep_request(run_id="run-1"))
        await store.update("run-1", RunStatus.PAUSED)
        await store.checkpoint("run-1", {
            "phase": "planned",
            "tasks": [
                {"id": "task-1", "title": "已提取条款", "status": "completed"},
                {"id": "task-2", "title": "分析风险", "status": "pending"},
            ],
            "planReason": "TOOL_RESULT",
        })
        await store.engine.dispose()

        send = AsyncMock()
        continue_from_checkpoint = AsyncMock(return_value=_result())
        monkeypatch.setattr(CallbackClient, "send_event", send)
        monkeypatch.setattr(DeepAgentExecutor, "continue_from_checkpoint", continue_from_checkpoint)

        settings = _settings(url)
        app = build_application(settings)
        async with app.router.lifespan_context(app):
            response = await _post(app, settings, "/v1/runs/run-1/resume", {})
            assert response.status_code == 202
            await asyncio.sleep(0.3)

        assert continue_from_checkpoint.await_count == 1
        plan_events = [call.args[0] for call in send.await_args_list if call.args[0].event_type == "plan.updated"]
        assert plan_events, "resume should re-publish the durable plan after restart"
        assert plan_events[0].data["reason"] == "RESUME"
        tasks = plan_events[0].data["tasks"]
        assert [task["title"] for task in tasks] == ["已提取条款", "分析风险"]
        assert tasks[0]["status"] == "running"  # 投影中的当前步骤，不会被重放为已完成

    asyncio.run(scenario())


def test_resume_without_decisions_resurfaces_tool_approval(tmp_path, monkeypatch) -> None:
    """暂停命中待审批中间态后，空 decisions 恢复不应报错，而是重新投递审批。"""
    async def scenario() -> None:
        url = _db_url(tmp_path)
        store = RunStore(url)
        await store.initialize()
        await store.create_if_absent(_deep_request(run_id="run-1"))
        await store.save_interaction("run-1", "tool_approval", {
            "actions": [{"name": "http_request", "args": {"url": "https://example.com"}}],
        })
        await store.engine.dispose()

        send = AsyncMock()
        resume = AsyncMock(return_value=_result())
        monkeypatch.setattr(CallbackClient, "send_event", send)
        monkeypatch.setattr(DeepAgentExecutor, "resume", resume)

        settings = _settings(url)
        app = build_application(settings)
        async with app.router.lifespan_context(app):
            response = await _post(app, settings, "/v1/runs/run-1/resume", {})
            assert response.status_code == 202
            await asyncio.sleep(0.2)

        assert response.json()["status"] == "WAITING_APPROVAL"
        assert resume.await_count == 0  # 未用空 decisions 盲目恢复图
        event_types = [call.args[0].event_type for call in send.await_args_list]
        assert "tool.approval.required" in event_types

        restarted = RunStore(url)
        await restarted.initialize()
        interaction = await restarted.take_interaction("run-1")
        await restarted.engine.dispose()
        assert interaction is not None and interaction.interaction_type == "tool_approval"

    asyncio.run(scenario())


def test_resume_without_answers_resurfaces_ask_user(tmp_path, monkeypatch) -> None:
    """ask_user 未提供 answers 时同样重新投递提问而非报错。"""
    async def scenario() -> None:
        url = _db_url(tmp_path)
        store = RunStore(url)
        await store.initialize()
        await store.create_if_absent(_deep_request(run_id="run-1"))
        await store.save_interaction("run-1", "ask_user", {
            "actions": [{"name": "ask_user", "args": {
                "question": "请选择目标文档", "questions": [{"id": "target", "question": "目标文档？"}],
            }}],
        })
        await store.engine.dispose()

        send = AsyncMock()
        resume = AsyncMock(return_value=_result())
        monkeypatch.setattr(CallbackClient, "send_event", send)
        monkeypatch.setattr(DeepAgentExecutor, "resume", resume)

        settings = _settings(url)
        app = build_application(settings)
        async with app.router.lifespan_context(app):
            response = await _post(app, settings, "/v1/runs/run-1/resume", {})
            assert response.status_code == 202
            await asyncio.sleep(0.2)

        assert response.json()["status"] == "WAITING_USER"
        assert resume.await_count == 0
        event_types = [call.args[0].event_type for call in send.await_args_list]
        assert "ask_user.required" in event_types

    asyncio.run(scenario())


def test_resume_plan_approval_continues_execution(tmp_path, monkeypatch) -> None:
    """计划先行确认：用户确认计划后，从检查点继续执行。"""
    async def scenario() -> None:
        url = _db_url(tmp_path)
        store = RunStore(url)
        await store.initialize()
        await store.create_if_absent(_deep_request(run_id="run-1"))
        await store.save_interaction("run-1", "plan_approval", {
            "plan": [{"title": "步骤一"}, {"title": "步骤二"}],
        })
        await store.engine.dispose()

        send = AsyncMock()
        execute = AsyncMock(return_value=_result())
        monkeypatch.setattr(CallbackClient, "send_event", send)
        monkeypatch.setattr(DeepAgentExecutor, "execute", execute)

        settings = _settings(url)
        app = build_application(settings)
        async with app.router.lifespan_context(app):
            response = await _post(app, settings, "/v1/runs/run-1/resume", {"plan_approved": True})
            assert response.status_code == 202
            await asyncio.sleep(0.3)

        assert response.json()["status"] == "RUNNING"
        assert execute.await_count == 1  # 计划批准后直接开始执行（无图检查点）

    asyncio.run(scenario())
