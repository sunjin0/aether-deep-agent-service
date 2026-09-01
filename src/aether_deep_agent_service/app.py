import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from .callbacks import CallbackClient
from .executor import DeepAgentExecutor, ExecutionResult, PendingApproval, PendingUserQuestion
from .schemas import (CallbackEvent, DeepRunRequest, DeepRunResponse, DeepRunStatusResponse,
                      DeepSessionStatusResponse, ResumeRunRequest, RunStatus)
from .security import verify_request_signature
from .settings import Settings, get_settings
from .store import RunStore
from .telemetry import OTelMiddleware, configure_tracing, shutdown_tracing


logger = logging.getLogger(__name__)


def update_task_plan(tasks: list[dict[str, str]], active_index: int | None = None,
                     completed: bool = False) -> list[dict[str, str]]:
    """复制模型生成的计划，并写入当前执行状态。"""
    result = [dict(task) for task in tasks]
    for index, task in enumerate(result):
        task["status"] = "completed" if completed or (active_index is not None and index < active_index) else (
            "running" if index == active_index else "pending"
        )
    return result


def todos_to_tasks(todos: list) -> list[dict[str, str]]:
    """把 write_todos 的 todos（{content,status}，小写状态）映射为计划 tasks（大写状态）。

    覆盖语义：最新一次 write_todos 就是模型当前的执行计划。
    """
    status_map = {"pending": "PENDING", "in_progress": "RUNNING", "completed": "COMPLETED"}
    tasks: list[dict[str, str]] = []
    for index, item in enumerate(todos, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("content") or "").strip()
        if not title:
            continue
        raw_status = str(item.get("status") or "pending").strip().lower()
        tasks.append({
            "id": f"todo-{index}",
            "title": title,
            "status": status_map.get(raw_status, "PENDING"),
        })
    return tasks


def normalize_ask_user_payload(payload: dict) -> dict:
    """将模型生成的问题规范为 Dashboard 可选择的问题。

    ``ask_user`` 用于收集缺失的业务信息，而非批准工具调用。因此始终提供选项和
    自定义输入框；工具审批仍通过独立的确认交互处理。
    """
    normalized = []
    for index, raw in enumerate(payload.get("questions") or []):
        if not isinstance(raw, dict) or not str(raw.get("question") or "").strip():
            continue
        question = {"id": str(raw.get("id") or f"question_{index + 1}"), "question": str(raw["question"]).strip()}
        options = raw.get("options") if isinstance(raw.get("options"), list) else []
        normalized_options = []
        used_values = set()
        for option_index, option in enumerate(options):
            if isinstance(option, dict):
                value = str(option.get("value") or option.get("id") or option.get("text") or option.get("label") or "").strip()
                label = str(option.get("label") or option.get("text") or option.get("name") or value).strip()
            else:
                value = label = str(option).strip()
            if not value or value in used_values:
                value = f"option_{option_index + 1}"
            if not label:
                label = f"选项 {option_index + 1}"
            used_values.add(value)
            normalized_options.append({"id": value, "label": label, "value": value})
        if not normalized_options:
            normalized_options = [
                {"id": "provide_details", "label": "提供具体信息", "value": "provide_details"},
                {"id": "not_available", "label": "暂无相关信息", "value": "not_available"},
            ]
        question.update({
            "type": "choice",
            "options": normalized_options,
            "multiple": bool(raw.get("multiple")),
            "allowCustomInput": True,
            "customInputPlaceholder": str(raw.get("customInputPlaceholder") or "请输入具体信息"),
        })
        normalized.append(question)
    return {"question": str(payload.get("question") or "请补充以下信息后继续"), "questions": normalized}


def build_ask_user_response_decisions(answers: dict) -> list[dict[str, str]]:
    """构造将用户答案传回暂停的 ``ask_user`` 调用的人工介入响应。"""
    answer_json = json.dumps(answers or {}, ensure_ascii=False)
    return [{
        "type": "respond",
        "message": (
            "The user has answered the ask_user questions. Treat the following JSON as authoritative "
            "and continue the original task. Do not ask the same questions again.\n"
            f"User answers: {answer_json}"
        ),
    }]


def resolve_run_timeout(payload: DeepRunRequest, settings: Settings) -> int:
    """确定本次运行使用的超时时间，优先采用请求中的显式配置。"""
    return payload.timeout_seconds if payload.timeout_seconds is not None else settings.run_timeout_seconds


def graph_checkpoint_url(database_url: str) -> str:
    """将 SQLAlchemy 的 asyncpg URL 转为 LangGraph 使用的 psycopg URL。"""
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def build_application(settings: Settings | None = None) -> FastAPI:
    """构建并配置 Deep Agent 的 FastAPI 应用实例。"""
    resolved_settings = settings or get_settings()
    store = RunStore(resolved_settings.database_url)
    callbacks = CallbackClient(resolved_settings)
    executor = DeepAgentExecutor(resolved_settings, callbacks)
    tasks: dict[str, asyncio.Task] = {}
    pending_approvals: dict[str, PendingApproval] = {}
    pending_questions: dict[str, PendingUserQuestion] = {}
    task_plans: dict[str, list[dict[str, str]]] = {}
    checkpointer = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        """初始化存储与检查点，并在启动时恢复未投递事件。"""
        nonlocal checkpointer
        await store.initialize()
        checkpoint_context = None
        if resolved_settings.database_url.startswith("postgresql"):
            checkpoint_context = AsyncPostgresSaver.from_conn_string(graph_checkpoint_url(resolved_settings.database_url))
            checkpointer = await checkpoint_context.__aenter__()
            await checkpointer.setup()
        else:
            # SQLite 仅用于轻量单元测试；部署环境必须使用 PostgreSQL。
            checkpointer = InMemorySaver()
        if hasattr(store, "pause_incomplete_runs"):
            for interrupted_run_id in await store.pause_incomplete_runs():
                await store.checkpoint(interrupted_run_id, {"phase": "paused", "reason": "service_restart"})
        if hasattr(store, "pending_callbacks"):
            for event in await store.pending_callbacks():
                try:
                    await callbacks.send_event(CallbackEvent(event_id=event.event_id, run_id=event.run_id, event_type=event.event_type, occurred_at=event.occurred_at, data=event.data))
                    await store.mark_callback_delivered(event.event_id)
                except Exception:
                    logger.exception("Failed to replay callback %s", event.event_id)
        try:
            yield
        finally:
            for task in tasks.values():
                task.cancel()
            if checkpoint_context is not None:
                await checkpoint_context.__aexit__(None, None, None)
            shutdown_tracing()

    app = FastAPI(title="Aether Deep Agent Service", version="0.1.0", lifespan=lifespan)
    configure_tracing()
    app.add_middleware(OTelMiddleware)

    async def authenticated(request: Request) -> None:
        """验证调用方的 HMAC 请求签名。"""
        await verify_request_signature(request, resolved_settings)

    async def safe_callback(run_id: str, event_type: str, data: dict) -> None:
        """持久化并投递回调事件；投递失败仅记录日志，不中断运行。"""
        event = CallbackEvent(event_id=str(uuid.uuid4()), event_type=event_type, run_id=run_id, occurred_at=int(time.time() * 1000), data=data)
        try:
            if hasattr(store, "enqueue_callback"):
                await store.enqueue_callback(event.event_id, run_id, event_type, data, event.occurred_at)
                await callbacks.send_event(event)
                await store.mark_callback_delivered(event.event_id)
            else:
                await callbacks.send(run_id, event_type, data)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to deliver callback %s for run %s", event_type, run_id)

    async def finish_execution(request: DeepRunRequest, result: ExecutionResult | PendingApproval) -> None:
        """处理执行结果，并持久化或通知等待中的人工交互。"""
        if isinstance(result, PendingApproval):
            if result.actions and result.actions[0].get("name") == "ask_user":
                pending_questions[request.run_id] = PendingUserQuestion(request, result.actions)
                if hasattr(store, "save_interaction"):
                    await store.save_interaction(request.run_id, "ask_user", {"actions": result.actions})
                await safe_callback(request.run_id, "ask_user.required", normalize_ask_user_payload(result.actions[0].get("args") or {}))
                return
            pending_approvals[request.run_id] = result
            if hasattr(store, "save_interaction"):
                await store.save_interaction(request.run_id, "tool_approval", {"actions": result.actions})
            await safe_callback(request.run_id, "tool.approval.required", {"actions": result.actions})
            return
        pending_approvals.pop(request.run_id, None)
        if hasattr(store, "take_interaction"):
            await store.take_interaction(request.run_id)
        await store.update(request.run_id, RunStatus.SUCCEEDED, result=result.content)
        await safe_callback(request.run_id, "run.completed", {
            "content": result.content, "citations": result.citations,
            "model": result.model, "tools": result.tools,
            "promptTokens": result.prompt_tokens, "completionTokens": result.completion_tokens,
            "totalTokens": ((result.prompt_tokens or 0) + (result.completion_tokens or 0)) or None,
        })

    async def run(request: DeepRunRequest, pending: PendingApproval | PendingUserQuestion | None = None,
                  decisions: list[dict] | None = None, skip_plan: bool = False, plan_approved: bool = False,
                  skip_analysis: bool = False, plan_feedback: str | None = None) -> None:
        """执行、恢复或重规划一次 Deep Agent 运行，并维护计划投影。"""
        task_plan = task_plans.get(request.run_id, [])
        # write_todos 去重：仅当模型更新了 todos 才覆盖发布计划，避免每个计划步骤都产生新版本。
        last_todos_json: str = ""
        # 内存任务计划只是投递缓存；服务重启后恢复最新持久化投影，避免恢复或完成时
        # 丢失用户可见的计划。
        if hasattr(store, "latest_checkpoint"):
            checkpoint = await store.latest_checkpoint(request.run_id)
            if checkpoint is not None and isinstance(checkpoint.state, dict):
                # 方案文档与复杂度始终从 checkpoint 恢复，保证审批/重启后发布的
                # RESUME 计划仍携带规范 §3 文档与正确的复杂度判断。
                if checkpoint.state.get("document"):
                    request.task_state["document"] = checkpoint.state["document"]
                if "complex" in checkpoint.state:
                    request.task_state["complex"] = checkpoint.state["complex"]
                if not task_plan:
                    checkpoint_tasks = checkpoint.state.get("tasks")
                    if isinstance(checkpoint_tasks, list):
                        task_plan = [dict(item) for item in checkpoint_tasks if isinstance(item, dict)]
                        if task_plan:
                            task_plans[request.run_id] = task_plan
                            request.task_state["plan"] = task_plan
                            request.task_state["plan_reason"] = checkpoint.state.get("planReason", "RESUME")

        async def publish_plan(reason: str, summary: str, plan: list[dict[str, str]],
                               active_index: int | None = 0, completed: bool = False,
                               preserve_status: bool = False) -> None:
            """保存任务计划投影，并向调用方发布计划更新事件。"""
            projected = ([dict(task) for task in plan]
                         if preserve_status else update_task_plan(plan, active_index=active_index, completed=completed))
            task_plans[request.run_id] = projected
            request.task_state["plan"] = projected
            request.task_state["plan_reason"] = reason
            if hasattr(store, "checkpoint"):
                await store.checkpoint(request.run_id, {
                    "phase": "replanned" if reason not in {"INITIAL", "COMPLETED"} else "planned",
                    "tasks": projected,
                    "currentStep": active_index or 0,
                    "planReason": reason,
                    "document": (request.task_state or {}).get("document"),
                    "complex": bool((request.task_state or {}).get("complex")),
                })
            plan_payload: dict[str, object] = {
                "reason": reason, "summary": summary, "maxSteps": request.max_steps,
                "tasks": projected, "complex": bool((request.task_state or {}).get("complex")),
            }
            if (request.task_state or {}).get("document"):
                plan_payload["document"] = request.task_state["document"]
            await safe_callback(request.run_id, "plan.updated", plan_payload)

        async def emit_runtime_event(event_type: str, data: dict) -> None:
            """转发执行期事件，并在工具结果改变路径时更新计划。"""
            nonlocal task_plan, last_todos_json
            if event_type == "todos.updated":
                # write_todos 是模型最新的执行计划：覆盖投影并发布 plan.updated（去重）。
                synced = todos_to_tasks(data.get("todos") or [])
                if synced:
                    snapshot = json.dumps(synced, ensure_ascii=False)
                    if snapshot != last_todos_json:
                        last_todos_json = snapshot
                        task_plan = synced
                        await publish_plan("TOOL_RESULT", "已根据执行进度更新计划", synced, preserve_status=True)
                return
            await safe_callback(request.run_id, event_type, data)
            # write_todos 已由 todos.updated 分支同步计划，工具完成事件不再触发模型重规划。
            if event_type in {"tool.completed", "tool.failed"} and data.get("toolName") == "write_todos":
                return
            # 工具观察结果可能改变可行路径。图决定下一步操作前先重规划；计划投影本身
            # 无权重放会产生副作用的工具调用。
            if event_type in {"tool.completed", "tool.failed"} and task_plan:
                failed = event_type == "tool.failed"
                reason = "STEP_FAILED" if failed else "TOOL_RESULT"
                observation = str(data.get("error") or data.get("outputSummary") or data.get("message")
                                  or ("工具调用失败" if failed else "工具调用完成"))
                task_plan = await executor.replan(request, task_plan, reason, observation)
                await publish_plan(reason, "工具调用失败，已调整后续计划" if failed else "已根据工具结果调整后续计划", task_plan)

        if pending is None:
            await store.update(request.run_id, RunStatus.RUNNING)
            if hasattr(store, "checkpoint"):
                await store.checkpoint(request.run_id, {"phase": "running", "task": request.task})
        try:
            if plan_feedback:
                # 用户反馈方案：按反馈重新生成方案文档与步骤，重新提交审批（再次批准后才执行）。
                complex_task, document = await executor.plan_document(request, feedback=plan_feedback)
                task_plan = await executor.plan(request)
                await publish_plan("USER_INPUT", "已根据用户反馈调整方案", task_plan)
                approval_payload: dict[str, object] = {"complex": bool(complex_task), "plan": task_plan}
                if document:
                    approval_payload["document"] = document
                await store.save_interaction(request.run_id, "plan_approval", approval_payload)
                await safe_callback(request.run_id, "plan.approval.required", approval_payload)
                return
            if plan_approved:
                # 规划文档已批准：复用已发布/已检查点的任务规划，避免重新生成与审批结果不一致的步骤。
                if not task_plan:
                    task_plan = await executor.plan(request)
                await publish_plan("RESUME", "规划文档已批准，生成任务规划", task_plan)
                result = await executor.execute(request, emit_runtime_event, checkpointer)
            elif pending is None and not skip_plan:
                await safe_callback(request.run_id, "run.started", {"status": RunStatus.RUNNING})
                # 需求分析：先检查用户问题是否信息完整，缺失则让用户补充，不盲目生成规划。
                if not skip_analysis:
                    missing = await executor.analyze_requirements(request)
                    if missing:
                        await store.save_interaction(request.run_id, "requirement_analysis", {"questions": missing})
                        await safe_callback(
                            request.run_id, "ask_user.required",
                            normalize_ask_user_payload({"question": "开始前请先补充以下信息", "questions": missing}),
                        )
                        return
                # 先生成规划文档（方案说明 + 步骤），先发布 INITIAL 计划供展示；
                # 复杂任务再暂停进入计划审批，审批通过后才开始执行。
                complex_task, document = await executor.plan_document(request)
                task_plan = await executor.plan(request)
                await publish_plan("INITIAL", "Task plan created", task_plan)
                if request.plan_approval_required and complex_task:
                    approval_payload: dict[str, object] = {"complex": True, "plan": task_plan}
                    if document:
                        approval_payload["document"] = document
                    await store.save_interaction(request.run_id, "plan_approval", approval_payload)
                    await safe_callback(request.run_id, "plan.approval.required", approval_payload)
                    return
                result = await executor.execute(request, emit_runtime_event, checkpointer)
            elif pending is None:
                if task_plan:
                    await publish_plan("RESUME", "从最近检查点继续执行", task_plan)
                result = await executor.continue_from_checkpoint(request, emit_runtime_event, checkpointer)
            else:
                if isinstance(pending, PendingUserQuestion) and task_plan:
                    observation = str((decisions or [{}])[0].get("message") or "用户已补充信息")
                    task_plan = await executor.replan(request, task_plan, "USER_INPUT", observation)
                    await publish_plan("USER_INPUT", "已根据用户补充信息调整计划", task_plan)
                result = await executor.resume(request, decisions or [], emit_runtime_event, checkpointer)
            if not isinstance(result, PendingApproval):
                task_plan = task_plans.pop(request.run_id, task_plans.get(request.run_id, []))
                if task_plan:
                    await publish_plan("COMPLETED", "Task plan completed", task_plan, completed=True)
            await finish_execution(request, result)
        except asyncio.CancelledError:
            record = await store.get(request.run_id) if hasattr(store, "get") else None
            paused = record is not None and record.pause_requested
            await store.update(request.run_id, RunStatus.PAUSED if paused else RunStatus.CANCELLED)
            if hasattr(store, "checkpoint"):
                await store.checkpoint(request.run_id, {"phase": "paused" if paused else "cancelled", "tasks": task_plans.get(request.run_id, [])})
            await safe_callback(request.run_id, "run.paused" if paused else "run.cancelled", {})
            raise
        except Exception as error:
            message = str(error)
            logger.exception("Deep run %s failed", request.run_id)
            await store.update(request.run_id, RunStatus.FAILED, error=message)
            task_plans.pop(request.run_id, None)
            await safe_callback(request.run_id, "run.failed", {"error": message})

    @app.get("/health")
    async def health() -> dict[str, str]:
        """返回服务存活状态。"""
        return {"status": "ok"}

    @app.post("/v1/runs", response_model=DeepRunResponse, status_code=status.HTTP_202_ACCEPTED,
              dependencies=[Depends(authenticated)])
    async def create_run(payload: DeepRunRequest) -> DeepRunResponse:
        """创建幂等运行记录，并异步启动新任务。"""
        payload.timeout_seconds = resolve_run_timeout(payload, resolved_settings)
        record, created = await store.create_if_absent(payload)
        if created:
            tasks[payload.run_id] = asyncio.create_task(run(payload))
        return DeepRunResponse(run_id=record.run_id, status=RunStatus(record.status), created=created)

    @app.post("/v1/sessions/{session_id}/tasks", response_model=DeepRunResponse,
              status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(authenticated)])
    async def create_session_task(session_id: str, payload: DeepRunRequest) -> DeepRunResponse:
        """面向新客户端的会话级别别名；旧版 Admin 调用方仍使用 ``/v1/runs``。"""
        if payload.session_id is not None and payload.session_id != session_id:
            raise HTTPException(status_code=422, detail="session_id does not match request path")
        return await create_run(payload.model_copy(update={"session_id": session_id}))

    @app.post("/v1/runs/{run_id}/cancel", status_code=status.HTTP_202_ACCEPTED,
              dependencies=[Depends(authenticated)])
    async def cancel_run(run_id: str) -> dict[str, str]:
        """取消正在执行的运行，或将尚未结束的记录标为已取消。"""
        record = await store.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        task = tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
        elif record.status not in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}:
            await store.update(run_id, RunStatus.CANCELLED)
        return {"runId": run_id, "status": "CANCELLED"}

    @app.post("/v1/runs/{run_id}/pause", status_code=status.HTTP_202_ACCEPTED,
              dependencies=[Depends(authenticated)])
    async def pause_run(run_id: str) -> dict[str, str]:
        """请求暂停运行并取消当前异步任务以保留恢复点。"""
        record = await store.request_pause(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        task = tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
        else:
            await store.update(run_id, RunStatus.PAUSED)
        return {"runId": run_id, "status": "PAUSED"}

    @app.get("/v1/runs/{run_id}", response_model=DeepRunStatusResponse,
             dependencies=[Depends(authenticated)])
    async def get_run(run_id: str) -> DeepRunStatusResponse:
        """查询指定运行及其最新检查点编号。"""
        record, checkpoint_no = await store.status(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return DeepRunStatusResponse(run_id=run_id, status=RunStatus(record.status), checkpoint_no=checkpoint_no, updated_at=record.updated_at)

    @app.get("/v1/sessions/{session_id}", response_model=DeepSessionStatusResponse,
             dependencies=[Depends(authenticated)])
    async def get_session(session_id: str) -> DeepSessionStatusResponse:
        """查询会话最近一次任务的持久化状态。"""
        record = await store.latest_for_session(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="session not found")
        _, checkpoint_no = await store.status(record.run_id)
        return DeepSessionStatusResponse(
            session_id=session_id, task_id=record.task_id, run_id=record.run_id,
            status=RunStatus(record.status), checkpoint_no=checkpoint_no, updated_at=record.updated_at,
        )

    async def resume_existing_run(run_id: str, payload: ResumeRunRequest) -> dict[str, str]:
        """根据持久化交互或检查点恢复已有运行。"""
        record = await store.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")

        async def resurface_interaction(interaction_type: str, actions: list[dict]) -> dict[str, str]:
            """保持等待状态并重新投递审批/提问，而不是用空 decisions 盲目恢复图。

            暂停可能命中"待审批工具调用"的中间态；此时用户应先回答而非继续，
            因此重新持久化交互并通知前端，避免 0 decisions vs N hanging tool calls。
            """
            await store.save_interaction(run_id, interaction_type, {"actions": actions})
            if interaction_type == "ask_user":
                first = actions[0] if actions else {}
                await safe_callback(run_id, "ask_user.required", normalize_ask_user_payload(first.get("args") or {}))
                return {"runId": run_id, "status": "WAITING_USER"}
            await safe_callback(run_id, "tool.approval.required", {"actions": actions})
            return {"runId": run_id, "status": "WAITING_APPROVAL"}

        question = pending_questions.pop(run_id, None)
        interaction = None if question is not None else await store.take_interaction(run_id)
        if question is not None:
            if not payload.answers:
                return await resurface_interaction("ask_user", question.actions)
            await store.update(run_id, RunStatus.RUNNING)
            tasks[run_id] = asyncio.create_task(
                run(question.request, question, build_ask_user_response_decisions(payload.answers)),
            )
            return {"runId": run_id, "status": "RUNNING"}
        pending = pending_approvals.pop(run_id, None)
        if interaction is not None and interaction.interaction_type == "ask_user":
            if not payload.answers:
                return await resurface_interaction("ask_user", interaction.payload.get("actions", []))
            resumed = DeepRunRequest.model_validate(record.request)
            restored_question = PendingUserQuestion(resumed, interaction.payload.get("actions", []))
            await store.update(run_id, RunStatus.RUNNING)
            tasks[run_id] = asyncio.create_task(
                run(resumed, restored_question, build_ask_user_response_decisions(payload.answers)),
            )
            return {"runId": run_id, "status": "RUNNING"}
        if pending is not None or (interaction is not None and interaction.interaction_type == "tool_approval"):
            actions = interaction.payload.get("actions", []) if interaction is not None else pending.actions
            if not payload.decisions:
                return await resurface_interaction("tool_approval", actions)
            if pending is None:
                resumed = DeepRunRequest.model_validate(record.request)
                pending = PendingApproval(resumed, None, {}, None, actions, resumed.timeout_seconds or 0, "")
            await store.update(run_id, RunStatus.RUNNING)
            tasks[run_id] = asyncio.create_task(run(pending.request, pending, payload.decisions))
            return {"runId": run_id, "status": "RUNNING"}
        if interaction is not None and interaction.interaction_type == "requirement_analysis":
            # 用户已补充需求分析缺的信息：跳过分析，继续生成规划。
            resumed = DeepRunRequest.model_validate(record.request)
            await store.update(run_id, RunStatus.RUNNING)
            tasks[run_id] = asyncio.create_task(run(resumed, skip_analysis=True))
            return {"runId": run_id, "status": "RUNNING"}
        if interaction is not None and interaction.interaction_type == "plan_approval":
            # 审批门是 Python 层暂停，图上无检查点。批准则直接开始执行；
            # 携带方案反馈则按反馈重规划并重新提交审批。
            resumed = DeepRunRequest.model_validate(record.request)
            await store.update(run_id, RunStatus.RUNNING)
            if payload.plan_feedback:
                tasks[run_id] = asyncio.create_task(run(resumed, plan_feedback=payload.plan_feedback))
            else:
                tasks[run_id] = asyncio.create_task(run(resumed, plan_approved=True))
            return {"runId": run_id, "status": "RUNNING"}
        if record.status == RunStatus.PAUSED:
            resumed = DeepRunRequest.model_validate(record.request)
            tasks[run_id] = asyncio.create_task(run(resumed, skip_plan=True))
            return {"runId": run_id, "status": "RUNNING"}
        raise HTTPException(status_code=409, detail="run is not resumable")

    @app.post("/v1/runs/{run_id}/resume", status_code=status.HTTP_202_ACCEPTED,
              dependencies=[Depends(authenticated)])
    async def resume_run(run_id: str, payload: ResumeRunRequest) -> dict[str, str]:
        """恢复指定运行。"""
        return await resume_existing_run(run_id, payload)

    @app.post("/v1/sessions/{session_id}/tasks/{task_id}/resume", status_code=status.HTTP_202_ACCEPTED,
              dependencies=[Depends(authenticated)])
    async def resume_session_task(session_id: str, task_id: str, payload: ResumeRunRequest) -> dict[str, str]:
        """按会话和任务标识定位并恢复最近一次运行。"""
        record = await store.latest_for_session(session_id, task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="task not found in session")
        return await resume_existing_run(record.run_id, payload)

    return app


app = build_application()
