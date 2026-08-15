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


logger = logging.getLogger(__name__)


def update_task_plan(tasks: list[dict[str, str]], active_index: int | None = None,
                     completed: bool = False) -> list[dict[str, str]]:
    """Return a copy of a model-generated plan with the current execution status applied."""
    result = [dict(task) for task in tasks]
    for index, task in enumerate(result):
        task["status"] = "completed" if completed or (active_index is not None and index < active_index) else (
            "running" if index == active_index else "pending"
        )
    return result


def normalize_ask_user_payload(payload: dict) -> dict:
    """Normalize model-generated questions to selectable dashboard questions.

    ``ask_user`` is used to collect missing business information, not to approve a
    tool call.  It therefore always exposes choices together with a custom input
    field.  Tool approvals remain a separate confirm interaction.
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
    """Return the HITL response that feeds answers into the suspended ask_user call."""
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
    return payload.timeout_seconds if payload.timeout_seconds is not None else settings.run_timeout_seconds


def graph_checkpoint_url(database_url: str) -> str:
    """Convert SQLAlchemy's asyncpg URL to the psycopg URL used by LangGraph."""
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def build_application(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    store = RunStore(resolved_settings.database_url)
    callbacks = CallbackClient(resolved_settings)
    executor = DeepAgentExecutor(resolved_settings)
    tasks: dict[str, asyncio.Task] = {}
    pending_approvals: dict[str, PendingApproval] = {}
    pending_questions: dict[str, PendingUserQuestion] = {}
    task_plans: dict[str, list[dict[str, str]]] = {}
    checkpointer = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal checkpointer
        await store.initialize()
        checkpoint_context = None
        if resolved_settings.database_url.startswith("postgresql"):
            checkpoint_context = AsyncPostgresSaver.from_conn_string(graph_checkpoint_url(resolved_settings.database_url))
            checkpointer = await checkpoint_context.__aenter__()
            await checkpointer.setup()
        else:
            # SQLite is retained solely for lightweight unit tests; deployed services require PostgreSQL.
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

    app = FastAPI(title="Aether Deep Agent Service", version="0.1.0", lifespan=lifespan)

    async def authenticated(request: Request) -> None:
        await verify_request_signature(request, resolved_settings)

    async def safe_callback(run_id: str, event_type: str, data: dict) -> None:
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
                  decisions: list[dict] | None = None, skip_plan: bool = False) -> None:
        task_plan = task_plans.get(request.run_id, [])
        # The in-memory task plan is merely a delivery cache. Restore the latest
        # durable projection after a service restart so resume/completion does
        # not discard the user's visible plan.
        if not task_plan and hasattr(store, "latest_checkpoint"):
            checkpoint = await store.latest_checkpoint(request.run_id)
            checkpoint_tasks = checkpoint.state.get("tasks") if checkpoint is not None and isinstance(checkpoint.state, dict) else None
            if isinstance(checkpoint_tasks, list):
                task_plan = [dict(item) for item in checkpoint_tasks if isinstance(item, dict)]
                if task_plan:
                    task_plans[request.run_id] = task_plan
                    request.task_state["plan"] = task_plan
                    request.task_state["plan_reason"] = checkpoint.state.get("planReason", "RESUME")

        async def publish_plan(reason: str, summary: str, plan: list[dict[str, str]],
                               active_index: int | None = 0, completed: bool = False) -> None:
            projected = update_task_plan(plan, active_index=active_index, completed=completed)
            task_plans[request.run_id] = projected
            request.task_state["plan"] = projected
            request.task_state["plan_reason"] = reason
            if hasattr(store, "checkpoint"):
                await store.checkpoint(request.run_id, {
                    "phase": "replanned" if reason not in {"INITIAL", "COMPLETED"} else "planned",
                    "tasks": projected,
                    "currentStep": active_index or 0,
                    "planReason": reason,
                })
            await safe_callback(request.run_id, "plan.updated", {
                "reason": reason, "summary": summary, "maxSteps": request.max_steps,
                "tasks": projected,
            })

        async def emit_runtime_event(event_type: str, data: dict) -> None:
            nonlocal task_plan
            await safe_callback(request.run_id, event_type, data)
            # Tool observations can change the viable path. Replan before the
            # graph decides its next action; the plan projection itself has no
            # authority to replay a side-effecting tool.
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
            if pending is None and not skip_plan:
                await safe_callback(request.run_id, "run.started", {"status": RunStatus.RUNNING})
                task_plan = await executor.plan(request)
                await publish_plan("INITIAL", "Task plan created", task_plan)
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
        return {"status": "ok"}

    @app.post("/v1/runs", response_model=DeepRunResponse, status_code=status.HTTP_202_ACCEPTED,
              dependencies=[Depends(authenticated)])
    async def create_run(payload: DeepRunRequest) -> DeepRunResponse:
        payload.timeout_seconds = resolve_run_timeout(payload, resolved_settings)
        record, created = await store.create_if_absent(payload)
        if created:
            tasks[payload.run_id] = asyncio.create_task(run(payload))
        return DeepRunResponse(run_id=record.run_id, status=RunStatus(record.status), created=created)

    @app.post("/v1/sessions/{session_id}/tasks", response_model=DeepRunResponse,
              status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(authenticated)])
    async def create_session_task(session_id: str, payload: DeepRunRequest) -> DeepRunResponse:
        """Session-scoped alias for new clients; legacy Admin callers keep using /v1/runs."""
        if payload.session_id is not None and payload.session_id != session_id:
            raise HTTPException(status_code=422, detail="session_id does not match request path")
        return await create_run(payload.model_copy(update={"session_id": session_id}))

    @app.post("/v1/runs/{run_id}/cancel", status_code=status.HTTP_202_ACCEPTED,
              dependencies=[Depends(authenticated)])
    async def cancel_run(run_id: str) -> dict[str, str]:
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
        record, checkpoint_no = await store.status(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return DeepRunStatusResponse(run_id=run_id, status=RunStatus(record.status), checkpoint_no=checkpoint_no, updated_at=record.updated_at)

    @app.get("/v1/sessions/{session_id}", response_model=DeepSessionStatusResponse,
             dependencies=[Depends(authenticated)])
    async def get_session(session_id: str) -> DeepSessionStatusResponse:
        record = await store.latest_for_session(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="session not found")
        _, checkpoint_no = await store.status(record.run_id)
        return DeepSessionStatusResponse(
            session_id=session_id, task_id=record.task_id, run_id=record.run_id,
            status=RunStatus(record.status), checkpoint_no=checkpoint_no, updated_at=record.updated_at,
        )

    async def resume_existing_run(run_id: str, payload: ResumeRunRequest) -> dict[str, str]:
        record = await store.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        question = pending_questions.pop(run_id, None)
        interaction = None if question is not None else await store.take_interaction(run_id)
        if question is not None:
            await store.update(run_id, RunStatus.RUNNING)
            tasks[run_id] = asyncio.create_task(
                run(question.request, question, build_ask_user_response_decisions(payload.answers)),
            )
            return {"runId": run_id, "status": "RUNNING"}
        pending = pending_approvals.pop(run_id, None)
        if interaction is not None and interaction.interaction_type == "ask_user":
            resumed = DeepRunRequest.model_validate(record.request)
            restored_question = PendingUserQuestion(resumed, interaction.payload.get("actions", []))
            await store.update(run_id, RunStatus.RUNNING)
            tasks[run_id] = asyncio.create_task(
                run(resumed, restored_question, build_ask_user_response_decisions(payload.answers)),
            )
            return {"runId": run_id, "status": "RUNNING"}
        if pending is not None or (interaction is not None and interaction.interaction_type == "tool_approval"):
            if pending is None:
                resumed = DeepRunRequest.model_validate(record.request)
                pending = PendingApproval(resumed, None, {}, None, interaction.payload.get("actions", []), resumed.timeout_seconds or 0, "")
            await store.update(run_id, RunStatus.RUNNING)
            tasks[run_id] = asyncio.create_task(run(pending.request, pending, payload.decisions))
            return {"runId": run_id, "status": "RUNNING"}
        if record.status == RunStatus.PAUSED:
            resumed = DeepRunRequest.model_validate(record.request)
            tasks[run_id] = asyncio.create_task(run(resumed, skip_plan=True))
            return {"runId": run_id, "status": "RUNNING"}
        raise HTTPException(status_code=409, detail="run is not resumable")

    @app.post("/v1/runs/{run_id}/resume", status_code=status.HTTP_202_ACCEPTED,
              dependencies=[Depends(authenticated)])
    async def resume_run(run_id: str, payload: ResumeRunRequest) -> dict[str, str]:
        return await resume_existing_run(run_id, payload)

    @app.post("/v1/sessions/{session_id}/tasks/{task_id}/resume", status_code=status.HTTP_202_ACCEPTED,
              dependencies=[Depends(authenticated)])
    async def resume_session_task(session_id: str, task_id: str, payload: ResumeRunRequest) -> dict[str, str]:
        record = await store.latest_for_session(session_id, task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="task not found in session")
        return await resume_existing_run(record.run_id, payload)

    return app


app = build_application()
