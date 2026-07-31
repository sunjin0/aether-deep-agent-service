import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status

from .callbacks import CallbackClient
from .executor import DeepAgentExecutor, ExecutionResult, PendingApproval, PendingUserQuestion
from .schemas import DeepRunRequest, DeepRunResponse, ResumeRunRequest, RunStatus
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
    """Normalize model-generated questions to the dashboard's interaction schema."""
    normalized = []
    for index, raw in enumerate(payload.get("questions") or []):
        if not isinstance(raw, dict) or not str(raw.get("question") or "").strip():
            continue
        question = {"id": str(raw.get("id") or f"question_{index + 1}"), "question": str(raw["question"]).strip()}
        options = raw.get("options") if isinstance(raw.get("options"), list) else []
        if str(raw.get("type") or "").lower() == "choice" and options:
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
            if normalized_options:
                question.update({"type": "choice", "options": normalized_options, "multiple": bool(raw.get("multiple")), "allowCustomInput": True})
            else:
                question.update({"type": "confirm", "confirmText": "确认", "cancelText": "取消"})
        else:
            question.update({"type": "confirm", "confirmText": str(raw.get("confirmText") or "确认"), "cancelText": str(raw.get("cancelText") or "取消")})
        normalized.append(question)
    return {"question": str(payload.get("question") or "请补充以下信息后继续"), "questions": normalized}


def resolve_run_timeout(payload: DeepRunRequest, settings: Settings) -> int:
    return payload.timeout_seconds if payload.timeout_seconds is not None else settings.run_timeout_seconds


def build_application(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    store = RunStore(resolved_settings.database_url)
    callbacks = CallbackClient(resolved_settings)
    executor = DeepAgentExecutor(resolved_settings)
    tasks: dict[str, asyncio.Task] = {}
    pending_approvals: dict[str, PendingApproval] = {}
    pending_questions: dict[str, PendingUserQuestion] = {}
    task_plans: dict[str, list[dict[str, str]]] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await store.initialize()
        yield
        for task in tasks.values():
            task.cancel()

    app = FastAPI(title="Aether Deep Agent Service", version="0.1.0", lifespan=lifespan)

    async def authenticated(request: Request) -> None:
        await verify_request_signature(request, resolved_settings)

    async def safe_callback(run_id: str, event_type: str, data: dict) -> None:
        try:
            await callbacks.send(run_id, event_type, data)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to deliver callback %s for run %s", event_type, run_id)

    async def finish_execution(request: DeepRunRequest, result: ExecutionResult | PendingApproval) -> None:
        if isinstance(result, PendingApproval):
            if result.actions and result.actions[0].get("name") == "ask_user":
                pending_questions[request.run_id] = PendingUserQuestion(request, result.actions)
                await safe_callback(request.run_id, "ask_user.required", normalize_ask_user_payload(result.actions[0].get("args") or {}))
                return
            pending_approvals[request.run_id] = result
            await safe_callback(request.run_id, "tool.approval.required", {"actions": result.actions})
            return
        pending_approvals.pop(request.run_id, None)
        await store.update(request.run_id, RunStatus.SUCCEEDED, result=result.content)
        await safe_callback(request.run_id, "run.completed", {
            "content": result.content, "citations": result.citations,
            "model": result.model, "tools": result.tools,
            "promptTokens": result.prompt_tokens, "completionTokens": result.completion_tokens,
            "totalTokens": ((result.prompt_tokens or 0) + (result.completion_tokens or 0)) or None,
        })

    async def run(request: DeepRunRequest, pending: PendingApproval | None = None,
                  decisions: list[dict] | None = None, skip_plan: bool = False) -> None:
        if pending is None:
            await store.update(request.run_id, RunStatus.RUNNING)
        try:
            if pending is None and not skip_plan:
                await safe_callback(request.run_id, "run.started", {"status": RunStatus.RUNNING})
                task_plan = await executor.plan(request)
                task_plans[request.run_id] = task_plan
                await safe_callback(request.run_id, "plan.updated", {
                    "summary": "Task plan created", "maxSteps": request.max_steps,
                    "tasks": update_task_plan(task_plan, active_index=0),
                })
                result = await executor.execute(request, lambda event_type, data: safe_callback(request.run_id, event_type, data))
            elif pending is None:
                result = await executor.execute(request, lambda event_type, data: safe_callback(request.run_id, event_type, data))
            else:
                result = await executor.resume(pending, decisions or [])
            if not isinstance(result, PendingApproval):
                task_plan = task_plans.pop(request.run_id, task_plans.get(request.run_id, []))
                if task_plan:
                    await safe_callback(request.run_id, "plan.updated", {
                        "summary": "Task plan completed", "maxSteps": request.max_steps,
                        "tasks": update_task_plan(task_plan, completed=True),
                    })
            await finish_execution(request, result)
        except asyncio.CancelledError:
            await store.update(request.run_id, RunStatus.CANCELLED)
            await safe_callback(request.run_id, "run.cancelled", {})
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

    @app.post("/v1/runs/{run_id}/resume", status_code=status.HTTP_202_ACCEPTED,
              dependencies=[Depends(authenticated)])
    async def resume_run(run_id: str, payload: ResumeRunRequest) -> dict[str, str]:
        record = await store.get(run_id)
        question = pending_questions.pop(run_id, None)
        if question is not None:
            resumed = question.request.model_copy(update={
                "task": question.request.task + "\n\nUser answers to ask_user:\n" + json.dumps(payload.answers, ensure_ascii=False),
            })
            tasks[run_id] = asyncio.create_task(run(resumed, skip_plan=True))
            return {"runId": run_id, "status": "RUNNING"}
        pending = pending_approvals.pop(run_id, None)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        if pending is None:
            raise HTTPException(status_code=409, detail="run is not waiting for tool approval")
        tasks[run_id] = asyncio.create_task(run(pending.request, pending, payload.decisions))
        return {"runId": run_id, "status": "RUNNING"}

    return app


app = build_application()
