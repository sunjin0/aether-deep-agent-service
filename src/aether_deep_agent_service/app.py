import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status

from .callbacks import CallbackClient
from .executor import DeepAgentExecutor
from .schemas import DeepRunRequest, DeepRunResponse, RunStatus
from .security import verify_request_signature
from .settings import Settings, get_settings
from .store import RunStore


logger = logging.getLogger(__name__)


def resolve_run_timeout(payload: DeepRunRequest, settings: Settings) -> int:
    return payload.timeout_seconds if payload.timeout_seconds is not None else settings.run_timeout_seconds


def build_application(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    store = RunStore(resolved_settings.database_url)
    callbacks = CallbackClient(resolved_settings)
    executor = DeepAgentExecutor(resolved_settings)
    tasks: dict[str, asyncio.Task] = {}

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

    async def run(request: DeepRunRequest) -> None:
        await store.update(request.run_id, RunStatus.RUNNING)
        try:
            await safe_callback(request.run_id, "run.started", {"status": RunStatus.RUNNING})
            await safe_callback(request.run_id, "plan.updated", {
                "summary": "Starting read-only knowledge analysis", "maxSteps": request.max_steps,
            })
            result = await executor.execute(request, lambda event_type, data: safe_callback(request.run_id, event_type, data))
            await store.update(request.run_id, RunStatus.SUCCEEDED, result=result.content)
            await safe_callback(request.run_id, "run.completed", {
                "content": result.content, "citations": result.citations,
                "model": result.model, "tools": result.tools,
                "promptTokens": result.prompt_tokens, "completionTokens": result.completion_tokens,
                "totalTokens": ((result.prompt_tokens or 0) + (result.completion_tokens or 0)) or None,
            })
        except asyncio.CancelledError:
            await store.update(request.run_id, RunStatus.CANCELLED)
            await safe_callback(request.run_id, "run.cancelled", {})
            raise
        except Exception as error:
            message = str(error)
            logger.exception("Deep run %s failed", request.run_id)
            await store.update(request.run_id, RunStatus.FAILED, error=message)
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

    return app


app = build_application()
