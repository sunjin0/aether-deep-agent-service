from langchain_core.outputs import LLMResult
from uuid import uuid4

from aether_deep_agent_service.executor import RunTelemetryHandler


async def test_telemetry_emits_tool_lifecycle_and_accumulates_usage() -> None:
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, data: dict) -> None:
        events.append((event_type, data))

    telemetry = RunTelemetryHandler(emit)
    tool_run_id = uuid4()
    await telemetry.on_tool_start({"name": "get_current_time"}, "{}", run_id=tool_run_id)
    await telemetry.on_tool_end({"timestamp": "2026-01-01T00:00:00Z"}, run_id=tool_run_id)
    await telemetry.on_llm_end(LLMResult(generations=[[]], llm_output={
        "token_usage": {"prompt_tokens": 12, "completion_tokens": 8},
    }))

    assert telemetry.tools == ["get_current_time"]
    assert telemetry.prompt_tokens == 12
    assert telemetry.completion_tokens == 8
    assert [event[0] for event in events] == ["tool.started", "tool.completed"]
    assert events[1][1]["toolName"] == "get_current_time"
