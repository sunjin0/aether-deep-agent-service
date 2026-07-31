import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import UUID

from deepagents import HarnessProfile, create_deep_agent, register_harness_profile
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .schemas import DeepRunRequest
from .settings import Settings


EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]


@tool
def ask_user(questions: list[dict[str, Any]], question: str = "") -> str:
    """Ask the user 1-4 structured choice or confirm questions when required to continue."""
    return "User answers will be provided before this tool continues."


class RunTelemetryHandler(AsyncCallbackHandler):
    """Forwards actual LangChain tool lifecycle events to the Java run audit."""

    def __init__(self, emit: EventSink) -> None:
        self.emit = emit
        self.tools: list[str] = []
        self._tool_calls_by_run: dict[UUID, tuple[str, float]] = {}
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None

    async def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, **_: Any) -> None:
        name = str(serialized.get("name") or serialized.get("id", ["tool"])[-1])
        self.tools.append(name)
        self._tool_calls_by_run[run_id] = (name, time.monotonic())
        await self.emit("tool.started", {
            "toolCallId": str(run_id), "toolName": name, "arguments": input_str,
            "message": "Calling " + name,
        })

    async def on_tool_end(self, output: Any, *, run_id: UUID, **_: Any) -> None:
        summary = str(output).replace("\n", " ")[:240]
        name, started_at = self._tool_calls_by_run.pop(run_id, ("tool", time.monotonic()))
        await self.emit("tool.completed", {
            "toolCallId": str(run_id), "toolName": name, "message": "Completed " + name,
            "outputSummary": summary, "latencyMs": int((time.monotonic() - started_at) * 1000),
        })

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **_: Any) -> None:
        name, started_at = self._tool_calls_by_run.pop(run_id, ("tool", time.monotonic()))
        await self.emit("tool.failed", {
            "toolCallId": str(run_id), "toolName": name, "message": "Tool failed: " + name,
            "error": str(error), "latencyMs": int((time.monotonic() - started_at) * 1000),
        })

    async def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        usage = (response.llm_output or {}).get("token_usage") or {}
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion = usage.get("completion_tokens") or usage.get("output_tokens")
        if isinstance(prompt, int):
            self.prompt_tokens = (self.prompt_tokens or 0) + prompt
        if isinstance(completion, int):
            self.completion_tokens = (self.completion_tokens or 0) + completion

    async def on_llm_new_token(self, token: str, **_: Any) -> None:
        if token:
            await self.emit("message.delta", {"chunk": token})


class DeepAgentExecutor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def plan(self, request: DeepRunRequest) -> list[dict[str, str]]:
        """Use the configured model to turn the user's request into an executable task plan."""
        if not self.settings.model:
            return self._fallback_plan(request.task)
        model = self.settings.model if ":" in self.settings.model else "openai:" + self.settings.model
        prompt = (
            "Create a concise execution plan for the user's task. Return JSON only in this exact shape: "
            '{"tasks":[{"title":"..."}]}. Generate 3 to 6 concrete, ordered steps. '
            "Each title must describe work needed for this specific task; do not use generic workflow phases. "
            "Do not answer the task itself and do not mention unavailable tools.\n\n"
            f"User task:\n{request.task}"
        )
        try:
            planner = init_chat_model(model, use_responses_api=False)
            response = await asyncio.wait_for(
                planner.ainvoke([{"role": "user", "content": prompt}]),
                timeout=min(request.timeout_seconds or self.settings.run_timeout_seconds, 60),
            )
            return self._parse_plan(getattr(response, "content", ""), request.task)
        except Exception:
            # The run can still proceed when a model provider does not support a separate planning call.
            return self._fallback_plan(request.task)

    @staticmethod
    def _parse_plan(content: Any, task: str) -> list[dict[str, str]]:
        if not isinstance(content, str):
            return DeepAgentExecutor._fallback_plan(task)
        normalized = content.strip()
        if normalized.startswith("```"):
            normalized = normalized.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            raw_tasks = json.loads(normalized).get("tasks", [])
        except (json.JSONDecodeError, AttributeError):
            return DeepAgentExecutor._fallback_plan(task)
        tasks: list[dict[str, str]] = []
        for item in raw_tasks[:6]:
            title = item.get("title") if isinstance(item, dict) else None
            if isinstance(title, str) and title.strip():
                tasks.append({"id": f"task-{len(tasks) + 1}", "title": title.strip(), "status": "pending"})
        return tasks if len(tasks) >= 2 else DeepAgentExecutor._fallback_plan(task)

    @staticmethod
    def _fallback_plan(task: str) -> list[dict[str, str]]:
        title = task.strip().replace("\n", " ")[:80] or "完成当前任务"
        return [{"id": "task-1", "title": title, "status": "pending"}]

    async def execute(self, request: DeepRunRequest, emit: EventSink) -> "ExecutionResult | PendingApproval":
        if not self.settings.model:
            raise RuntimeError("AETHER_DEEP_AGENT_MODEL is not configured")

        # Java 模型供应商使用 OpenAI 兼容端点时通常只保存模型名（如 deepseek-v4-flash）。
        # 显式补充 provider，避免 LangChain 将其误判为需要额外 SDK 的原生 DeepSeek 模型。
        model = self.settings.model if ":" in self.settings.model else "openai:" + self.settings.model

        register_harness_profile(
            model,
            HarnessProfile(excluded_tools=frozenset({
                "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute",
            })),
        )
        source_context = "\n\n".join(
            f"[{source.citation}] {source.documentName or source.title}\n{source.content}" for source in request.knowledge_sources
        )
        instructions = (
            f"{request.system_prompt}\n\n"
            "You are a read-only knowledge analysis agent. Use only the supplied evidence. "
            "Do not claim to have used a source that is not in the evidence. Cite evidence using its bracketed citation. "
            "When a required goal, constraint, preference, or decision is missing, call ask_user instead of guessing. "
            "Use 1-4 structured choice or confirm questions and continue only after the user answers."
        )
        tools = await self._load_mcp_tools(request)
        tools.append(ask_user)
        await emit("step.started", {"message": "Preparing delegated MCP tools", "toolCount": len(tools)})
        telemetry = RunTelemetryHandler(emit)
        # DeepSeek 的 OpenAI 兼容端点支持 Chat Completions，不提供 Responses API。
        # 预初始化模型以明确关闭 Responses API，避免客户端请求 /responses 返回 404。
        chat_model = init_chat_model(model, use_responses_api=False, streaming=True)
        # 所有委托 MCP 工具均须在实际执行前中断，避免 Deep 模式绕过平台审批。
        agent = create_deep_agent(
            model=chat_model,
            tools=tools,
            system_prompt=instructions,
            interrupt_on={tool.name: {"allowed_decisions": ["approve", "reject"]} for tool in tools},
            checkpointer=InMemorySaver(),
        )
        # 一次 MCP 工具调用至少包含模型决策、工具调用和结果归纳三个图节点。
        # Java 配置 max_steps=1 时，原先的 4 次递归预算不足以完成这条最短链路。
        recursion_limit = max(16, request.max_steps * 4)
        config = {
            "recursion_limit": recursion_limit,
            "callbacks": [telemetry],
            "configurable": {"thread_id": request.run_id},
        }
        state = await asyncio.wait_for(agent.ainvoke({"messages": [{"role": "user", "content": (
            f"Task:\n{request.task}\n\nEvidence:\n{source_context}"
        )}]}, config=config), timeout=request.timeout_seconds)
        return self._result_or_pending(state, request, agent, config, telemetry, model)

    async def resume(self, pending: "PendingApproval", decisions: list[dict[str, Any]]) -> "ExecutionResult | PendingApproval":
        state = await asyncio.wait_for(
            pending.agent.ainvoke(Command(resume={"decisions": decisions}), config=pending.config),
            timeout=pending.timeout_seconds,
        )
        return self._result_or_pending(state, pending.request, pending.agent, pending.config, pending.telemetry, pending.model)

    def _result_or_pending(self, state: dict[str, Any], request: DeepRunRequest,
                           agent: Any, config: dict[str, Any], telemetry: RunTelemetryHandler,
                           model: str) -> "ExecutionResult | PendingApproval":
        interrupts = state.get("__interrupt__") or []
        if interrupts:
            raw = getattr(interrupts[0], "value", interrupts[0])
            actions = raw.get("action_requests", []) if isinstance(raw, dict) else []
            if not actions:
                raise RuntimeError("Deep Agent returned an invalid tool approval interrupt")
            return PendingApproval(request, agent, config, telemetry, actions, request.timeout_seconds, model)
        messages = state.get("messages", [])
        if not messages:
            raise RuntimeError("Deep Agent returned no messages")
        content = getattr(messages[-1], "content", "")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Deep Agent returned an empty final message")
        # 模型偶尔会把要求的全角引用【1】输出为半角[1]。在持久化前统一格式，
        # 以保证 Java 的引用审计和 Dashboard 的来源锚点使用同一个编号。
        content = self._normalize_citation_format(content)
        citations = [source.model_dump() for source in request.knowledge_sources if source.citation in content]
        return ExecutionResult(
            content=content,
            citations=citations,
            model=model,
            tools=list(dict.fromkeys(telemetry.tools)),
            prompt_tokens=telemetry.prompt_tokens,
            completion_tokens=telemetry.completion_tokens,
        )

    @staticmethod
    def _normalize_citation_format(content: str) -> str:
        return re.sub(r"(?<!【)\[(\d+)\]", r"【\1】", content)

    async def _load_mcp_tools(self, request: DeepRunRequest) -> list:
        if not request.allowed_tools:
            return []
        if not self.settings.mcp_url:
            raise RuntimeError("AETHER_DEEP_AGENT_MCP_URL is not configured for requested MCP tools")
        client = MultiServerMCPClient({
            "aether": {
                "transport": "http",
                "url": self.settings.mcp_url,
                "headers": {"Authorization": "Bearer " + request.delegation_token},
            }
        })
        loaded = await client.get_tools()
        allowed = set(request.allowed_tools)
        return [tool for tool in loaded if tool.name in allowed]


@dataclass
class ExecutionResult:
    content: str
    citations: list[dict]
    model: str
    tools: list[str]
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass
class PendingApproval:
    request: DeepRunRequest
    agent: Any
    config: dict[str, Any]
    telemetry: RunTelemetryHandler
    actions: list[dict[str, Any]]
    timeout_seconds: int
    model: str


@dataclass
class PendingUserQuestion:
    request: DeepRunRequest
    actions: list[dict[str, Any]]
