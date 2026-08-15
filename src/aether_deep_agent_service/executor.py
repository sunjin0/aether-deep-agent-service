import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal
from uuid import UUID

from deepagents import HarnessProfile, create_deep_agent, register_harness_profile
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command
from pydantic import BaseModel, Field

from .callbacks import CallbackClient
from .schemas import DeepRunRequest
from .settings import Settings


EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]


class AskUserOption(BaseModel):
    """A selectable answer exposed in the chat interaction card."""

    id: str = Field(description="Stable option identifier")
    label: str = Field(description="Option text shown to the user")
    value: str = Field(description="Value returned when the option is selected")


class AskUserQuestion(BaseModel):
    """Required schema for collecting missing user information.

    This is intentionally distinct from tool approval. An ask_user interaction
    always provides concrete choices and a free-text fallback; it must never be
    modelled as a confirm/cancel decision.
    """

    id: str = Field(description="Stable question identifier")
    type: Literal["choice"] = Field(
        default="choice",
        description="Must be choice; do not use confirm or yes/no",
    )
    question: str = Field(description="The missing information to collect")
    options: list[AskUserOption] = Field(
        min_length=2,
        max_length=4,
        description="Two to four concrete, mutually exclusive choices",
    )
    multiple: bool = Field(default=False, description="Whether multiple choices are permitted")
    allowCustomInput: Literal[True] = Field(
        default=True,
        description="Always true so the user can provide information outside the choices",
    )
    customInputPlaceholder: str = Field(
        default="请输入具体信息",
        description="Placeholder for the free-text answer field",
    )


@tool
def ask_user(
    questions: list[AskUserQuestion] = Field(
        min_length=1,
        max_length=4,
        description="One to four structured choice questions",
    ),
    question: str = "",
) -> str:
    """Collect missing user information through choices plus a custom input field.

    Do not use this tool for an approval decision. Every question must use type
    ``choice``, include 2-4 concrete options, and enable custom input.
    """
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
    def __init__(self, settings: Settings, callbacks: CallbackClient | None = None) -> None:
        self.settings = settings
        self._callbacks = callbacks
        self._model_config_cache: dict[str, dict] = {}

    async def _resolve_model(self, request: DeepRunRequest) -> tuple[str, str | None, str | None]:
        """按 Admin 的 agent/provider 配置解析 (model, base_url, api_key)。

        apiKey 通过签名内部通道按需拉取，只在内存中缓存，绝不持久化；
        Admin 不可达时回退到环境变量配置的模型。
        """
        if self._callbacks is not None and request.agent_id:
            cached = self._model_config_cache.get(request.agent_id)
            if cached is None:
                cached = await self._callbacks.fetch_model_config(request.agent_id)
                if cached is not None:
                    self._model_config_cache[request.agent_id] = cached
            if cached and cached.get("model"):
                return cached["model"], cached.get("base_url"), cached.get("api_key")
        if not self.settings.model:
            raise RuntimeError("模型未配置：无法从 Admin 获取且 AETHER_DEEP_AGENT_MODEL 为空")
        model = self.settings.model if ":" in self.settings.model else "openai:" + self.settings.model
        return model, None, None

    @staticmethod
    def _model_kwargs(base_url: str | None, api_key: str | None) -> dict[str, str]:
        kwargs: dict[str, str] = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        return kwargs

    async def plan(self, request: DeepRunRequest) -> list[dict[str, str]]:
        """Use the configured model to turn the user's request into an executable task plan."""
        try:
            model, base_url, api_key = await self._resolve_model(request)
        except Exception:
            return self._fallback_plan(request.task)
        prompt = (
            "Create a concise execution plan for the user's task. Return JSON only in this exact shape: "
            '{"tasks":[{"title":"..."}]}. Generate 3 to 6 concrete, ordered steps. '
            "Each title must describe work needed for this specific task; do not use generic workflow phases. "
            "Do not answer the task itself and do not mention unavailable tools.\n\n"
            f"User task:\n{request.task}"
        )
        try:
            planner = init_chat_model(model, use_responses_api=False, **self._model_kwargs(base_url, api_key))
            response = await asyncio.wait_for(
                planner.ainvoke([{"role": "user", "content": prompt}]),
                timeout=min(request.timeout_seconds or self.settings.run_timeout_seconds, 60),
            )
            return self._parse_plan(getattr(response, "content", ""), request.task)
        except Exception:
            # The run can still proceed when a model provider does not support a separate planning call.
            return self._fallback_plan(request.task)

    async def replan(self, request: DeepRunRequest, previous_plan: list[dict[str, str]],
                     reason: str, observation: str) -> list[dict[str, str]]:
        """Create a new user-visible plan after a material execution observation.

        This deliberately changes only the auditable plan projection. The durable
        LangGraph state remains the authority for the next executable action, so
        a planning call can never repeat a side-effecting tool invocation.
        """
        completed = [dict(item) for item in previous_plan if item.get("status") == "completed"]
        try:
            model, base_url, api_key = await self._resolve_model(request)
        except Exception:
            return completed + self._fallback_replan(request.task, reason, observation, len(completed))
        previous = json.dumps(previous_plan, ensure_ascii=False)
        prompt = (
            "Revise the execution plan after a runtime observation. Return JSON only in this exact shape: "
            '{"tasks":[{"title":"..."}]}. Return 1 to 5 concrete remaining steps. '
            "Do not repeat completed steps, do not answer the task, and do not call tools.\n\n"
            f"User task:\n{request.task}\n\n"
            f"Replan reason: {reason}\n"
            f"Observation (may be summarized):\n{observation[:1000]}\n\n"
            f"Previous plan:\n{previous}"
        )
        try:
            planner = init_chat_model(model, use_responses_api=False, **self._model_kwargs(base_url, api_key))
            response = await asyncio.wait_for(
                planner.ainvoke([{"role": "user", "content": prompt}]),
                timeout=min(request.timeout_seconds or self.settings.run_timeout_seconds, 60),
            )
            remaining = self._parse_plan(getattr(response, "content", ""), request.task)
        except Exception:
            remaining = self._fallback_replan(request.task, reason, observation, len(completed))
        for index, item in enumerate(remaining, start=len(completed) + 1):
            item["id"] = f"replan-{index}"
            item["status"] = "pending"
        return completed + remaining

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

    @staticmethod
    def _fallback_replan(task: str, reason: str, observation: str, completed_count: int) -> list[dict[str, str]]:
        detail = observation.strip().replace("\n", " ")[:80]
        suffix = f"（{detail}）" if detail else ""
        title = task.strip().replace("\n", " ")[:70] or "完成当前任务"
        return [{
            "id": f"replan-{completed_count + 1}",
            "title": f"根据{reason}调整后继续处理：{title}{suffix}",
            "status": "pending",
        }]

    async def execute(self, request: DeepRunRequest, emit: EventSink, checkpointer: Any) -> "ExecutionResult | PendingApproval":
        agent, config, telemetry, model = await self._create_agent(request, emit, checkpointer)
        state = await asyncio.wait_for(agent.ainvoke({"messages": self._initial_messages(request)}, config=config), timeout=request.timeout_seconds)
        return self._result_or_pending(state, request, agent, config, telemetry, model)

    async def continue_from_checkpoint(self, request: DeepRunRequest, emit: EventSink, checkpointer: Any) -> "ExecutionResult | PendingApproval":
        """Resume a paused graph from LangGraph's durable thread checkpoint."""
        agent, config, telemetry, model = await self._create_agent(request, emit, checkpointer)
        state = await asyncio.wait_for(agent.ainvoke(None, config=config), timeout=request.timeout_seconds)
        return self._result_or_pending(state, request, agent, config, telemetry, model)

    async def resume(self, request: DeepRunRequest, decisions: list[dict[str, Any]], emit: EventSink, checkpointer: Any) -> "ExecutionResult | PendingApproval":
        """Rebuild the graph and use the durable thread state to resolve an interrupt."""
        agent, config, telemetry, model = await self._create_agent(request, emit, checkpointer)
        state = await asyncio.wait_for(agent.ainvoke(Command(resume={"decisions": decisions}), config=config), timeout=request.timeout_seconds)
        return self._result_or_pending(state, request, agent, config, telemetry, model)

    async def _create_agent(self, request: DeepRunRequest, emit: EventSink, checkpointer: Any) -> tuple[Any, dict[str, Any], RunTelemetryHandler, str]:
        # 模型配置（model/baseUrl/apiKey）优先来自 Admin 的 agent/provider 解析；
        # Java 供应商通常只保存模型名（如 deepseek-v4-flash），显式补充 provider
        # 避免 LangChain 将其误判为需要额外 SDK 的原生 DeepSeek 模型。
        model, model_base_url, model_api_key = await self._resolve_model(request)

        register_harness_profile(
            model,
            HarnessProfile(excluded_tools=frozenset({
                "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute",
            })),
        )
        instructions = (
            f"{request.system_prompt}\n\n"
            "You are a read-only knowledge analysis agent. Use only the supplied evidence. "
            "Do not claim to have used a source that is not in the evidence. Cite evidence using its bracketed citation. "
            "When a required goal, constraint, preference, or decision is missing, call ask_user instead of guessing. "
            "Use 1-4 structured choice questions, each with 2-4 concrete options; the user can also provide custom input. "
            "After answers are returned, treat them as authoritative and continue without repeating the same question."
        )
        if request.task_state:
            instructions += (
                "\n\nCurrent durable task state (a concise execution projection, not hidden reasoning):\n"
                + json.dumps(request.task_state, ensure_ascii=False)
                + "\nUse verified tool results to adjust the next action; do not repeat completed work."
            )
        tools = await self._load_mcp_tools(request)
        tools.append(ask_user)
        await emit("step.started", {"message": "Preparing delegated MCP tools", "toolCount": len(tools)})
        telemetry = RunTelemetryHandler(emit)
        # DeepSeek 的 OpenAI 兼容端点支持 Chat Completions，不提供 Responses API。
        # 预初始化模型以明确关闭 Responses API，避免客户端请求 /responses 返回 404。
        # base_url/api_key 来自 Admin 的 provider 配置，仅在内存中使用。
        chat_model = init_chat_model(
            model, use_responses_api=False, streaming=True,
            **self._model_kwargs(model_base_url, model_api_key),
        )
        # `never` is an explicit run-scoped grant issued by Java. Other policies
        # interrupt here; Java evaluates per-action risk and auto-resumes a
        # low-risk `risky` batch without exposing a confirmation card.
        interrupt_on = self._build_interrupt_on(tools, request.tool_approval_policy)
        agent = create_deep_agent(
            model=chat_model,
            tools=tools,
            system_prompt=instructions,
            interrupt_on=interrupt_on,
            checkpointer=checkpointer,
        )
        # 一次 MCP 工具调用至少包含模型决策、工具调用和结果归纳三个图节点。
        # Java 配置 max_steps=1 时，原先的 4 次递归预算不足以完成这条最短链路。
        recursion_limit = max(16, request.max_steps * 4)
        config = {
            "recursion_limit": recursion_limit,
            "callbacks": [telemetry],
            # A session is durable across user requests. run_id remains an
            # execution-attempt/audit identifier rather than the memory key.
            "configurable": {"thread_id": request.session_id or request.conversation_id},
        }
        return agent, config, telemetry, model

    @staticmethod
    def _initial_messages(request: DeepRunRequest) -> list[dict[str, str]]:
        messages = [{"role": item.role, "content": item.content} for item in request.conversation_memory]
        messages.append({"role": "user", "content": f"Task:\n{request.task}\n\nEvidence:\n{DeepAgentExecutor._source_context(request)}"})
        return messages

    @staticmethod
    def _source_context(request: DeepRunRequest) -> str:
        return "\n\n".join(f"[{source.citation}] {source.documentName or source.title}\n{source.content}" for source in request.knowledge_sources)

    @staticmethod
    def _build_interrupt_on(tools: list, approval_policy: str) -> dict[str, dict[str, list[str]]]:
        decision_config = {"allowed_decisions": ["approve", "reject"]}
        # ask_user is answered by the human; it is not a binary tool approval.
        interrupts = {ask_user.name: {"allowed_decisions": ["respond"]}}
        if approval_policy != "never":
            interrupts.update({tool.name: decision_config for tool in tools if tool.name != ask_user.name})
        return interrupts

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
