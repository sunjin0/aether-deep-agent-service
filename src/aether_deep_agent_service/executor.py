import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import UUID

from deepagents import HarnessProfile, create_deep_agent, register_harness_profile
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_mcp_adapters.client import MultiServerMCPClient

from .schemas import DeepRunRequest
from .settings import Settings


EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]


class RunTelemetryHandler(AsyncCallbackHandler):
    """Forwards actual LangChain tool lifecycle events to the Java run audit."""

    def __init__(self, emit: EventSink) -> None:
        self.emit = emit
        self.tools: list[str] = []
        self._tool_names_by_run: dict[UUID, str] = {}
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None

    async def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, **_: Any) -> None:
        name = str(serialized.get("name") or serialized.get("id", ["tool"])[-1])
        self.tools.append(name)
        self._tool_names_by_run[run_id] = name
        await self.emit("tool.started", {"toolName": name, "message": "Calling " + name})

    async def on_tool_end(self, output: Any, *, run_id: UUID, **_: Any) -> None:
        summary = str(output).replace("\n", " ")[:240]
        name = self._tool_names_by_run.pop(run_id, "tool")
        await self.emit("tool.completed", {"toolName": name, "message": "Completed " + name, "outputSummary": summary})

    async def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        usage = (response.llm_output or {}).get("token_usage") or {}
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion = usage.get("completion_tokens") or usage.get("output_tokens")
        if isinstance(prompt, int):
            self.prompt_tokens = (self.prompt_tokens or 0) + prompt
        if isinstance(completion, int):
            self.completion_tokens = (self.completion_tokens or 0) + completion


class DeepAgentExecutor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def execute(self, request: DeepRunRequest, emit: EventSink) -> "ExecutionResult":
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
            f"[{source.citation}] {source.title}\n{source.content}" for source in request.knowledge_sources
        )
        instructions = (
            f"{request.system_prompt}\n\n"
            "You are a read-only knowledge analysis agent. Use only the supplied evidence. "
            "Do not claim to have used a source that is not in the evidence. Cite evidence using its bracketed citation."
        )
        tools = await self._load_mcp_tools(request)
        await emit("step.started", {"message": "Preparing delegated MCP tools", "toolCount": len(tools)})
        telemetry = RunTelemetryHandler(emit)
        agent = create_deep_agent(model=model, tools=tools, system_prompt=instructions)
        state = await asyncio.wait_for(agent.ainvoke({"messages": [{"role": "user", "content": (
            f"Task:\n{request.task}\n\nEvidence:\n{source_context}"
        )}]}, config={"recursion_limit": request.max_steps * 4, "callbacks": [telemetry]}), timeout=request.timeout_seconds)
        messages = state.get("messages", [])
        if not messages:
            raise RuntimeError("Deep Agent returned no messages")
        content = getattr(messages[-1], "content", "")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Deep Agent returned an empty final message")
        citations = [source.model_dump() for source in request.knowledge_sources if source.citation in content]
        return ExecutionResult(
            content=content,
            citations=citations,
            model=model,
            tools=list(dict.fromkeys(telemetry.tools)),
            prompt_tokens=telemetry.prompt_tokens,
            completion_tokens=telemetry.completion_tokens,
        )

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
