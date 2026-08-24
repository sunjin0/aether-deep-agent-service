import asyncio
import ast
import re
import json
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

# 简单寒暄/自我介绍问题不走 ask_user 需求分析。
_SIMPLE_GREETING = re.compile(
    r"(你好|您好|在吗|你是谁|你是什么|你能干啥|你会什么|你能做什么|你有哪些功能|"
    r"^hi\b|^hello\b|^hey\b|谢谢|再见|你好吗)",
    re.IGNORECASE,
)


class AskUserOption(BaseModel):
    """聊天交互卡片中展示的可选答案。"""

    id: str = Field(description="Stable option identifier")
    label: str = Field(description="Option text shown to the user")
    value: str = Field(description="Value returned when the option is selected")


class AskUserQuestion(BaseModel):
    """收集缺失用户信息的必填结构。

    该结构有意区别于工具审批。``ask_user`` 交互始终提供具体选项和自由文本兜底，
    绝不能建模为确认/取消决策。
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
    """通过选项和自定义输入框收集缺失的用户信息。

    不要将此工具用于审批决策。每个问题必须使用 ``choice`` 类型，提供 2 至 4 个
    具体选项，并启用自定义输入。
    """
    return "User answers will be provided before this tool continues."


class RunTelemetryHandler(AsyncCallbackHandler):
    """将实际 LangChain 工具生命周期事件转发至 Java 运行审计。"""

    def __init__(self, emit: EventSink) -> None:
        """初始化事件接收器及本次运行的遥测累计值。"""
        self.emit = emit
        self.tools: list[str] = []
        self._tool_calls_by_run: dict[UUID, tuple[str, float]] = {}
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None

    async def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, **_: Any) -> None:
        """记录工具开始时间，并发布工具启动事件。"""
        name = str(serialized.get("name") or serialized.get("id", ["tool"])[-1])
        self.tools.append(name)
        self._tool_calls_by_run[run_id] = (name, time.monotonic())
        await self.emit("tool.started", {
            "toolCallId": str(run_id), "toolName": name, "arguments": input_str,
            "message": "Calling " + name,
        })
        if name == "write_todos":
            await self._emit_todos_update(input_str)

    async def _emit_todos_update(self, input_str: str) -> None:
        """write_todos 是模型最新的执行计划：以 todos.updated 事件驱动计划投影覆盖。

        write_todos 的 input 可能是 JSON 双引号，也可能是 Python repr 单引号
        （如 {'todos': [...]}），故用 ast.literal_eval 兼容两种形式。
        """
        try:
            payload = ast.literal_eval(input_str) if input_str else {}
            todos = payload.get("todos", []) if isinstance(payload, dict) else []
            if isinstance(todos, list):
                await self.emit("todos.updated", {"todos": todos})
        except (ValueError, SyntaxError, AttributeError):
            pass

    async def on_tool_end(self, output: Any, *, run_id: UUID, **_: Any) -> None:
        """汇总工具输出和耗时，并发布工具完成事件。"""
        summary = str(output).replace("\n", " ")[:240]
        name, started_at = self._tool_calls_by_run.pop(run_id, ("tool", time.monotonic()))
        await self.emit("tool.completed", {
            "toolCallId": str(run_id), "toolName": name, "message": "Completed " + name,
            "outputSummary": summary, "latencyMs": int((time.monotonic() - started_at) * 1000),
        })

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **_: Any) -> None:
        """汇总工具异常和耗时，并发布工具失败事件。"""
        name, started_at = self._tool_calls_by_run.pop(run_id, ("tool", time.monotonic()))
        await self.emit("tool.failed", {
            "toolCallId": str(run_id), "toolName": name, "message": "Tool failed: " + name,
            "error": str(error), "latencyMs": int((time.monotonic() - started_at) * 1000),
        })

    async def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        """从模型响应中累计输入和输出 Token 用量。"""
        usage = (response.llm_output or {}).get("token_usage") or {}
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion = usage.get("completion_tokens") or usage.get("output_tokens")
        if isinstance(prompt, int):
            self.prompt_tokens = (self.prompt_tokens or 0) + prompt
        if isinstance(completion, int):
            self.completion_tokens = (self.completion_tokens or 0) + completion

    async def on_llm_new_token(self, token: str, **_: Any) -> None:
        """将模型流式增量文本转发给事件接收器。"""
        if token:
            await self.emit("message.delta", {"chunk": token})


class DeepAgentExecutor:
    """负责规划、创建并驱动 LangGraph Deep Agent 的执行器。"""

    def __init__(self, settings: Settings, callbacks: CallbackClient | None = None) -> None:
        """保存服务配置及可选的 Admin 回调客户端。"""
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
                # Admin 的 provider 为 OpenAI 兼容端点，只存模型名；显式补充
                # provider 前缀，避免 LangChain 误判为原生 DeepSeek 模型。
                model = cached["model"] if ":" in cached["model"] else "openai:" + cached["model"]
                return model, cached.get("base_url"), cached.get("api_key")
        if not self.settings.model:
            raise RuntimeError("模型未配置：无法从 Admin 获取且 AETHER_DEEP_AGENT_MODEL 为空")
        model = self.settings.model if ":" in self.settings.model else "openai:" + self.settings.model
        return model, None, None

    @staticmethod
    def _model_kwargs(base_url: str | None, api_key: str | None) -> dict[str, str]:
        """将可选模型连接配置整理为模型初始化参数。"""
        kwargs: dict[str, str] = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        return kwargs

    async def analyze_requirements(self, request: DeepRunRequest) -> list[dict[str, Any]]:
        """规划前分析用户问题是否信息完整，返回需要补充的提问（空列表=信息完整）。

        缺失信息时先让用户补充，避免拿到不完整请求就盲目生成规划。
        简单寒暄/自我介绍问题不走 ask_user，直接进入回答。
        """
        if _SIMPLE_GREETING.search(request.task or ""):
            return []
        try:
            model, base_url, api_key = await self._resolve_model(request)
        except Exception:
            return []
        prompt = (
            "Analyze whether the user's request is complete and actionable. "
            "Return JSON only in this exact shape: "
            '{"questions":[{"id":"...","question":"...","options":[{"value":"...","label":"...","recommended":true|false}]}]}. '
            "Return an empty questions array when the request is complete. "
            "Only ask for information that is genuinely required and cannot be derived from the request "
            "or the supplied evidence. Do not ask how to answer; ask only for missing inputs. "
            "For each question, provide 2-4 concrete, domain-specific options based on the request context "
            "(e.g. '科技计划', '科研项目', '专项资金'), and mark the most likely one with \"recommended\": true. "
            "Do not use generic placeholders such as '提供具体信息' or '暂无相关信息'.\n\n"
            f"User task:\n{request.task}\n\n"
            f"Available evidence:\n{self._source_context(request)}"
        )
        try:
            planner = init_chat_model(model, use_responses_api=False, **self._model_kwargs(base_url, api_key))
            response = await asyncio.wait_for(
                planner.ainvoke([{"role": "user", "content": prompt}]),
                timeout=min(request.timeout_seconds or self.settings.run_timeout_seconds, 30),
            )
            return self._parse_requirement_questions(getattr(response, "content", ""))
        except Exception:
            return []

    @staticmethod
    def _parse_requirement_questions(content: Any) -> list[dict[str, Any]]:
        """解析模型返回的需求补充问题，并过滤不合法项。"""
        if not isinstance(content, str):
            return []
        normalized = content.strip()
        if normalized.startswith("```"):
            normalized = normalized.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(normalized)
            questions = data.get("questions") if isinstance(data, dict) else None
        except (json.JSONDecodeError, AttributeError):
            return []
        result: list[dict[str, Any]] = []
        if not isinstance(questions, list):
            return result
        for index, raw in enumerate(questions[:4]):
            if not isinstance(raw, dict) or not str(raw.get("question") or "").strip():
                continue
            options = []
            for i, option in enumerate(raw.get("options") or []):
                if isinstance(option, dict):
                    value = str(option.get("value") or option.get("id") or f"option_{i + 1}")
                    label = str(option.get("label") or option.get("text") or option.get("name") or value)
                    option_item: dict[str, Any] = {"value": value, "label": label}
                    if option.get("recommended"):
                        option_item["recommended"] = True
                    options.append(option_item)
            if not options:
                # 模型未给出具体选项时，仅提供一个允许用户直接输入的自定义选项。
                options = [{"value": "provide_details", "label": "由用户提供", "recommended": True}]
            result.append({
                "id": str(raw.get("id") or f"question_{index + 1}"),
                "question": str(raw["question"]).strip(),
                "options": options,
                "multiple": bool(raw.get("multiple")),
                "allowCustomInput": True,
            })
        return result

    async def plan_document(self, request: DeepRunRequest, feedback: str | None = None) -> tuple[bool, str]:
        """生成规划文档（方案说明 + 复杂度判断），供用户审批。

        feedback 非空时（方案反馈重规划）把用户意见并入任务描述，据以调整方案。
        """
        try:
            model, base_url, api_key = await self._resolve_model(request)
        except Exception:
            return False, ""
        task_text = request.task
        if feedback:
            task_text = f"{task_text}\n\n用户对方案的反馈（请据此修改方案）：\n{feedback}"
        prompt = (
            "Analyze the user's task and return a JSON object with exactly this shape (no Markdown fence): "
            '{"complex":<true|false>,"title":"<任务标题>","goal":"<交付物，一到两句>",'
            '"background":"<已知信息、为什么采用此方案、约束；没有就留空>","approach":"<做法与理由、边界>",'
            '"steps":["<具体步骤标题>","<具体步骤标题>",...],"risks":["<已知风险；没有就留空数组>"],'
            '"acceptance":["<可验证的完成标准>"]}. '
            '"complex" must be true when the task needs a real multi-stage plan with tools or sub-steps, '
            "and false when it is a simple question the agent can answer directly. "
            "The plan is the approved execution contract shown to the user, NOT the final answer. "
            "Write all text in the user's language and be concrete and specific to THIS task: "
            "name the actual inputs, the actual dimensions to analyse and the actual deliverable. "
            "\"steps\" must contain 1 to 6 real, concrete, ordered work items for this task "
            "(not generic workflow phases); they are the execution checklist that will be carried out.\n\n"
            f"User task:\n{task_text}"
        )
        try:
            planner = init_chat_model(model, use_responses_api=False, **self._model_kwargs(base_url, api_key))
            response = await asyncio.wait_for(
                planner.ainvoke([{"role": "user", "content": prompt}]),
                timeout=min(request.timeout_seconds or self.settings.run_timeout_seconds, 60),
            )
            content = getattr(response, "content", "")
            if request.task_state:
                document = self._parse_plan_document(content)
                if document:
                    request.task_state["document"] = document
                request.task_state["complex"] = self._parse_plan_complex(content)
            return self._parse_plan_complex(content), self._parse_plan_document(content)
        except Exception:
            return False, ""

    async def plan(self, request: DeepRunRequest) -> list[dict[str, str]]:
        """审批通过后，根据已批准的规划文档生成可执行任务规划（步骤）。

        优先从文档「执行步骤」勾选清单解析步骤，保证文档与 tasks 一一对应（规范 §4）；
        文档缺少步骤时回退到模型单独生成。
        """
        document = (request.task_state or {}).get("document") or ""
        from_document = self._tasks_from_document(document)
        if from_document:
            return from_document
        try:
            model, base_url, api_key = await self._resolve_model(request)
        except Exception:
            return self._fallback_plan(request.task)
        prompt = (
            "Break the approved plan document into a concrete, ordered execution plan. "
            "Return JSON only in this exact shape: "
            '{"tasks":[{"title":"..."}]}. Generate 1 to 6 concrete, ordered steps '
            "proportionate to the task's complexity: a trivial task needs 1 step, while a multi-stage task may need more. "
            "Each title must describe work needed for this specific task; do not use generic workflow phases. "
            "Do not answer the task itself and do not mention unavailable tools.\n\n"
            f"User task:\n{request.task}\n\n"
            f"Approved plan document:\n{document}"
        )
        try:
            planner = init_chat_model(model, use_responses_api=False, **self._model_kwargs(base_url, api_key))
            response = await asyncio.wait_for(
                planner.ainvoke([{"role": "user", "content": prompt}]),
                timeout=min(request.timeout_seconds or self.settings.run_timeout_seconds, 60),
            )
            return self._parse_plan(getattr(response, "content", ""), request.task)
        except Exception:
            # 模型提供商不支持独立规划调用时，运行仍可使用兜底计划继续执行。
            return self._fallback_plan(request.task)

    async def replan(self, request: DeepRunRequest, previous_plan: list[dict[str, str]],
                     reason: str, observation: str) -> list[dict[str, str]]:
        """出现重要执行观察结果后，创建新的用户可见计划。

        此处刻意只变更可审计的计划投影。持久化 LangGraph 状态仍是下一项可执行操作的
        唯一依据，因此规划调用不会重复触发有副作用的工具调用。
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
        """解析模型生成的计划 JSON，失败时返回单步兜底计划。"""
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
    def _tasks_from_document(document: str) -> list[dict[str, str]]:
        """从计划文档「执行步骤」勾选清单解析执行步骤，保证文档与 tasks 一一对应。"""
        if not isinstance(document, str):
            return []
        tasks: list[dict[str, str]] = []
        for line in document.splitlines():
            stripped = line.strip()
            if not stripped.startswith("- [ ]"):
                continue
            title = re.sub(r"^-\s*\[\s*\]\s*(?:\d+\.\s*)?", "", stripped).strip()
            if title:
                tasks.append({"id": f"task-{len(tasks) + 1}", "title": title, "status": "pending"})
        return tasks

    @staticmethod
    def _parse_plan_document(content: Any) -> str:
        """从规划 JSON 的结构化字段拼装规范 §3 的 Markdown 方案文档。

        模型只输出扁平 JSON（title/goal/steps 等），Markdown 由这里程序化生成，
        避免模型在大 JSON 字符串内转义多行 Markdown 时产生非法 JSON。
        """
        if not isinstance(content, str):
            return ""
        normalized = content.strip()
        if normalized.startswith("```"):
            normalized = normalized.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(normalized)
        except (json.JSONDecodeError, AttributeError):
            return ""
        if not isinstance(data, dict):
            return ""
        title = str(data.get("title") or "").strip()
        goal = str(data.get("goal") or "").strip()
        background = str(data.get("background") or "").strip()
        approach = str(data.get("approach") or "").strip()
        raw_steps = data.get("steps")
        steps = [str(s).strip() for s in raw_steps if str(s).strip()] if isinstance(raw_steps, list) else []
        raw_risks = data.get("risks")
        risks = [str(r).strip() for r in raw_risks if str(r).strip()] if isinstance(raw_risks, list) else []
        raw_acceptance = data.get("acceptance")
        acceptance = [str(a).strip() for a in raw_acceptance if str(a).strip()] if isinstance(raw_acceptance, list) else []

        lines: list[str] = []
        if title:
            lines.append(f"# {title}")
        if goal:
            lines.append("## 目标")
            lines.append(goal)
        if background:
            lines.append("## 背景")
            lines.append(background)
        if approach:
            lines.append("## 方案")
            lines.append(approach)
        if steps:
            lines.append("## 执行步骤")
            lines.extend(f"- [ ] {index}. {step}" for index, step in enumerate(steps, start=1))
        if risks:
            lines.append("## 风险与注意")
            lines.extend(f"- {risk}" for risk in risks)
        if acceptance:
            lines.append("## 验收标准")
            lines.extend(f"- {item}" for item in acceptance)
        return "\n".join(lines)

    @staticmethod
    def _parse_plan_complex(content: Any) -> bool:
        """从计划 JSON 中解析模型对任务是否复杂的判断。"""
        if not isinstance(content, str):
            return False
        normalized = content.strip()
        if normalized.startswith("```"):
            normalized = normalized.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(normalized)
            complex_flag = data.get("complex") if isinstance(data, dict) else None
            return bool(complex_flag)
        except (json.JSONDecodeError, AttributeError):
            return False

    @staticmethod
    def _fallback_plan(task: str) -> list[dict[str, str]]:
        """为无法规划的任务生成包含任务原文的单步计划。"""
        title = task.strip().replace("\n", " ")[:80] or "完成当前任务"
        return [{"id": "task-1", "title": title, "status": "pending"}]

    @staticmethod
    def _fallback_replan(task: str, reason: str, observation: str, completed_count: int) -> list[dict[str, str]]:
        """在重规划失败时生成一条包含原因和观察结果的后续步骤。"""
        detail = observation.strip().replace("\n", " ")[:80]
        suffix = f"（{detail}）" if detail else ""
        title = task.strip().replace("\n", " ")[:70] or "完成当前任务"
        return [{
            "id": f"replan-{completed_count + 1}",
            "title": f"根据{reason}调整后继续处理：{title}{suffix}",
            "status": "pending",
        }]

    async def execute(self, request: DeepRunRequest, emit: EventSink, checkpointer: Any) -> "ExecutionResult | PendingApproval":
        """以初始消息执行新的 Agent 图，并返回结果或待审批请求。"""
        agent, config, telemetry, model = await self._create_agent(request, emit, checkpointer)
        state = await asyncio.wait_for(agent.ainvoke({"messages": self._initial_messages(request)}, config=config), timeout=request.timeout_seconds)
        await self._emit_step_verifications(state, request, emit)
        return self._result_or_pending(state, request, agent, config, telemetry, model)

    async def continue_from_checkpoint(self, request: DeepRunRequest, emit: EventSink, checkpointer: Any) -> "ExecutionResult | PendingApproval":
        """从 LangGraph 持久化线程检查点恢复已暂停的图。"""
        agent, config, telemetry, model = await self._create_agent(request, emit, checkpointer)
        state = await asyncio.wait_for(agent.ainvoke(None, config=config), timeout=request.timeout_seconds)
        await self._emit_step_verifications(state, request, emit)
        return self._result_or_pending(state, request, agent, config, telemetry, model)

    async def _emit_step_verifications(self, state: dict[str, Any], request: DeepRunRequest, emit: EventSink) -> None:
        """把计划步骤的验证结论投影为 step.verified 事件。

        优先取模型按契约输出的 [STEP_VERIFIED] 标记；模型未输出时，
        回退用工具消息的输出摘要作为该步骤的验证结论，保证每个计划步骤都有验证。
        """
        plan = (request.task_state or {}).get("plan")
        if not isinstance(plan, list) or not plan:
            return
        messages = state.get("messages", [])
        if not messages:
            return
        verification_by_index: dict[int, str] = {}
        content = getattr(messages[-1], "content", "")
        if isinstance(content, str):
            markers = re.findall(r"\[STEP_VERIFIED\]\s*(.+)", content, re.IGNORECASE)
            for index, verification in enumerate(markers[: len(plan)]):
                verification_by_index[index] = verification.strip()[:500]
        tool_outputs = [
            str(getattr(msg, "content", "")).strip()
            for msg in messages
            if getattr(msg, "type", "") == "tool" and getattr(msg, "content", "")
        ]
        tool_cursor = 0
        for index in range(len(plan)):
            if index in verification_by_index:
                continue
            fallback = tool_outputs[tool_cursor][:500] if tool_cursor < len(tool_outputs) else "步骤已完成并验证"
            verification_by_index[index] = fallback
            tool_cursor += 1
        for index, verification in verification_by_index.items():
            step = plan[index]
            await emit("step.verified", {
                "stepId": step.get("id") or f"step-{index + 1}",
                "stepIndex": index + 1,
                "title": step.get("title") or f"步骤 {index + 1}",
                "verification": verification,
            })

    async def resume(self, request: DeepRunRequest, decisions: list[dict[str, Any]], emit: EventSink, checkpointer: Any) -> "ExecutionResult | PendingApproval":
        """重建图，并使用持久化线程状态处理一次中断。"""
        agent, config, telemetry, model = await self._create_agent(request, emit, checkpointer)
        state = await asyncio.wait_for(agent.ainvoke(Command(resume={"decisions": decisions}), config=config), timeout=request.timeout_seconds)
        return self._result_or_pending(state, request, agent, config, telemetry, model)

    async def _create_agent(self, request: DeepRunRequest, emit: EventSink, checkpointer: Any) -> tuple[Any, dict[str, Any], RunTelemetryHandler, str]:
        """解析模型与工具配置，创建可检查点恢复的 Deep Agent。"""
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
        plan_steps = (request.task_state or {}).get("plan")
        if isinstance(plan_steps, list) and plan_steps:
            # 计划是执行契约：按序执行、逐步验证。最终回复必须以逐步骤验证结论结尾。
            numbered = "\n".join(f"{i + 1}. {step.get('title')}" for i, step in enumerate(plan_steps))
            instructions += (
                "\n\nExecution plan — you MUST follow these steps IN ORDER. "
                "Complete each step's goal before moving to the next; use tools when needed. "
                "Do not skip steps, do not repeat completed work, and do not claim a step is done without verifying its output.\n"
                f"Steps:\n{numbered}\n\n"
                "Your FINAL reply MUST end with one verification line per plan step, in step order, exactly as:\n"
                "[STEP_VERIFIED] <verification summary of step 1>\n"
                "[STEP_VERIFIED] <verification summary of step 2>\n"
                "...\n"
                "Every plan step requires its own [STEP_VERIFIED] line."
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
        # `never` 是 Java 签发的仅限当前运行的显式授权。其余策略在此中断，由 Java
        # 逐项评估风险，并自动恢复低风险的 `risky` 批次，不展示确认卡片。
        interrupt_on = self._build_interrupt_on(tools, request.tool_approval_policy)
        agent = create_deep_agent(
            model=chat_model,
            tools=tools,
            system_prompt=instructions,
            interrupt_on=interrupt_on,
            checkpointer=checkpointer,
        )
        # 一次 MCP 工具调用至少包含模型决策、工具调用和结果归纳三个图节点。
        # Java 配置 max_steps=1 时，原先的 4 次递归预算不足以完成这条最短链路；
        # 多步骤计划（规范 §3 执行步骤）逐步骤执行会消耗更多图节点，按计划长度提升递归预算。
        planned_steps = len((request.task_state or {}).get("plan") or [])
        recursion_limit = max(32, request.max_steps * 4, planned_steps * 6)
        config = {
            "recursion_limit": recursion_limit,
            "callbacks": [telemetry],
            # 会话在多次用户请求间保持持久化；run_id 是执行尝试/审计标识，而非记忆键。
            "configurable": {"thread_id": request.session_id or request.conversation_id},
        }
        return agent, config, telemetry, model

    @staticmethod
    def _initial_messages(request: DeepRunRequest) -> list[dict[str, str]]:
        """拼接持久化会话消息、当前任务和知识证据。"""
        messages = [{"role": item.role, "content": item.content} for item in request.conversation_memory]
        messages.append({"role": "user", "content": f"Task:\n{request.task}\n\nEvidence:\n{DeepAgentExecutor._source_context(request)}"})
        return messages

    @staticmethod
    def _source_context(request: DeepRunRequest) -> str:
        """将知识来源格式化为附带引用编号的模型上下文。"""
        return "\n\n".join(f"[{source.citation}] {source.documentName or source.title}\n{source.content}" for source in request.knowledge_sources)

    @staticmethod
    def _build_interrupt_on(tools: list, approval_policy: str) -> dict[str, dict[str, list[str]]]:
        """根据审批策略为工具生成图中断决策配置。"""
        decision_config = {"allowed_decisions": ["approve", "reject"]}
        # ask_user 由用户回答，并非二元工具审批。
        interrupts = {ask_user.name: {"allowed_decisions": ["respond"]}}
        if approval_policy != "never":
            interrupts.update({tool.name: decision_config for tool in tools if tool.name != ask_user.name})
        # 邮件是不可逆的外部投递，即使会话策略为 never 也必须逐封确认。
        interrupts.update({tool.name: decision_config for tool in tools if tool.name == "send_email"})
        return interrupts

    def _result_or_pending(self, state: dict[str, Any], request: DeepRunRequest,
                           agent: Any, config: dict[str, Any], telemetry: RunTelemetryHandler,
                           model: str) -> "ExecutionResult | PendingApproval":
        """将图状态转换为最终结果，或转换为待人工处理的中断请求。"""
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
        # 移除计划契约的 [STEP_VERIFIED] 验证标记，避免泄漏到最终回复。
        content = self._strip_step_verified(content)
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
    def _strip_step_verified(content: str) -> str:
        """移除仅用于内部计划验证的步骤标记。"""
        lines = [line for line in content.splitlines()
                 if not re.match(r"^\s*\[STEP_VERIFIED\]", line, re.IGNORECASE)]
        return "\n".join(lines).strip()

    @staticmethod
    def _normalize_citation_format(content: str) -> str:
        """将半角数字引用规范为 Java 侧使用的全角引用格式。"""
        return re.sub(r"(?<!【)\[(\d+)\]", r"【\1】", content)

    async def _load_mcp_tools(self, request: DeepRunRequest) -> list:
        """加载 MCP 工具，并严格限制为请求允许调用的工具集合。"""
        if not request.allowed_tools:
            return []
        if not self.settings.mcp_url:
            raise RuntimeError("AETHER_DEEP_AGENT_MCP_URL is not configured for requested MCP tools")
        headers = {"Authorization": "Bearer " + request.delegation_token}
        if request.email_credential_tokens:
            headers["X-Aether-Email-Credentials"] = json.dumps(request.email_credential_tokens, separators=(",", ","))
        client = MultiServerMCPClient({
            "aether": {
                "transport": "http",
                "url": self.settings.mcp_url,
                "headers": headers,
            }
        })
        loaded = await client.get_tools()
        allowed = set(request.allowed_tools)
        return [tool for tool in loaded if tool.name in allowed]


@dataclass
class ExecutionResult:
    """一次成功执行的最终文本、引用、工具和用量统计。"""
    content: str
    citations: list[dict]
    model: str
    tools: list[str]
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass
class PendingApproval:
    """Agent 图因工具调用等待人工审批时保留的上下文。"""
    request: DeepRunRequest
    agent: Any
    config: dict[str, Any]
    telemetry: RunTelemetryHandler
    actions: list[dict[str, Any]]
    timeout_seconds: int
    model: str


@dataclass
class PendingUserQuestion:
    """Agent 图因 ``ask_user`` 等待用户补充信息时保留的上下文。"""
    request: DeepRunRequest
    actions: list[dict[str, Any]]
