from types import SimpleNamespace

from aether_deep_agent_service.executor import DeepAgentExecutor, ask_user
from aether_deep_agent_service.schemas import DeepRunRequest
from aether_deep_agent_service.settings import Settings


async def test_load_mcp_tools_filters_to_java_allowed_tools(monkeypatch) -> None:
    captured: dict = {}

    class FakeMcpClient:
        def __init__(self, config: dict) -> None:
            captured["config"] = config

        async def get_tools(self):
            return [SimpleNamespace(name="get_current_time"), SimpleNamespace(name="process_document")]

    monkeypatch.setattr("aether_deep_agent_service.executor.MultiServerMCPClient", FakeMcpClient)
    executor = DeepAgentExecutor(Settings(model="test-model", mcp_url="http://mcp:8000/mcp"))
    request = DeepRunRequest(
        run_id="run-1", user_id="user-1", agent_id="agent-1", conversation_id="conversation-1",
        task="Tell me the time", allowed_tools=["get_current_time"], delegation_token="java-issued-token",
    )

    tools = await executor._load_mcp_tools(request)

    assert [tool.name for tool in tools] == ["get_current_time"]
    assert captured["config"]["aether"]["headers"]["Authorization"] == "Bearer java-issued-token"


def test_parse_plan_keeps_task_specific_titles() -> None:
    tasks = DeepAgentExecutor._parse_plan(
        '{"tasks":[{"title":"提取合同中的付款与违约条款"}, {"title":"识别高风险条款并说明影响"}, {"title":"形成带优先级的修改建议"}]}',
        "审查合同风险",
    )

    assert [task["title"] for task in tasks] == [
        "提取合同中的付款与违约条款",
        "识别高风险条款并说明影响",
        "形成带优先级的修改建议",
    ]


async def test_replan_preserves_completed_steps_without_model() -> None:
    executor = DeepAgentExecutor(Settings())
    request = DeepRunRequest(
        run_id="run-1", user_id="user-1", agent_id="agent-1", conversation_id="conversation-1",
        task="审查合同风险", delegation_token="java-issued-token",
    )

    tasks = await executor.replan(request, [
        {"id": "task-1", "title": "提取条款", "status": "completed"},
        {"id": "task-2", "title": "分析风险", "status": "running"},
    ], "STEP_FAILED", "文档解析工具超时")

    assert tasks[0] == {"id": "task-1", "title": "提取条款", "status": "completed"}
    assert tasks[1]["id"] == "replan-2"
    assert tasks[1]["status"] == "pending"


def test_initial_messages_include_persisted_conversation_memory_before_current_task() -> None:
    request = DeepRunRequest(
        run_id="run-1", user_id="user-1", agent_id="agent-1", conversation_id="conversation-1",
        session_id="session-1", task="继续处理", delegation_token="java-issued-token",
        conversation_memory=[{"role": "assistant", "content": "前序结论"}],
    )

    messages = DeepAgentExecutor._initial_messages(request)

    assert messages[0] == {"role": "assistant", "content": "前序结论"}
    assert "继续处理" in messages[-1]["content"]


def test_deep_run_request_accepts_only_known_tool_approval_policies() -> None:
    request = DeepRunRequest(
        run_id="run-1", user_id="user-1", agent_id="agent-1", conversation_id="conversation-1",
        task="Read the report", delegation_token="java-issued-token", tool_approval_policy="never",
    )

    assert request.tool_approval_policy == "never"


def test_never_policy_skips_mcp_interrupts_but_keeps_ask_user_interrupt() -> None:
    tools = [SimpleNamespace(name="read_document"), SimpleNamespace(name="ask_user")]

    interrupts = DeepAgentExecutor._build_interrupt_on(tools, "never")

    assert interrupts == {"ask_user": {"allowed_decisions": ["respond"]}}


def test_ask_user_uses_human_response_instead_of_tool_approval() -> None:
    tools = [SimpleNamespace(name="read_document"), SimpleNamespace(name="ask_user")]

    interrupts = DeepAgentExecutor._build_interrupt_on(tools, "ask")

    assert interrupts["ask_user"] == {"allowed_decisions": ["respond"]}
    assert interrupts["read_document"] == {"allowed_decisions": ["approve", "reject"]}


def test_ask_user_schema_requires_choices_and_custom_input() -> None:
    schema = ask_user.args_schema.model_json_schema()
    question_schema = schema["$defs"]["AskUserQuestion"]

    assert question_schema["properties"]["type"]["const"] == "choice"
    assert question_schema["properties"]["options"]["minItems"] == 2
    assert question_schema["properties"]["options"]["maxItems"] == 4
    assert question_schema["properties"]["allowCustomInput"]["const"] is True


def test_normalize_citation_format_supports_half_width_model_output() -> None:
    content = "依据：[4]，并保留【5】和 Markdown [链接](https://example.test)。"

    assert DeepAgentExecutor._normalize_citation_format(content) == (
        "依据：【4】，并保留【5】和 Markdown [链接](https://example.test)。"
    )


async def test_resolve_model_prefixes_admin_provider_model(monkeypatch) -> None:
    from aether_deep_agent_service.callbacks import CallbackClient

    settings = Settings()
    callbacks = CallbackClient(settings)
    executor = DeepAgentExecutor(settings, callbacks)

    async def fake_fetch(_agent_id):
        return {"model": "deepseek-v4-flash", "base_url": "https://api.deepseek.com", "api_key": "secret"}

    monkeypatch.setattr(callbacks, "fetch_model_config", fake_fetch)
    request = DeepRunRequest(
        run_id="run-1", user_id="user-1", agent_id="agent-1", conversation_id="conversation-1",
        task="t", delegation_token="tok",
    )

    model, base_url, api_key = await executor._resolve_model(request)

    assert model == "openai:deepseek-v4-flash"  # Admin 的 OpenAI 兼容 provider 需补充前缀
    assert base_url == "https://api.deepseek.com"
    assert api_key == "secret"


async def test_emit_step_verifications_parses_contract_markers() -> None:
    executor = DeepAgentExecutor(Settings())
    request = DeepRunRequest(
        run_id="run-1", user_id="user-1", agent_id="agent-1", conversation_id="conversation-1",
        task="t", delegation_token="tok",
        task_state={"plan": [{"title": "提取条款"}, {"title": "分析风险"}]},
    )
    emitted: list[tuple[str, dict]] = []

    async def emit(event_type: str, data: dict) -> None:
        emitted.append((event_type, data))

    state = {"messages": [type("Msg", (), {"content": (
        "第一步完成。\n[STEP_VERIFIED] 已提取关键条款\n"
        "第二步完成。\n[STEP_VERIFIED] 已按风险等级汇总\n"
    )})()]}

    await executor._emit_step_verifications(state, request, emit)

    assert [e[0] for e in emitted] == ["step.verified", "step.verified"]
    assert emitted[0][1]["stepIndex"] == 1
    assert emitted[0][1]["verification"] == "已提取关键条款"
    assert emitted[1][1]["stepIndex"] == 2
    assert emitted[1][1]["verification"] == "已按风险等级汇总"


async def test_emit_step_verifications_skips_without_plan() -> None:
    executor = DeepAgentExecutor(Settings())
    request = DeepRunRequest(
        run_id="run-1", user_id="user-1", agent_id="agent-1", conversation_id="conversation-1",
        task="t", delegation_token="tok",
    )
    emitted: list[tuple[str, dict]] = []

    async def emit(event_type: str, data: dict) -> None:
        emitted.append((event_type, data))

    await executor._emit_step_verifications({"messages": [type("Msg", (), {"content": "[STEP_VERIFIED] x"})()]}, request, emit)

    assert emitted == []


async def test_emit_step_verifications_falls_back_to_tool_outputs() -> None:
    executor = DeepAgentExecutor(Settings())
    request = DeepRunRequest(
        run_id="run-1", user_id="user-1", agent_id="agent-1", conversation_id="conversation-1",
        task="t", delegation_token="tok",
        task_state={"plan": [{"title": "获取 t1"}, {"title": "获取 t2"}]},
    )
    emitted: list[tuple[str, dict]] = []

    async def emit(event_type: str, data: dict) -> None:
        emitted.append((event_type, data))

    # 模型未输出 [STEP_VERIFIED] 标记，回退用工具消息输出摘要作为每步验证结论。
    state = {
        "messages": [
            type("Msg", (), {"type": "tool", "content": "t1: 2026-08-16T10:00:00"}),
            type("Msg", (), {"type": "tool", "content": "t2: 2026-08-16T11:00:00"}),
            type("Msg", (), {"type": "ai", "content": "完成"}),
        ]
    }

    await executor._emit_step_verifications(state, request, emit)

    assert len(emitted) == 2
    assert emitted[0][1]["stepIndex"] == 1
    assert emitted[0][1]["verification"] == "t1: 2026-08-16T10:00:00"
    assert emitted[1][1]["stepIndex"] == 2
    assert emitted[1][1]["verification"] == "t2: 2026-08-16T11:00:00"
