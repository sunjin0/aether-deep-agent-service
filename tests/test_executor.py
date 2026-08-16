import json
import uuid
from types import SimpleNamespace

from aether_deep_agent_service.executor import DeepAgentExecutor, RunTelemetryHandler, ask_user
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


def test_parse_plan_document_and_complex() -> None:
    content = json.dumps({
        "complex": True,
        "title": "分析合同高风险条款并生成整改清单",
        "goal": "审查《采购合同》初稿，输出按风险等级排序的整改清单。",
        "background": "用户已上传合同 PDF（共 12 页）。",
        "approach": "分三段推进：抽取条款、按四维评估风险、汇总建议。",
        "steps": ["抽取合同全部条款并编号", "按赔付、违约、知识产权、合规维度评估风险", "生成按风险等级排序的整改清单"],
        "risks": ["违约金比例过高需单独提示"],
        "acceptance": ["覆盖全部条款，高风险条款均有修改建议"],
    }, ensure_ascii=False)
    document = DeepAgentExecutor._parse_plan_document(content)
    expected = (
        "# 分析合同高风险条款并生成整改清单\n"
        "## 目标\n审查《采购合同》初稿，输出按风险等级排序的整改清单。\n"
        "## 背景\n用户已上传合同 PDF（共 12 页）。\n"
        "## 方案\n分三段推进：抽取条款、按四维评估风险、汇总建议。\n"
        "## 执行步骤\n"
        "- [ ] 1. 抽取合同全部条款并编号\n"
        "- [ ] 2. 按赔付、违约、知识产权、合规维度评估风险\n"
        "- [ ] 3. 生成按风险等级排序的整改清单\n"
        "## 风险与注意\n- 违约金比例过高需单独提示\n"
        "## 验收标准\n- 覆盖全部条款，高风险条款均有修改建议"
    )
    assert document == expected
    assert DeepAgentExecutor._parse_plan_complex(content) is True
    assert DeepAgentExecutor._parse_plan_complex('{"complex": false, "tasks": []}') is False
    assert DeepAgentExecutor._parse_plan_document("not json") == ""
    assert DeepAgentExecutor._parse_plan_document('{"complex": true}') == ""


def test_tasks_from_document_checklist() -> None:
    document = (
        "# 项目交付风险评估\n"
        "## 执行步骤\n"
        "- [ ] 1. 识别技术风险并制定应对策略\n"
        "- [ ] 2. 识别进度风险并制定应对策略\n"
        "- [ ] 3. 识别资源风险并制定应对策略\n"
        "- [ ] 4. 汇总输出 Markdown 报告\n"
        "## 验收标准\n报告覆盖三个维度。"
    )
    tasks = DeepAgentExecutor._tasks_from_document(document)
    assert [t["title"] for t in tasks] == [
        "识别技术风险并制定应对策略",
        "识别进度风险并制定应对策略",
        "识别资源风险并制定应对策略",
        "汇总输出 Markdown 报告",
    ]
    assert tasks[0]["id"] == "task-1"
    assert tasks[0]["status"] == "pending"
    assert DeepAgentExecutor._tasks_from_document("no checklist here") == []
    assert DeepAgentExecutor._tasks_from_document("") == []


def test_parse_requirement_questions() -> None:
    content = (
        '{"questions":[{"id":"target","question":"目标文档？",'
        '"options":[{"value":"doc_a","label":"文档A"}]}]}'
    )
    questions = DeepAgentExecutor._parse_requirement_questions(content)
    assert len(questions) == 1
    assert questions[0]["id"] == "target"
    assert questions[0]["options"][0]["label"] == "文档A"
    assert DeepAgentExecutor._parse_requirement_questions('{"questions": []}') == []
    assert DeepAgentExecutor._parse_requirement_questions("not json") == []


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


async def test_write_todos_emits_todos_update() -> None:
    calls: list[tuple[str, dict]] = []

    async def capture(event_type: str, data: dict) -> None:
        calls.append((event_type, data))

    handler = RunTelemetryHandler(capture)
    await handler.on_tool_start(
        {"name": "write_todos"},
        '{"todos": [{"content": "识别风险", "status": "pending"}, {"content": "输出报告", "status": "in_progress"}]}',
        run_id=uuid.UUID(int=1),
    )

    todo_events = [c for c in calls if c[0] == "todos.updated"]
    assert todo_events, f"expected todos.updated, got {calls}"
    assert todo_events[0][1]["todos"][0]["content"] == "识别风险"
    assert todo_events[0][1]["todos"][1]["status"] == "in_progress"
    # 非 write_todos 工具不产生 todos.updated
    calls.clear()
    await handler.on_tool_start({"name": "get_current_time"}, "{}", run_id=uuid.UUID(int=2))
    assert not [c for c in calls if c[0] == "todos.updated"]

    # write_todos 的 input 也可能是 Python repr 单引号形式
    calls.clear()
    await handler.on_tool_start(
        {"name": "write_todos"},
        "{'todos': [{'content': '复核报告', 'status': 'in_progress'}]}",
        run_id=uuid.UUID(int=3),
    )
    todo_events = [c for c in calls if c[0] == "todos.updated"]
    assert todo_events
    assert todo_events[0][1]["todos"][0]["content"] == "复核报告"
