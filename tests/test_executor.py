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
