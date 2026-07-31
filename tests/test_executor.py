from types import SimpleNamespace

from aether_deep_agent_service.executor import DeepAgentExecutor
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


def test_normalize_citation_format_supports_half_width_model_output() -> None:
    content = "依据：[4]，并保留【5】和 Markdown [链接](https://example.test)。"

    assert DeepAgentExecutor._normalize_citation_format(content) == (
        "依据：【4】，并保留【5】和 Markdown [链接](https://example.test)。"
    )
