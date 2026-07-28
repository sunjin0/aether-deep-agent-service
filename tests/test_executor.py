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
