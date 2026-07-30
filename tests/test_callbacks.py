import json
import time
from unittest.mock import AsyncMock, call, patch

import httpx
import pytest
from pydantic import ValidationError
from aether_deep_agent_service.callbacks import CallbackClient
from aether_deep_agent_service.settings import Settings
from aether_deep_agent_service.security import build_signature


def make_client(**kw):
    defaults = {"shared_secret": "test-secret", "callback_base_url": "http://java:8080"}
    return CallbackClient(Settings(**(defaults | kw)))


def make_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "http://java:8080/callback"),
    )


@pytest.mark.asyncio
async def test_callback_retries_on_connect_error_and_succeeds():
    client = make_client(callback_max_retries=2, callback_retry_backoff_seconds=0.01)
    count = [0]

    async def fake_post(url, content, headers):
        count[0] += 1
        if count[0] <= 2:
            raise httpx.ConnectError("refused")
        return make_response(200)

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        await client.send("run-1", "tool.completed", {"toolName": "search", "message": "done"})
    assert count[0] == 3


@pytest.mark.asyncio
async def test_callback_retry_reuses_the_same_event_body_and_event_id():
    client = make_client(callback_max_retries=1, callback_retry_backoff_seconds=0)
    captured = []

    async def fake_post(url, content, headers):
        captured.append((content, headers.copy()))
        return make_response(503 if len(captured) == 1 else 200)

    with patch("aether_deep_agent_service.callbacks.time.time", side_effect=[1000, 1001, 1002]):
        with patch("httpx.AsyncClient.post", side_effect=fake_post):
            await client.send("run-1", "tool.completed", {"toolName": "search"})

    assert len(captured) == 2
    assert captured[0][0] == captured[1][0]
    assert json.loads(captured[0][0])["event_id"] == json.loads(captured[1][0])["event_id"]


@pytest.mark.asyncio
async def test_callback_retry_refreshes_a_valid_signature_for_each_timestamp():
    client = make_client(callback_max_retries=1, callback_retry_backoff_seconds=0)
    captured = []

    async def fake_post(url, content, headers):
        captured.append((content, headers.copy()))
        return make_response(503 if len(captured) == 1 else 200)

    with patch("aether_deep_agent_service.callbacks.time.time", side_effect=[1000, 1001, 1002]):
        with patch("httpx.AsyncClient.post", side_effect=fake_post):
            await client.send("run-1", "tool.completed", {"toolName": "search"})

    assert len(captured) == 2
    timestamps = [headers["X-Aether-Timestamp"] for _, headers in captured]
    assert timestamps == ["1001", "1002"]
    for body, headers in captured:
        assert headers["X-Aether-Signature"] == build_signature(
            "test-secret", headers["X-Aether-Timestamp"], body
        )
    assert captured[0][1]["X-Aether-Signature"] != captured[1][1]["X-Aether-Signature"]


@pytest.mark.asyncio
async def test_callback_reraises_final_request_error_after_exhausting_retries():
    client = make_client(callback_max_retries=2, callback_retry_backoff_seconds=0)
    first_error = httpx.ConnectError("first failure")
    second_error = httpx.ReadError("second failure")
    final_error = httpx.ConnectError("final failure")

    with patch(
        "httpx.AsyncClient.post", side_effect=[first_error, second_error, final_error]
    ) as post:
        with pytest.raises(httpx.RequestError) as exc_info:
            await client.send("run-1", "tool.completed", {"toolName": "search"})

    assert exc_info.value is final_error
    assert post.await_count == 3


@pytest.mark.asyncio
async def test_callback_retry_uses_linear_backoff():
    client = make_client(callback_max_retries=2, callback_retry_backoff_seconds=0.25)
    sleep = AsyncMock()

    with patch("httpx.AsyncClient.post", side_effect=[make_response(503), make_response(503), make_response(200)]):
        with patch("aether_deep_agent_service.callbacks.asyncio.sleep", sleep):
            await client.send("run-1", "tool.completed", {"toolName": "search"})

    assert sleep.await_args_list == [call(0.25), call(0.5)]


@pytest.mark.asyncio
async def test_callback_raises_without_retrying_404():
    client = make_client(callback_max_retries=2)
    count = [0]

    async def fake_post(url, content, headers):
        count[0] += 1
        return make_response(404)

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        with pytest.raises(httpx.HTTPStatusError):
            await client.send("run-1", "tool.completed", {"toolName": "x"})
    assert count[0] == 1


@pytest.mark.asyncio
async def test_callback_body_contains_all_required_fields():
    client = make_client()
    captured = [None]

    async def fake_post(url, content, headers):
        captured[0] = content
        return make_response(200)

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        await client.send("run-abc", "run.completed", {
            "content": "hello", "citations": [], "model": "gpt-5",
            "tools": [], "promptTokens": 10, "completionTokens": 5,
            "totalTokens": 15,
        })

    body = json.loads(captured[0])
    assert body["run_id"] == "run-abc"
    assert body["event_type"] == "run.completed"
    assert "event_id" in body
    assert body["data"]["content"] == "hello"
    assert body["data"]["promptTokens"] == 10
    assert body["data"]["totalTokens"] == 15


@pytest.mark.asyncio
async def test_callback_signature_header_is_set():
    client = make_client()
    captured = [None, None, None]

    async def fake_post(url, content, headers):
        captured[0] = url
        captured[1] = content
        captured[2] = headers
        return make_response(200)

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        await client.send("run-1", "run.started", {"status": "RUNNING"})

    url, body, headers = captured
    assert url == "http://java:8080/api/agent/deep-runs/callback/run-1"
    assert headers["X-Aether-Key-Id"] == "deep-agent-v1"
    assert headers["X-Aether-Signature"] == build_signature(
        "test-secret", headers["X-Aether-Timestamp"], body
    )


@pytest.mark.asyncio
async def test_run_completed_event_fields_match_contract():
    client = make_client()
    captured = [None]

    async def fake_post(url, content, headers):
        captured[0] = content
        return make_response(200)

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        await client.send("run-xyz", "run.completed", {
            "content": "分析结果", "citations": [], "model": "claude-5",
            "tools": ["search_knowledge"], "promptTokens": 120,
            "completionTokens": 80, "totalTokens": 200,
        })

    body = json.loads(captured[0])
    data = body["data"]
    for field in ("content", "citations", "model", "tools", "promptTokens", "completionTokens", "totalTokens"):
        assert field in data, f"缺少字段: {field}"


@pytest.mark.asyncio
async def test_skips_when_callback_base_url_is_empty():
    client = CallbackClient(Settings(shared_secret="test-secret", callback_base_url=""))

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        await client.send("run-1", "run.failed", {"error": "timeout"})

    post.assert_not_called()


@pytest.mark.asyncio
async def test_callback_retries_5xx_and_succeeds():
    client = make_client(callback_max_retries=2, callback_retry_backoff_seconds=0)
    responses = [make_response(503), make_response(200)]

    with patch("httpx.AsyncClient.post", side_effect=responses) as post:
        await client.send("run-1", "tool.completed", {"toolName": "search"})

    assert post.await_count == 2


@pytest.mark.asyncio
async def test_callback_retries_429_and_succeeds():
    client = make_client(callback_max_retries=2, callback_retry_backoff_seconds=0)
    responses = [make_response(429), make_response(200)]

    with patch("httpx.AsyncClient.post", side_effect=responses) as post:
        await client.send("run-1", "tool.completed", {"toolName": "search"})

    assert post.await_count == 2


@pytest.mark.asyncio
async def test_callback_retries_read_error_and_succeeds():
    client = make_client(callback_max_retries=2, callback_retry_backoff_seconds=0)
    responses = [
        httpx.ReadError("connection reset", request=httpx.Request("POST", "http://java:8080")),
        make_response(200),
    ]

    with patch("httpx.AsyncClient.post", side_effect=responses) as post:
        await client.send("run-1", "tool.completed", {"toolName": "search"})

    assert post.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [429, 503])
async def test_callback_raises_after_exhausting_retryable_http_failures(status_code):
    client = make_client(callback_max_retries=1, callback_retry_backoff_seconds=0)

    with patch("httpx.AsyncClient.post", return_value=make_response(status_code)) as post:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.send("run-1", "tool.completed", {"toolName": "search"})

    assert exc_info.value.response.status_code == status_code
    assert post.await_count == 2


def test_settings_reject_negative_callback_max_retries():
    with pytest.raises(ValidationError):
        Settings(callback_max_retries=-1)


def test_settings_reject_negative_callback_retry_backoff_seconds():
    with pytest.raises(ValidationError):
        Settings(callback_retry_backoff_seconds=-0.1)
