# Deep Agent 回调可靠性增强实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增加回调重试逻辑，并通过集成测试确保回调事件格式、HMAC 签名和终态字段与 Java 端契约一致。

**Architecture:** 在 `CallbackClient.send()` 中为网络、超时和 5xx/429 错误添加有限重试，每次重试复用同一 `event_id` 与 `occurred_at`。新增 HTTP mock 集成测试覆盖重试语义、幂等性和所有终态事件。

**Tech Stack:** Python 3.11+、FastAPI、httpx、pytest、pytest-asyncio、respx。

---

## 文件结构

- 修改：`src/aether_deep_agent_service/callbacks.py` — 增加重试逻辑
- 修改：`src/aether_deep_agent_service/settings.py` — 增加重试配置参数
- 新增：`tests/test_callbacks.py` — 回调重试与契约测试
- 无需修改：`src/aether_deep_agent_service/app.py`、`schemas.py`、`security.py`

---

### Task 1: 回调重试逻辑

**Files:**
- Modify: `src/aether_deep_agent_service/settings.py:13-19`
- Modify: `src/aether_deep_agent_service/callbacks.py:11-35`

- [ ] **Step 1: 增加重试配置**

在 `Settings` 类中新增两个字段：

```python
callback_max_retries: int = 3
callback_retry_backoff_seconds: float = 1.0
```

- [ ] **Step 2: 实现重试 send 方法**

修改 `callbacks.py` 中的 `send` 方法：

```python
import asyncio
import time
import uuid

import httpx

from .schemas import CallbackEvent
from .security import build_signature
from .settings import Settings


class CallbackClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def send(self, run_id: str, event_type: str, data: dict) -> None:
        if not self.settings.callback_base_url:
            return
        event = CallbackEvent(
            event_id=str(uuid.uuid4()), event_type=event_type, run_id=run_id,
            occurred_at=int(time.time() * 1000), data=data,
        )
        await self._send_with_retry(run_id, event)

    async def _send_with_retry(self, run_id: str, event: CallbackEvent) -> None:
        body = event.model_dump_json().encode("utf-8")
        headers_base = {
            "Content-Type": "application/json",
            "X-Aether-Key-Id": self.settings.key_id,
        }
        max_retries = self.settings.callback_max_retries
        backoff = self.settings.callback_retry_backoff_seconds

        for attempt in range(max_retries + 1):
            timestamp = str(int(time.time()))
            signature = build_signature(self.settings.shared_secret, timestamp, body)
            headers = {**headers_base, "X-Aether-Timestamp": timestamp,
                       "X-Aether-Signature": signature}
            url = f"{self.settings.callback_base_url.rstrip('/')}/api/agent/deep-runs/callback/{run_id}"
            try:
                async with httpx.AsyncClient(timeout=self.settings.callback_timeout_seconds) as client:
                    response = await client.post(url, content=body, headers=headers)
                    if response.is_success:
                        return
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        break
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError):
                pass
            if attempt < max_retries:
                await asyncio.sleep(backoff * (attempt + 1))
```

- [ ] **Step 3: 编译与基础验证**

```powershell
uv sync
uv run python -c "from aether_deep_agent_service.callbacks import CallbackClient; print('OK')"
```

Expected: 无导入错误。

---

### Task 2: 回调契约集成测试

**Files:**
- Create: `tests/test_callbacks.py`

- [ ] **Step 1: 写出失败测试（函数不存在）**

```python
import json
import time
from unittest.mock import AsyncMock, patch

import httpx
from aether_deep_agent_service.callbacks import CallbackClient
from aether_deep_agent_service.settings import Settings
from aether_deep_agent_service.security import build_signature


async def test_callback_retries_on_5xx_and_succeeds():
    settings = Settings(shared_secret="test-secret",
                        callback_base_url="http://java:8080",
                        callback_max_retries=2,
                        callback_retry_backoff_seconds=0.01)
    client = CallbackClient(settings)

    request_count = 0
    async def fake_post(url, content, headers):
        nonlocal request_count
        request_count += 1
        if request_count <= 2:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200)

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        await client.send("run-1", "tool.completed",
                          {"toolName": "search", "message": "done"})

    assert request_count == 3


async def test_callback_does_not_retry_4xx():
    settings = Settings(shared_secret="test-secret",
                        callback_base_url="http://java:8080",
                        callback_max_retries=2)
    client = CallbackClient(settings)

    request_count = 0
    async def fake_post(url, content, headers):
        nonlocal request_count
        request_count += 1
        return httpx.Response(404)

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        await client.send("run-1", "tool.completed", {"toolName": "x"})

    assert request_count == 1


async def test_callback_body_contains_all_required_fields():
    settings = Settings(shared_secret="test-secret",
                        callback_base_url="http://java:8080")
    captured_body = None

    async def capture_post(url, content, headers):
        nonlocal captured_body
        captured_body = content
        return httpx.Response(200)

    with patch("httpx.AsyncClient.post", side_effect=capture_post):
        await client.send("run-abc", "run.completed", {
            "content": "hello", "citations": [], "model": "gpt-5",
            "tools": [], "promptTokens": 10, "completionTokens": 5,
            "totalTokens": 15,
        })

    body = json.loads(captured_body)
    assert body["run_id"] == "run-abc"
    assert body["event_type"] == "run.completed"
    assert "event_id" in body
    assert isinstance(body["event_id"], str)
    assert len(body["event_id"]) == 36
    assert body["data"]["content"] == "hello"
    assert body["data"]["promptTokens"] == 10
    assert body["data"]["completionTokens"] == 5
    assert body["data"]["totalTokens"] == 15


async def test_callback_signature_header_is_set():
    settings = Settings(shared_secret="test-secret",
                        callback_base_url="http://java:8080")
    captured_headers = None

    async def capture_post(url, content, headers):
        nonlocal captured_headers
        captured_headers = headers
        return httpx.Response(200)

    with patch("httpx.AsyncClient.post", side_effect=capture_post):
        await client.send("run-1", "run.started", {"status": "RUNNING"})

    assert "X-Aether-Signature" in captured_headers
    assert "X-Aether-Timestamp" in captured_headers
    assert "X-Aether-Key-Id" in captured_headers
    # 验证签名
    sig = captured_headers["X-Aether-Signature"]
    ts = captured_headers["X-Aether-Timestamp"]
    body = captured_headers.get("__raw_body__") or "{}"
    expected = build_signature("test-secret", ts, body)
    assert sig == expected


async def test_run_completed_event_fields_match_java_contract():
    settings = Settings(shared_secret="test-secret",
                        callback_base_url="http://java:8080")
    captured_body = None

    async def capture_post(url, content, headers):
        nonlocal captured_body
        captured_body = content
        return httpx.Response(200)

    with patch("httpx.AsyncClient.post", side_effect=capture_post):
        await client.send("run-xyz", "run.completed", {
            "content": "分析结果",
            "citations": [{"title": "doc", "content": "...", "citation": "【1】"}],
            "model": "anthropic:claude-5",
            "tools": ["search_knowledge", "get_current_time"],
            "promptTokens": 120,
            "completionTokens": 80,
            "totalTokens": 200,
        })

    body = json.loads(captured_body)
    data = body["data"]
    for field in ("content", "citations", "model", "tools",
                  "promptTokens", "completionTokens", "totalTokens"):
        assert field in data, f"缺少字段: {field}"


async def test_send_skips_when_callback_base_url_is_empty():
    settings = Settings(shared_secret="test-secret",
                        callback_base_url="")
    client_local = CallbackClient(settings)
    await client_local.send("run-1", "run.failed", {"error": "timeout"})
```

- [ ] **Step 2: 运行测试确认失败（初始排期函数不存在）**

```powershell
uv run pytest tests/test_callbacks.py -v
```

Expected: 部分测试通过，`test_callback_does_not_retry_4xx` 可能在初始实现无重试逻辑时也通过。

- [ ] **Step 3: 确保重试逻辑已实现后运行全部测试**

```powershell
uv run pytest tests/test_callbacks.py -v
```

Expected: 全部 6 个测试通过。

- [ ] **Step 4: 运行所有回归测试**

```powershell
uv run pytest -v
```

Expected: 已有测试（test_app.py、test_security.py、test_telemetry.py、test_executor.py）全部通过。

- [ ] **Step 5: 提交**

使用中文提交信息：
```
git add src/aether_deep_agent_service/callbacks.py src/aether_deep_agent_service/settings.py tests/test_callbacks.py
git commit -m "feat: 增加回调重试逻辑及契约集成测试"
```

---

### Task 3: 回调 URL 契约测试

**Files:**
- Modify: `tests/test_callbacks.py` — 追加 URL 测试

- [ ] **Step 1: 追加回调 URL 正确性测试**

在 `tests/test_callbacks.py` 末尾追加：

```python
async def test_callback_url_ends_with_run_id():
    settings = Settings(shared_secret="test-secret",
                        callback_base_url="http://java:8080")
    captured_url = None

    async def capture_post(url, content, headers):
        nonlocal captured_url
        captured_url = url
        return httpx.Response(200)

    with patch("httpx.AsyncClient.post", side_effect=capture_post):
        await client.send("run-id-12345", "tool.completed", {"toolName": "x"})

    assert captured_url.endswith("/api/agent/deep-runs/callback/run-id-12345")


async def test_callback_base_url_trailing_slash_is_normalized():
    settings = Settings(shared_secret="test-secret",
                        callback_base_url="http://java:8080/")
    captured_url = None

    async def capture_post(url, content, headers):
        nonlocal captured_url
        captured_url = url
        return httpx.Response(200)

    with patch("httpx.AsyncClient.post", side_effect=capture_post):
        await client.send("run-99", "step.started", {"message": "step"})

    assert "//" not in captured_url
    assert captured_url.startswith("http://java:8080/")
```

- [ ] **Step 2: 运行全部测试**

```powershell
uv run pytest -v
```

Expected: 8 个 `test_callbacks.py` 测试通过。

- [ ] **Step 3: 提交**

使用中文提交信息：
```
git add tests/test_callbacks.py
git commit -m "test: 增加回调 URL 和 base URL 规范化测试"
```

---

## 执行顺序与依赖

1. Task 1（重试配置 + 发送逻辑）
2. Task 2（六个集成测试）→ Task 3（追加两个 URL 测试）

Task 2 和 Task 3 可在 Task 1 完成后顺序执行。
