import asyncio
import logging
import time
import uuid

import httpx

from .schemas import CallbackEvent
from .security import build_signature
from .settings import Settings

logger = logging.getLogger(__name__)


class CallbackClient:
    """向 Java Admin 投递运行事件并获取 Agent 模型配置的客户端。"""

    def __init__(self, settings: Settings) -> None:
        """保存回调地址、签名密钥及重试配置。"""
        self.settings = settings

    async def send(self, run_id: str, event_type: str, data: dict) -> None:
        """创建回调事件并在配置了地址时投递。"""
        if not self.settings.callback_base_url:
            return
        event = CallbackEvent(
            event_id=str(uuid.uuid4()), event_type=event_type, run_id=run_id,
            occurred_at=int(time.time() * 1000), data=data,
        )
        await self.send_event(event)

    async def send_event(self, event: CallbackEvent) -> None:
        """投递已创建的回调事件。"""
        await self._send_with_retry(event.run_id, event)

    async def fetch_model_config(self, agent_id: str) -> dict | None:
        """从 Admin 拉取已解析的模型配置（model/baseUrl/apiKey）。

        apiKey 仅通过签名内部通道返回，调用方只在内存中使用，本服务不持久化。
        """
        if not self.settings.callback_base_url:
            return None
        url = f"{self.settings.callback_base_url.rstrip('/')}/api/agent/deep-runs/model-config/{agent_id}"
        timestamp = str(int(time.time()))
        signature = build_signature(self.settings.shared_secret, timestamp, b"")
        headers = {
            "X-Aether-Key-Id": self.settings.key_id,
            "X-Aether-Timestamp": timestamp,
            "X-Aether-Signature": signature,
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.callback_timeout_seconds) as client:
                response = await client.get(url, headers=headers)
            if response.status_code != 200:
                logger.warning("model config fetch failed: agentId=%s status=%s", agent_id, response.status_code)
                return None
            data = response.json()
            if not data.get("model"):
                return None
            return {
                "model": data["model"],
                "base_url": data.get("baseUrl"),
                "api_key": data.get("apiKey"),
            }
        except Exception:
            logger.exception("model config fetch error: agentId=%s", agent_id)
            return None

    async def _send_with_retry(self, run_id: str, event: CallbackEvent) -> None:
        """使用线性退避重试可恢复的 HTTP 回调失败。"""
        body = event.model_dump_json().encode("utf-8")
        headers_base = {
            "Content-Type": "application/json",
            "X-Aether-Key-Id": self.settings.key_id,
        }
        max_retries = self.settings.callback_max_retries
        backoff = self.settings.callback_retry_backoff_seconds
        url = f"{self.settings.callback_base_url.rstrip('/')}/api/agent/deep-runs/callback/{run_id}"

        for attempt in range(max_retries + 1):
            timestamp = str(int(time.time()))
            signature = build_signature(self.settings.shared_secret, timestamp, body)
            headers = {
                **headers_base,
                "X-Aether-Timestamp": timestamp,
                "X-Aether-Signature": signature,
            }
            try:
                async with httpx.AsyncClient(timeout=self.settings.callback_timeout_seconds) as client:
                    response = await client.post(url, content=body, headers=headers)
            except httpx.RequestError:
                if attempt == max_retries:
                    raise
            else:
                if response.is_success:
                    return
                if response.status_code != 429 and not response.is_server_error:
                    response.raise_for_status()
                if attempt == max_retries:
                    response.raise_for_status()
            if attempt < max_retries:
                await asyncio.sleep(backoff * (attempt + 1))
