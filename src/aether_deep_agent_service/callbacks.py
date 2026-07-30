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
