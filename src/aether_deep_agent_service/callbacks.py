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
        body = event.model_dump_json().encode("utf-8")
        timestamp = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            "X-Aether-Key-Id": self.settings.key_id,
            "X-Aether-Timestamp": timestamp,
            "X-Aether-Signature": build_signature(self.settings.shared_secret, timestamp, body),
        }
        async with httpx.AsyncClient(timeout=self.settings.callback_timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.callback_base_url.rstrip('/')}/api/agent/deep-runs/callback/{run_id}",
                content=body, headers=headers,
            )
            response.raise_for_status()
