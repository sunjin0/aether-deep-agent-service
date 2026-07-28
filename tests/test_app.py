import json
import time

from fastapi.testclient import TestClient

from aether_deep_agent_service.app import build_application
from aether_deep_agent_service.security import build_signature
from aether_deep_agent_service.settings import Settings


def test_health() -> None:
    app = build_application(Settings(shared_secret="test-secret", database_url="sqlite+aiosqlite://"))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_create_run_requires_valid_signature() -> None:
    settings = Settings(shared_secret="test-secret", database_url="sqlite+aiosqlite://")
    app = build_application(settings)
    payload = {
        "run_id": "run-1", "user_id": "user-1", "agent_id": "agent-1",
        "conversation_id": "conversation-1", "task": "Summarize the supplied evidence.",
        "delegation_token": "delegation-token",
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    timestamp = str(int(time.time()))
    headers = {
        "X-Aether-Key-Id": settings.key_id,
        "X-Aether-Timestamp": timestamp,
        "X-Aether-Signature": build_signature(settings.shared_secret, timestamp, body),
        "Content-Type": "application/json",
    }
    with TestClient(app) as client:
        response = client.post("/v1/runs", content=body, headers=headers)
    assert response.status_code == 202
    assert response.json()["run_id"] == "run-1"
    assert response.json()["created"] is True
