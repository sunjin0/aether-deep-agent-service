import time

from aether_deep_agent_service.security import build_signature


def test_signature_is_stable() -> None:
    timestamp = str(int(time.time()))
    assert build_signature("secret", timestamp, b'{"runId":"run-1"}') == build_signature(
        "secret", timestamp, b'{"runId":"run-1"}'
    )
