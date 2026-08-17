import hashlib
import hmac
import time

from fastapi import HTTPException, Request

from .settings import Settings


MAX_SIGNATURE_AGE_SECONDS = 300


def build_signature(secret: str, timestamp: str, body: bytes) -> str:
    """使用共享密钥为时间戳和请求体生成 HMAC-SHA256 签名。"""
    payload = timestamp.encode("utf-8") + b"." + body
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


async def verify_request_signature(
    request: Request,
    settings: Settings,
) -> None:
    """校验请求身份、签名时效和请求体 HMAC 完整性。"""
    key_id = request.headers.get("X-Aether-Key-Id")
    timestamp = request.headers.get("X-Aether-Timestamp")
    signature = request.headers.get("X-Aether-Signature")
    if not settings.shared_secret:
        raise HTTPException(status_code=503, detail="service shared secret is not configured")
    if key_id != settings.key_id or not timestamp or not signature:
        raise HTTPException(status_code=401, detail="missing or invalid service authentication headers")
    try:
        timestamp_value = int(timestamp)
    except ValueError as error:
        raise HTTPException(status_code=401, detail="invalid signature timestamp") from error
    # 限制签名时间窗口，防止截获的有效请求被长期重放。
    if abs(int(time.time()) - timestamp_value) > MAX_SIGNATURE_AGE_SECONDS:
        raise HTTPException(status_code=401, detail="expired request signature")
    expected = build_signature(settings.shared_secret, timestamp, await request.body())
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="invalid request signature")
