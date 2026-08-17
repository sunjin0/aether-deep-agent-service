from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境变量和 ``.env`` 文件加载服务运行配置。"""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AETHER_DEEP_AGENT_", extra="ignore"
    )

    shared_secret: str = ""
    key_id: str = "deep-agent-v1"
    callback_base_url: str = ""
    database_url: str = "postgresql+asyncpg://aether:aether_dev@postgres:5432/aether"
    model: str = ""
    mcp_url: str = ""
    max_steps: int = 12
    run_timeout_seconds: int = 600
    callback_timeout_seconds: float = 10.0
    callback_max_retries: int = Field(default=3, ge=0)
    callback_retry_backoff_seconds: float = Field(default=1.0, ge=0)


@lru_cache
def get_settings() -> Settings:
    """返回进程内缓存的服务配置实例。"""
    return Settings()
