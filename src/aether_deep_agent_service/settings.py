from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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
    return Settings()
