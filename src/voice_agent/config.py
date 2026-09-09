from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central, env-driven configuration. Validated at startup — fail fast."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    provider_api_key: str = Field(default="", alias="PROVIDER_API_KEY")
    provider_base_url: str = Field(
        default="https://api.openai.com/v1", alias="PROVIDER_BASE_URL"
    )
    provider_model: str = Field(default="gpt-4o-mini", alias="PROVIDER_MODEL")

    agent_max_tool_iterations: int = Field(default=6, alias="AGENT_MAX_TOOL_ITERATIONS")
    agent_temperature: float = Field(default=0.2, alias="AGENT_TEMPERATURE")

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")


@lru_cache
def get_settings() -> Settings:
    return Settings()
