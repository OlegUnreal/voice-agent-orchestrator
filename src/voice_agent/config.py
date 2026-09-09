from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central, env-driven configuration. Validated at startup — fail fast."""

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", populate_by_name=True
    )

    provider_api_key: str = Field(default="", alias="PROVIDER_API_KEY")
    provider_base_url: str = Field(
        default="https://api.openai.com/v1", alias="PROVIDER_BASE_URL"
    )
    provider_model: str = Field(default="gpt-4o-mini", alias="PROVIDER_MODEL")

    agent_max_tool_iterations: int = Field(default=6, alias="AGENT_MAX_TOOL_ITERATIONS")
    agent_temperature: float = Field(default=0.2, alias="AGENT_TEMPERATURE")
    tool_timeout_s: float = Field(default=8.0, alias="TOOL_TIMEOUT_S")
    tool_allowlist: str = Field(default="", alias="TOOL_ALLOWLIST")
    auto_approve_sensitive: bool = Field(default=False, alias="AUTO_APPROVE_SENSITIVE")

    memory_max_messages: int = Field(default=24, alias="MEMORY_MAX_MESSAGES")
    session_token_budget: int = Field(default=8000, alias="SESSION_TOKEN_BUDGET")

    rag_min_score: float = Field(default=0.18, alias="RAG_MIN_SCORE")
    rag_top_k: int = Field(default=3, alias="RAG_TOP_K")

    input_token_usd_per_1m: float = Field(default=0.15, alias="INPUT_TOKEN_USD_PER_1M")
    output_token_usd_per_1m: float = Field(default=0.60, alias="OUTPUT_TOKEN_USD_PER_1M")

    stt_model: str = Field(default="whisper-1", alias="STT_MODEL")
    tts_model: str = Field(default="gpt-4o-mini-tts", alias="TTS_MODEL")
    tts_voice: str = Field(default="alloy", alias="TTS_VOICE")

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    def allowlisted_tools(self) -> set[str] | None:
        raw = [p.strip() for p in self.tool_allowlist.split(",") if p.strip()]
        return set(raw) if raw else None


@lru_cache
def get_settings() -> Settings:
    return Settings()
