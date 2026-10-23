"""Typed, secret-safe application settings.

Secrets are held as ``SecretStr`` and are never rendered by :func:`redacted_summary`.
Empty environment values are ignored so that blank placeholders in ``.env`` fall back
to defaults / "not configured" instead of becoming empty-string credentials.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

AppEnv = Literal["development", "test", "staging", "production"]

DEFAULT_LLM_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
DEFAULT_JUDGE_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


class ConfigurationError(RuntimeError):
    """Raised when configuration is unsafe or internally inconsistent."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
        case_sensitive=False,
    )

    # --- application ---
    app_env: AppEnv = "development"
    demo_mode: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- NVIDIA model gateway ---
    nvidia_api_key: SecretStr | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_model: str = DEFAULT_LLM_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    max_concurrency: int = Field(default=4, ge=1, le=32)

    # --- embeddings (Phase 5) ---
    embedding_provider: str | None = None
    embedding_model: str | None = None
    gemini_api_key: SecretStr | None = None

    # --- persistence ---
    database_url: SecretStr = SecretStr("postgresql+asyncpg://carepath:carepath@localhost:5432/carepath")
    db_connect_timeout_seconds: float = Field(default=5.0, gt=0, le=60)

    # --- Qdrant (Phase 5) ---
    qdrant_mode: Literal["local", "cloud", "disabled"] = "local"
    qdrant_url: str | None = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "carepath_evidence"

    # --- Neo4j (Phase 6) ---
    neo4j_enabled: bool = False
    neo4j_uri: str | None = "bolt://localhost:7687"
    neo4j_user: str | None = "neo4j"
    neo4j_password: SecretStr | None = None

    # --- search (Phase 4) ---
    searxng_base_url: str | None = None

    # --- observability ---
    langsmith_tracing: bool = False
    langsmith_endpoint: str | None = "https://api.smith.langchain.com"
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str | None = "carepath-research"
    logfire_token: SecretStr | None = None

    # --- UI <-> backend ---
    backend_url: str = "http://localhost:8000"
    api_bearer_token: SecretStr | None = None
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:8501"]
    max_request_bytes: int = Field(default=1_000_000, ge=1_024, le=50_000_000)

    # --- voice (Phase 8) ---
    voice_enabled: bool = False
    livekit_url: str | None = None
    livekit_api_key: SecretStr | None = None
    livekit_api_secret: SecretStr | None = None
    stt_provider: str | None = None
    stt_api_key: SecretStr | None = None
    tts_provider: str | None = None
    tts_api_key: SecretStr | None = None

    # --- jobs / research limits ---
    research_max_workers: int = Field(default=3, ge=1, le=16)
    research_max_steps: int = Field(default=12, ge=1, le=100)
    research_max_sources: int = Field(default=30, ge=1, le=200)
    job_timeout_seconds: int = Field(default=300, ge=10, le=3600)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("nvidia_base_url", "backend_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("must be an absolute http(s) URL")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _deployment_guards(self) -> Settings:
        if self.app_env in {"staging", "production"}:
            if self.api_bearer_token is None:
                raise ConfigurationError(
                    "API_BEARER_TOKEN is required when APP_ENV is staging/production."
                )
            if "*" in self.cors_origins:
                raise ConfigurationError("Wildcard CORS origins are not allowed outside development.")
        return self

    # ---- derived helpers (never return secret values) ----
    @property
    def model_configured(self) -> bool:
        return self.nvidia_api_key is not None and bool(self.nvidia_api_key.get_secret_value())

    @property
    def auth_required(self) -> bool:
        return self.api_bearer_token is not None

    @property
    def database_backend(self) -> str:
        return _db_backend(self.database_url.get_secret_value())

    def redacted_database_target(self) -> str:
        """Return ``driver://host:port/db`` without credentials."""
        parts = urlsplit(self.database_url.get_secret_value())
        if parts.scheme.startswith("sqlite"):
            return f"{parts.scheme}:///<local file>"
        host = parts.hostname or "?"
        port = f":{parts.port}" if parts.port else ""
        return f"{parts.scheme}://{host}{port}{parts.path}"


def _db_backend(url: str) -> str:
    scheme = urlsplit(url).scheme
    if scheme.startswith("postgresql"):
        return "postgresql"
    if scheme.startswith("sqlite"):
        return "sqlite"
    return scheme or "unknown"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings loaded from environment and ``.env``."""
    return Settings()
