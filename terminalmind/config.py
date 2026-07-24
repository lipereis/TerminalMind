from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".terminalmind",
        validation_alias="TERMINALMIND_DATA_DIR",
    )
    log_level: str = Field(default="INFO", validation_alias="TERMINALMIND_LOG_LEVEL")
    max_context_chars: int = Field(
        default=12000,
        validation_alias="TERMINALMIND_MAX_CONTEXT_CHARS",
    )
    chunk_size: int = 800
    chunk_overlap: int = 100


def get_settings(**overrides: object) -> Settings:
    """Build settings; optional overrides used by CLI (`data_dir`, `log_level`)."""
    settings = Settings()
    return settings.model_copy(update=overrides) if overrides else settings
