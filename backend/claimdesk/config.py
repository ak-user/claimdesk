"""Runtime settings read from environment variables or a `.env` file."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DATA_DIR = Path(__file__).parent / "data"

# The demo data is fixed in time, so "today" is fixed too. Override with CLAIMDESK_AS_OF.
DEFAULT_AS_OF = date(2026, 10, 26)

Provider = Literal["azure", "openai", "offline"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Azure OpenAI. Leave AZURE_OPENAI_API_KEY empty to sign in with Microsoft Entra ID
    # (managed identity in Azure, `az login` on a laptop).
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    # OpenAI, for people without an Azure subscription.
    openai_api_key: str | None = None
    openai_chat_model: str | None = None
    openai_embedding_model: str | None = None

    claimdesk_as_of: date = DEFAULT_AS_OF
    claimdesk_static_dir: Path | None = None
    claimdesk_cors_origins: list[str] = ["http://localhost:5173"]
    retrieval_top_k: int = Field(default=4, ge=1, le=10)
    agent_max_steps: int = Field(default=6, ge=1, le=12)

    @property
    def llm_provider(self) -> Provider:
        if self.azure_openai_endpoint and self.azure_openai_chat_deployment:
            return "azure"
        if self.openai_api_key and self.openai_chat_model:
            return "openai"
        return "offline"
