"""Thin wrappers around the OpenAI SDK for Azure OpenAI and OpenAI.

The rest of the code depends on the small `ChatClient` and `Embedder` protocols, not on the SDK,
so tests can pass in fakes and the provider can change without touching the agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .config import Settings

log = logging.getLogger(__name__)

# Data-plane scope for keyless Azure OpenAI calls, per Microsoft's managed identity guide.
AZURE_SCOPE = "https://ai.azure.com/.default"


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatResult:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)


class ChatClient(Protocol):
    provider: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIChatClient:
    def __init__(self, client: Any, model: str, provider: str) -> None:
        self._client = client
        self.model = model
        self.provider = provider

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        import openai

        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as error:
            # Reasoning models reject `temperature`; retry once without it.
            if "temperature" not in str(error).lower():
                raise
            kwargs.pop("temperature")
            response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls = [
            ToolCallRequest(call.id, call.function.name, call.function.arguments or "{}")
            for call in (message.tool_calls or [])
            if getattr(call, "type", "function") == "function"
        ]
        return ChatResult(message.content, calls)


class OpenAIEmbedder:
    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def build_clients(settings: Settings) -> tuple[ChatClient | None, Embedder | None]:
    provider = settings.llm_provider
    if provider == "offline":
        return None, None

    from openai import OpenAI

    if provider == "azure":
        # Azure OpenAI's v1 API takes the standard OpenAI client and needs no api-version.
        # The model argument is the deployment name.
        client = OpenAI(
            base_url=azure_v1_url(settings.azure_openai_endpoint or ""),
            api_key=settings.azure_openai_api_key or _entra_token_provider(),
        )
        chat = OpenAIChatClient(client, settings.azure_openai_chat_deployment or "", "azure")
        embedding_model = settings.azure_openai_embedding_deployment
    else:
        client = OpenAI(api_key=settings.openai_api_key)
        chat = OpenAIChatClient(client, settings.openai_chat_model or "", "openai")
        embedding_model = settings.openai_embedding_model

    embedder = OpenAIEmbedder(client, embedding_model) if embedding_model else None
    return chat, embedder


def azure_v1_url(endpoint: str) -> str:
    """https://name.openai.azure.com -> https://name.openai.azure.com/openai/v1/"""
    base = endpoint.strip().rstrip("/")
    if not base.endswith("/openai/v1"):
        base += "/openai/v1"
    return base + "/"


def _entra_token_provider() -> Any:
    """Keyless auth: a managed identity in Azure, or `az login` on a developer machine."""
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except ImportError as error:  # pragma: no cover - depends on the optional extra
        raise RuntimeError(
            "Set AZURE_OPENAI_API_KEY, or install the Azure extra for keyless auth: "
            'pip install -e ".[azure]"'
        ) from error
    return get_bearer_token_provider(DefaultAzureCredential(), AZURE_SCOPE)
