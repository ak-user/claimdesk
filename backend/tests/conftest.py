from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from claimdesk.app_state import Components, build_components
from claimdesk.config import Settings
from claimdesk.llm import ChatResult
from claimdesk.store import read_sample

AS_OF = date(2026, 10, 26)


@pytest.fixture(scope="session")
def settings() -> Settings:
    # Explicit values beat environment variables, so the suite is offline on any machine.
    return Settings(
        _env_file=None,
        azure_openai_endpoint=None,
        azure_openai_chat_deployment=None,
        openai_api_key=None,
        openai_chat_model=None,
        claimdesk_as_of=AS_OF,
    )


@pytest.fixture(scope="session")
def components(settings: Settings) -> Components:
    return build_components(settings)


@pytest.fixture
def sample():
    def load(name: str) -> str:
        text = read_sample(name)
        assert text is not None, name
        return text

    return load


class FakeChat:
    """Replays scripted model replies and records every request."""

    provider = "fake"

    def __init__(self, replies: list[ChatResult | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, response_format=None) -> ChatResult:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "response_format": response_format,
            }
        )
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
