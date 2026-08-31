from types import SimpleNamespace

import openai
import pytest

from claimdesk import llm
from claimdesk.config import Settings
from claimdesk.llm import OpenAIChatClient, OpenAIEmbedder, azure_v1_url, build_clients

NO_PROVIDER = {
    "azure_openai_endpoint": None,
    "azure_openai_chat_deployment": None,
    "azure_openai_api_key": None,
    "openai_api_key": None,
    "openai_chat_model": None,
}


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **{**NO_PROVIDER, **overrides})


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://demo.openai.azure.com",
        "https://demo.openai.azure.com/",
        "https://demo.openai.azure.com/openai/v1/",
    ],
)
def test_azure_v1_url(endpoint):
    assert azure_v1_url(endpoint) == "https://demo.openai.azure.com/openai/v1/"


def test_no_settings_means_offline():
    assert build_clients(settings()) == (None, None)


def test_azure_with_a_key():
    chat, embedder = build_clients(
        settings(
            azure_openai_endpoint="https://demo.openai.azure.com",
            azure_openai_chat_deployment="chat",
            azure_openai_embedding_deployment="embedding",
            azure_openai_api_key="test-key",
        )
    )
    assert chat.provider == "azure"
    assert chat.model == "chat"
    assert str(chat._client.base_url) == "https://demo.openai.azure.com/openai/v1/"
    assert embedder.model == "embedding"


def test_azure_without_a_key_uses_entra_id(monkeypatch):
    calls = []

    def fake_provider():
        calls.append("provider")
        return lambda: "entra-token"

    monkeypatch.setattr(llm, "_entra_token_provider", fake_provider)
    chat, embedder = build_clients(
        settings(
            azure_openai_endpoint="https://demo.openai.azure.com",
            azure_openai_chat_deployment="chat",
        )
    )
    assert calls == ["provider"]
    assert chat.provider == "azure"
    assert embedder is None


def test_openai_needs_a_model_name():
    assert settings(openai_api_key="sk-test").llm_provider == "offline"
    chat, _ = build_clients(settings(openai_api_key="sk-test", openai_chat_model="small-model"))
    assert chat.provider == "openai" and chat.model == "small-model"


def completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        behaviour = self.behaviours.pop(0)
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour


def sdk(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def bad_request(message):
    response = SimpleNamespace(request=SimpleNamespace(), status_code=400, headers={})
    return openai.BadRequestError(message, response=response, body=None)


def test_chat_returns_tool_calls():
    call = SimpleNamespace(
        id="c1",
        type="function",
        function=SimpleNamespace(name="search_policy", arguments='{"query": "physio"}'),
    )
    completions = FakeCompletions([completion(None, [call])])
    result = OpenAIChatClient(sdk(completions), "chat", "azure").chat(
        [{"role": "user", "content": "hi"}], tools=[{"type": "function"}]
    )
    assert result.content is None
    assert result.tool_calls[0].name == "search_policy"
    assert completions.calls[0]["model"] == "chat"
    assert completions.calls[0]["temperature"] == 0


def test_chat_retries_without_temperature_for_models_that_reject_it():
    completions = FakeCompletions(
        [bad_request("Unsupported parameter: 'temperature'"), completion("ok")]
    )
    result = OpenAIChatClient(sdk(completions), "chat", "azure").chat([])
    assert result.content == "ok"
    assert "temperature" not in completions.calls[1]


def test_other_bad_requests_are_raised():
    completions = FakeCompletions([bad_request("messages must not be empty")])
    with pytest.raises(openai.BadRequestError):
        OpenAIChatClient(sdk(completions), "chat", "azure").chat([])


def test_embedder_returns_vectors_in_input_order():
    data = [
        SimpleNamespace(index=1, embedding=[0.0, 1.0]),
        SimpleNamespace(index=0, embedding=[1.0]),
    ]
    client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda model, input: SimpleNamespace(data=data))
    )
    assert OpenAIEmbedder(client, "embedding").embed(["a", "b"]) == [[1.0], [0.0, 1.0]]
