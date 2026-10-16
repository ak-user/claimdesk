import pytest

from claimdesk.app_state import build_components
from claimdesk.azure_search import AzureSearchRetriever, documents
from claimdesk.ingest import parse_booklet
from claimdesk.store import load_booklet


@pytest.fixture(scope="module")
def chunks():
    return parse_booklet(load_booklet())


class FakeSearchClient:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def search(self, search_text, **kwargs):
        self.calls.append({"search_text": search_text, **kwargs})
        if self.error:
            raise self.error
        return iter(self.results)


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_results_map_to_booklet_sections(chunks):
    client = FakeSearchClient([{"section": "4.2", "@search.score": 3.5}, {"section": "3.2"}])
    retriever = AzureSearchRetriever(chunks, client)
    hits = retriever.search("physio maximum", k=3)
    assert [h.chunk.section for h in hits] == ["4.2", "3.2"]
    assert hits[0].score == 3.5
    assert client.calls == [{"search_text": "physio maximum", "top": 3, "select": ["section"]}]
    assert retriever.mode == "azure-bm25"


def test_hybrid_query_sends_a_vector(chunks):
    client = FakeSearchClient([{"section": "4.2", "@search.score": 0.03}])
    retriever = AzureSearchRetriever(
        chunks, client, FakeEmbedder(), make_vector_query=lambda vector, k: ("vq", vector, k)
    )
    retriever.search("physio", k=2)
    assert client.calls[0]["vector_queries"] == [("vq", [0.1, 0.2, 0.3], 2)]
    assert retriever.mode == "azure-hybrid"


def test_unknown_sections_are_skipped(chunks):
    client = FakeSearchClient([{"section": "99.9"}, {"section": "4.2"}])
    hits = AzureSearchRetriever(chunks, client).search("physio")
    assert [h.chunk.section for h in hits] == ["4.2"]


def test_service_errors_fall_back_to_the_local_index(chunks):
    client = FakeSearchClient(error=ConnectionError("503"))
    hits = AzureSearchRetriever(chunks, client).search("How long do I have to submit a claim?")
    assert "3.2" in [h.chunk.section for h in hits]


def test_get_uses_the_local_copy(chunks):
    retriever = AzureSearchRetriever(chunks, FakeSearchClient())
    assert retriever.get("4.2").title == chunks[[c.section for c in chunks].index("4.2")].title
    assert retriever.get("99.9") is None


def test_index_documents(chunks):
    docs = documents(chunks[:2], vectors=[[1.0], [2.0]])
    assert docs[0]["id"] == "s" + chunks[0].section.replace(".", "-")
    assert "." not in docs[1]["id"]
    assert docs[1]["vector"] == [2.0]
    assert "vector" not in documents(chunks[:1])[0]


def test_bad_search_config_falls_back_to_local(settings):
    update = {"azure_search_endpoint": "https://example.search.windows.net"}
    components = build_components(settings.model_copy(update=update))
    # Without the Azure SDK installed (as in CI), the app still starts on the local index.
    assert components.retriever.mode in {"bm25", "azure-bm25"}
