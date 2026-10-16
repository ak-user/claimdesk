"""Azure AI Search as the booklet index, behind the same interface as the local `Retriever`.

The local index rebuilds in memory on every start, which is fine for one booklet. With several
plans' booklets, or documents that change without a redeploy, the index belongs in a service.
Azure AI Search runs BM25 and vector search in one query and fuses them with RRF itself, so the
results match the local hybrid mode.

Build the index once, then point ClaimDesk at it:

    pip install -e ".[azure]"
    python -m claimdesk.azure_search      # creates the index and uploads the sections

If the service fails during a search, ClaimDesk answers from the local BM25 index and logs it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Protocol

from .ingest import Chunk
from .llm import Embedder
from .retrieval import Retriever, SearchHit, _embed_text

if TYPE_CHECKING:
    from .config import Settings

log = logging.getLogger(__name__)

VECTOR_FIELD = "vector"


class SearchClientLike(Protocol):
    """The part of `azure.search.documents.SearchClient` ClaimDesk uses."""

    def search(self, search_text: str, **kwargs: Any) -> Iterable[dict[str, Any]]: ...


class AzureSearchRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        client: SearchClientLike,
        embedder: Embedder | None = None,
        make_vector_query: Callable[[list[float], int], Any] | None = None,
    ) -> None:
        # The local index keeps `get` instant and is the fallback when the service fails.
        self._local = Retriever(chunks)
        self.chunks = chunks
        self._client = client
        self._embedder = embedder
        self._make_vector_query = make_vector_query or _sdk_vector_query

    @property
    def mode(self) -> str:
        return "azure-hybrid" if self._embedder is not None else "azure-bm25"

    def get(self, section: str) -> Chunk | None:
        return self._local.get(section)

    def search(self, query: str, k: int = 4) -> list[SearchHit]:
        kwargs: dict[str, Any] = {"top": k, "select": ["section"]}
        try:
            if self._embedder is not None:
                vector = self._embedder.embed([query])[0]
                kwargs["vector_queries"] = [self._make_vector_query(vector, k)]
            results = list(self._client.search(search_text=query, **kwargs))
        except Exception:  # noqa: BLE001 - any service error means "use the local index"
            log.warning("Azure AI Search failed; answering from the local index.", exc_info=True)
            return self._local.search(query, k)
        hits = []
        for result in results:
            chunk = self.get(str(result.get("section")))
            if chunk is None:
                # The index is older or newer than the booklet this build ships with.
                log.warning("Index returned unknown section %r; skipping.", result.get("section"))
                continue
            hits.append(SearchHit(chunk, round(float(result.get("@search.score", 0.0)), 6)))
        return hits[:k]


def documents(chunks: list[Chunk], vectors: list[list[float]] | None = None) -> list[dict]:
    """The index documents: one per section. Keys can't contain dots, so 4.2 becomes s4-2."""
    docs = []
    for i, chunk in enumerate(chunks):
        doc: dict[str, Any] = {
            "id": "s" + chunk.section.replace(".", "-"),
            "section": chunk.section,
            "title": chunk.title,
            "parent_title": chunk.parent_title,
            "text": chunk.text,
        }
        if vectors is not None:
            doc[VECTOR_FIELD] = vectors[i]
        docs.append(doc)
    return docs


def build_retriever(
    settings: Settings, chunks: list[Chunk], embedder: Embedder | None
) -> AzureSearchRetriever:
    from azure.search.documents import SearchClient

    client = SearchClient(
        settings.azure_search_endpoint or "", settings.azure_search_index, _credential(settings)
    )
    return AzureSearchRetriever(chunks, client, embedder)


def create_index(settings: Settings, chunks: list[Chunk], embedder: Embedder | None) -> int:
    """Creates or updates the index and uploads every section. Returns the number uploaded."""
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    vectors = embedder.embed([_embed_text(c) for c in chunks]) if embedder else None
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="section", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="title", analyzer_name="en.lucene"),
        SearchableField(name="parent_title", analyzer_name="en.lucene"),
        SearchableField(name="text", analyzer_name="en.lucene"),
    ]
    vector_search = None
    if vectors:
        fields.append(
            SearchField(
                name=VECTOR_FIELD,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=len(vectors[0]),
                vector_search_profile_name="default",
            )
        )
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="hnsw")],
        )
    credential = _credential(settings)
    endpoint = settings.azure_search_endpoint or ""
    index = SearchIndex(
        name=settings.azure_search_index, fields=fields, vector_search=vector_search
    )
    SearchIndexClient(endpoint, credential).create_or_update_index(index)
    client = SearchClient(endpoint, settings.azure_search_index, credential)
    client.upload_documents(documents=documents(chunks, vectors))
    return len(chunks)


def _sdk_vector_query(vector: list[float], k: int) -> Any:
    from azure.search.documents.models import VectorizedQuery

    return VectorizedQuery(vector=vector, k_nearest_neighbors=k, fields=VECTOR_FIELD)


def _credential(settings: Settings) -> Any:
    """An admin or query key if set, otherwise Microsoft Entra ID, as for Azure OpenAI."""
    if settings.azure_search_api_key:
        from azure.core.credentials import AzureKeyCredential

        return AzureKeyCredential(settings.azure_search_api_key)
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()


if __name__ == "__main__":  # pragma: no cover - needs a real service
    from .config import Settings
    from .ingest import parse_booklet
    from .llm import build_clients
    from .store import load_booklet

    logging.basicConfig(level=logging.INFO)
    current = Settings()
    if not current.azure_search_endpoint:
        raise SystemExit("Set AZURE_SEARCH_ENDPOINT first.")
    _, embedder = build_clients(current)
    count = create_index(current, parse_booklet(load_booklet()), embedder)
    print(f"Uploaded {count} sections to {current.azure_search_index}.")
