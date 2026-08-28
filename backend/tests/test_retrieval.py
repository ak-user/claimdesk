import zlib

import pytest

from claimdesk.ingest import parse_booklet
from claimdesk.retrieval import Retriever, tokenize
from claimdesk.store import load_booklet


@pytest.mark.parametrize(
    ("query", "section"),
    [
        ("How long do I have to submit a claim?", "3.2"),
        ("Do I need a doctor's referral for physio?", "4.2"),
        ("What does the plan pay for a crown?", "5.2"),
        ("How much can I get for glasses?", "4.6"),
        ("Until what age are my kids covered?", "2.2"),
        ("What's the dental maximum?", "5.3"),
        ("How do I appeal a claim decision?", "8"),
        ("Are missed appointment fees covered?", "6"),
        ("My wife has another group plan. Where do I submit first?", "7"),
    ],
)
def test_bm25_finds_the_right_section(components, query, section):
    hits = components.retriever.search(query, k=3)
    assert section in [h.chunk.section for h in hits]


def test_synonyms_map_member_words_to_booklet_words():
    assert "physiotherapi" in tokenize("physio", expand=True)
    assert "eyeglass" in tokenize("glasses", expand=True)
    assert "physiotherapi" not in tokenize("physio")


def test_stopwords_and_single_letters_are_dropped():
    assert tokenize("How do I get a refund?") == ["refund"]


class HashEmbedder:
    """A deterministic bag-of-words embedder, so hybrid search runs without an API."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            vector = [0.0] * 128
            for token in tokenize(text, expand=True):
                vector[zlib.crc32(token.encode()) % 128] += 1.0
            vectors.append(vector)
        return vectors


class BrokenEmbedder:
    def embed(self, texts):
        raise RuntimeError("embedding endpoint unavailable")


def test_hybrid_search_fuses_dense_and_lexical_rankings():
    embedder = HashEmbedder()
    retriever = Retriever(parse_booklet(load_booklet()), embedder)
    assert retriever.mode == "hybrid"
    hits = retriever.search("How much can I get for glasses?", k=3)
    assert hits[0].chunk.section == "4.6"
    assert embedder.calls == 2  # once for the booklet, once for the query


def test_failed_embeddings_fall_back_to_bm25():
    retriever = Retriever(parse_booklet(load_booklet()), BrokenEmbedder())
    assert retriever.mode == "bm25"
    assert retriever.search("crown", k=1)[0].chunk.section == "5.2"


def test_no_match_returns_nothing(components):
    assert components.retriever.search("zebra xylophone", k=3) == []
