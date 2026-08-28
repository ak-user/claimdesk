"""Search over the booklet: BM25, plus dense embeddings fused with Reciprocal Rank Fusion.

BM25 runs everywhere with no API key and is strong on exact policy terms ("crown", "12 months").
When an embedding model is configured, dense similarity catches paraphrases ("glasses" vs
"eyeglasses"), and RRF merges the two rankings without having to calibrate their scores.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import snowballstemmer

from .ingest import Chunk
from .llm import Embedder

log = logging.getLogger(__name__)

_STEMMER = snowballstemmer.stemmer("english")
TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by can could do does for from get got had has have how i if in into is "
    "it its me my of on or our per so than that the their them then there these they this to us "
    "was we were what when where which why will with would you your".split()
)
# Member vocabulary on the left, booklet vocabulary on the right.
SYNONYMS: dict[str, list[str]] = {
    "physio": ["physiotherapy"],
    "rmt": ["massage"],
    "chiro": ["chiropractic"],
    "glasses": ["eyeglasses", "vision"],
    "contacts": ["contact", "lenses"],
    "crown": ["major", "dental"],
    "crowns": ["major", "dental"],
    "tooth": ["dental"],
    "teeth": ["dental"],
    "dentist": ["dental"],
    "doctor": ["physician"],
    "kid": ["children"],
    "kids": ["children"],
    "son": ["children"],
    "daughter": ["children"],
    "wife": ["spouse"],
    "husband": ["spouse"],
    "partner": ["spouse"],
    "late": ["deadline"],
    "long": ["within"],
    "counseling": ["counselling"],
    "meds": ["prescription", "drugs"],
    "medication": ["prescription", "drugs"],
    "medicine": ["prescription", "drugs"],
    "refund": ["reimbursed"],
    "limit": ["maximum"],
    "max": ["maximum"],
}
RRF_K = 60


def tokenize(text: str, expand: bool = False) -> list[str]:
    words = [w for w in TOKEN.findall(text.lower()) if len(w) > 1 and w not in STOPWORDS]
    if expand:
        words += [syn for w in words for syn in SYNONYMS.get(w, [])]
    return _STEMMER.stemWords(words)


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(doc) for doc in docs]
        self.lengths = [len(doc) for doc in docs]
        self.avg_length = sum(self.lengths) / max(len(docs), 1)
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(set(doc))
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        terms = set(query)
        out = []
        for tf, length in zip(self.tf, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            out.append(
                sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            )
        return out


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None) -> None:
        self.chunks = chunks
        self._by_section = {c.section: c for c in chunks}
        # The title is repeated so a heading match outweighs a passing mention in another section.
        self._bm25 = BM25(
            [tokenize(f"{c.title} {c.title} {c.parent_title} {c.text}") for c in chunks]
        )
        self._embedder = embedder
        self._vectors: np.ndarray | None = None
        if embedder is not None:
            try:
                self._vectors = _normalize(
                    np.array(embedder.embed([_embed_text(c) for c in chunks]))
                )
            except Exception:  # noqa: BLE001 - any provider error means "no dense index"
                log.warning("Embedding the booklet failed; using BM25 only.", exc_info=True)

    @property
    def mode(self) -> str:
        return "hybrid" if self._vectors is not None else "bm25"

    def get(self, section: str) -> Chunk | None:
        return self._by_section.get(section)

    def search(self, query: str, k: int = 4) -> list[SearchHit]:
        lexical = self._bm25.scores(tokenize(query, expand=True))
        lexical_rank = [i for i in np.argsort(lexical)[::-1] if lexical[i] > 0]
        dense_rank = self._dense_rank(query)
        if dense_rank is None:
            return [SearchHit(self.chunks[i], round(lexical[i], 4)) for i in lexical_rank[:k]]
        fused: dict[int, float] = {}
        for ranking in (lexical_rank, dense_rank):
            for position, i in enumerate(ranking):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + position + 1)
        order = sorted(fused, key=fused.get, reverse=True)[:k]
        return [SearchHit(self.chunks[i], round(fused[i], 6)) for i in order]

    def _dense_rank(self, query: str) -> list[int] | None:
        if self._vectors is None or self._embedder is None:
            return None
        try:
            query_vector = _normalize(np.array(self._embedder.embed([query])))[0]
        except Exception:  # noqa: BLE001
            log.warning("Query embedding failed; using BM25 for this search.", exc_info=True)
            return None
        return [int(i) for i in np.argsort(self._vectors @ query_vector)[::-1]]


def _embed_text(chunk: Chunk) -> str:
    return f"{chunk.parent_title} > {chunk.title}\n{chunk.text}"


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)
