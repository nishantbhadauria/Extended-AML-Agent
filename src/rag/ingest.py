"""Regulatory corpus ingestion — adapted from AML-MultiAgent-RAG.

Changes from the fork (which chunks US/EU/Brazil PDFs with a recursive
character splitter into Qdrant):

  * Article-aware chunking. UAE instruments are cited by article
    ("Article 16(2) of Cabinet Decision 134/2025"), so chunks carry the article
    number as metadata and never straddle two articles. Citations the answer
    agent produces can then be checked mechanically.
  * Document versioning. Every chunk carries doc_id + version + effective date,
    which the policy loop needs for its audit trail (which version of which
    guidance drove a threshold change).
  * Pluggable embeddings and store: Databricks Vector Search + a Databricks
    embedding endpoint in production; Qdrant (as in the fork) or in-memory for
    local work and tests.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

# Suggested UAE corpus (drop PDFs in docs/raw_docs/uae/ with this manifest).
UAE_CORPUS_MANIFEST = [
    {"doc_id": "FDL-10-2025", "title": "Federal Decree-Law No. 10 of 2025 (AML/CFT/CPF)",
     "effective": "2025-10-14", "jurisdiction": "UAE"},
    {"doc_id": "CD-134-2025", "title": "Cabinet Decision No. 134 of 2025 (Executive Regulations)",
     "effective": "2025-12-14", "jurisdiction": "UAE"},
    {"doc_id": "CBUAE-GL-2026-04", "title": "CBUAE AML/CFT/CPF guidance package (Apr 2026)",
     "effective": "2026-04-16", "jurisdiction": "UAE"},
    {"doc_id": "DFSA-AML-2026", "title": "DFSA AML Module (RMI 432 of 2026)",
     "effective": "2026-04-01", "jurisdiction": "UAE-DIFC"},
    {"doc_id": "FATF-RECS", "title": "FATF Recommendations", "effective": None,
     "jurisdiction": "INTL"},
]

_ARTICLE_RE = re.compile(r"(?im)^\s*(Article\s*\(?\s*(\d+)\s*\)?)")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    version: str
    effective: str | None
    jurisdiction: str
    article: str | None
    text: str

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


def load_pdf_text(path: str | Path) -> str:
    from pypdf import PdfReader

    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def split_by_article(text: str, max_chars: int = 2500, overlap: int = 200) -> list[tuple[str | None, str]]:
    """Split on 'Article N' headings; sub-split long articles with overlap.
    Text before the first article (preamble/definitions) gets article=None."""
    marks = list(_ARTICLE_RE.finditer(text))
    spans = []
    if not marks:
        spans.append((None, text))
    else:
        if marks[0].start() > 0:
            spans.append((None, text[: marks[0].start()]))
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            spans.append((m.group(2), text[m.start(): end]))
    out = []
    for art, body in spans:
        body = body.strip()
        if not body:
            continue
        step = max_chars - overlap
        for start in range(0, len(body), step):
            piece = body[start: start + max_chars]
            if piece.strip():
                out.append((art, piece))
            if start + max_chars >= len(body):
                break
    return out


def chunk_document(text: str, doc_id: str, title: str, version: str = "v1",
                   effective: str | None = None, jurisdiction: str = "UAE") -> list[Chunk]:
    chunks = []
    for art, piece in split_by_article(text):
        cid = hashlib.sha1(f"{doc_id}|{version}|{art}|{piece[:64]}".encode()).hexdigest()[:16]
        chunks.append(Chunk(cid, doc_id, title, version, effective, jurisdiction, art, piece))
    return chunks


# ------------------------------- embeddings -------------------------------- #
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Dependency-free bag-of-words hashing embedder for tests / offline dev."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        M = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                M[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.where(n == 0, 1, n)


class DatabricksEmbedder:
    """Databricks model-serving embedding endpoint (e.g. databricks-gte-large-en)."""

    def __init__(self, endpoint: str = "databricks-gte-large-en") -> None:
        from mlflow.deployments import get_deploy_client

        self.client, self.endpoint = get_deploy_client("databricks"), endpoint

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.predict(endpoint=self.endpoint, inputs={"input": texts})
        return np.array([d["embedding"] for d in resp["data"]], dtype=np.float32)


# ------------------------------- vector stores ----------------------------- #
class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray, k: int, filters: dict | None = None) -> list[tuple[Chunk, float]]: ...


class InMemoryStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.M: np.ndarray | None = None

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks += chunks
        self.M = vectors if self.M is None else np.vstack([self.M, vectors])

    def search(self, vector, k=5, filters=None):
        if self.M is None:
            return []
        sims = self.M @ vector
        idx = [i for i in np.argsort(-sims)
               if not filters or all(getattr(self.chunks[i], f) == v for f, v in filters.items())]
        return [(self.chunks[i], float(sims[i])) for i in idx[:k]]


class DatabricksVectorSearchStore:
    """Delta-sync index over a Delta table of chunks. Create the source table
    with the Chunk fields + an embedding column, then a Vector Search index on
    it; Databricks keeps the index in sync as documents are re-ingested."""

    def __init__(self, endpoint: str, index_name: str) -> None:
        from databricks.vector_search.client import VectorSearchClient

        self.index = VectorSearchClient().get_index(endpoint_name=endpoint, index_name=index_name)
        self.cols = list(Chunk.__dataclass_fields__)

    def upsert(self, chunks, vectors):
        raise NotImplementedError("Write chunks to the source Delta table; the index syncs.")

    def _rows(self, res):
        rows = res.get("result", {}).get("data_array", [])
        return [(Chunk(*r[: len(self.cols)]), float(r[-1])) for r in rows]

    def search(self, vector, k=5, filters=None):
        return self._rows(self.index.similarity_search(
            query_vector=list(map(float, vector)), columns=self.cols,
            num_results=k, filters=filters))

    def search_text(self, text: str, k=5, filters=None):
        """For managed-embedding indexes (embedding_source_column): Databricks
        embeds the query with the same endpoint used at index time."""
        return self._rows(self.index.similarity_search(
            query_text=text, columns=self.cols, num_results=k, filters=filters))


class QdrantStore:
    """Qdrant, as in the AML-MultiAgent-RAG fork — handy for local dev."""

    def __init__(self, url: str = "http://localhost:6333", collection: str = "uae-aml") -> None:
        from qdrant_client import QdrantClient

        self.client, self.collection = QdrantClient(url=url), collection

    def upsert(self, chunks, vectors):
        from qdrant_client.models import Distance, PointStruct, VectorParams

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=VectorParams(
                size=vectors.shape[1], distance=Distance.COSINE))
        self.client.upsert(self.collection, points=[
            PointStruct(id=int(c.chunk_id, 16) % (2**63), vector=v.tolist(),
                        payload={**c.meta(), "text": c.text})
            for c, v in zip(chunks, vectors)])

    def search(self, vector, k=5, filters=None):
        hits = self.client.query_points(self.collection, query=vector.tolist(), limit=k).points
        return [(Chunk(**{f: h.payload.get(f) for f in Chunk.__dataclass_fields__}), h.score)
                for h in hits]


def ingest(paths_and_meta: list[tuple[str, dict]], embedder: Embedder, store: VectorStore,
           version: str = "v1") -> int:
    """Load, chunk, embed and upsert a set of PDFs. Returns chunk count."""
    total = 0
    for path, meta in paths_and_meta:
        text = load_pdf_text(path)
        chunks = chunk_document(text, meta["doc_id"], meta["title"], version,
                                meta.get("effective"), meta.get("jurisdiction", "UAE"))
        store.upsert(chunks, embedder.embed([c.text for c in chunks]))
        total += len(chunks)
    return total
