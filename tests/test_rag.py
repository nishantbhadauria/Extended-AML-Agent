"""RAG ingestion + deterministic citation gate (no LLM needed)."""
from __future__ import annotations

# pytest idioms: fixtures are injected by argument name; test names document intent.
# pylint: disable=missing-function-docstring

from src.rag.agents import citation_gate
from src.rag.ingest import HashingEmbedder, InMemoryStore, chunk_document

DOC = """Preamble text about definitions.
Article 1
Customer due diligence must be applied when establishing a business relationship.
Article 2
Enhanced due diligence applies to politically exposed persons and high-risk countries.
Article 3
Suspicious transaction reports are filed with the FIU through goAML without delay.
"""


def _store():
    chunks = chunk_document(DOC, "CD-TEST", "Test decision")
    store, emb = InMemoryStore(), HashingEmbedder()
    store.upsert(chunks, emb.embed([c.text for c in chunks]))
    return chunks, store, emb


def test_article_aware_chunking():
    chunks, _, _ = _store()
    assert [c.article for c in chunks] == [None, "1", "2", "3"]
    assert all("Article 2" not in c.text for c in chunks if c.article == "1")


def test_retrieval_finds_right_article():
    _, store, emb = _store()
    hits = store.search(emb.embed(["where are suspicious transaction reports filed"])[0], k=1)
    assert hits[0][0].article == "3"


def test_citation_gate_rejects_unretrieved_ids():
    _, store, emb = _store()
    hits = store.search(emb.embed(["enhanced due diligence"])[0], k=2)
    good = f"EDD applies to PEPs [{hits[0][0].chunk_id}]."
    bad = "EDD applies to PEPs [deadbeefdeadbeef]."
    assert citation_gate(good, hits)[0] is True
    ok, _, bogus = citation_gate(bad, hits)
    assert ok is False and bogus == ["deadbeefdeadbeef"]


def test_citation_gate_requires_a_citation():
    _, store, emb = _store()
    hits = store.search(emb.embed(["due diligence"])[0], k=2)
    assert citation_gate("No citations here.", hits)[0] is False
