"""Multi-agent compliance RAG — the AML-MultiAgent-RAG pattern, in DSPy.

Fork (LangChain):  RAG agent -> Confidence agent -> Consistency agent -> Orchestrator
Here (DSPy):       same four roles, plus a *deterministic* citation gate that
                   runs before any LLM judge. An LLM grading its own citations is
                   weak evidence; a code check that every cited chunk id was
                   actually retrieved is not.

Quality gates (all must pass, else the answer is returned as
INSUFFICIENT_GROUNDING with the retrieved passages for a human to read):
  1. citation gate  — every [chunk_id] cited exists in the retrieved set, and
                      at least one is cited.
  2. confidence     — confidence agent score >= min_confidence.
  3. consistency    — consistency agent finds no contradiction with sources.

The same retriever feeds `regulatory_context()`, which the investigator,
summarizer and policy mapper consume — one grounded source of regulatory text
across the whole system.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import dspy

from src.rag.ingest import Chunk, Embedder, VectorStore


class ComplianceAnswer(dspy.Signature):
    """Answer a UAE AML/CFT/CPF compliance question using ONLY the passages.
    Cite every claim with the passage's [chunk_id]. If the passages don't answer
    the question, say so plainly. Do not give legal advice; describe obligations."""

    question: str = dspy.InputField()
    passages: str = dspy.InputField(desc="Numbered passages: [chunk_id] doc_id Art.N (version): text")
    answer: str = dspy.OutputField(desc="Answer with inline [chunk_id] citations.")
    jurisdictions: str = dspy.OutputField(desc="Jurisdictions the answer relies on, e.g. UAE, UAE-DIFC.")


class ConfidenceAssessment(dspy.Signature):
    """Rate how well the passages support the answer. Penalise hedging, gaps,
    and claims that rest on a single weak passage."""

    question: str = dspy.InputField()
    passages: str = dspy.InputField()
    answer: str = dspy.InputField()
    confidence: float = dspy.OutputField(desc="0.0–1.0")
    uncertainty_notes: str = dspy.OutputField()


class ConsistencyCheck(dspy.Signature):
    """Check the answer against the passages for contradictions, misattributed
    articles, or mixing jurisdictions (e.g. DIFC rules stated as federal)."""

    passages: str = dspy.InputField()
    answer: str = dspy.InputField()
    consistent: bool = dspy.OutputField()
    issues: str = dspy.OutputField(desc="Specific problems, or 'none'.")


@dataclass
class RAGResult:
    question: str
    answer: str
    status: str                       # GROUNDED | INSUFFICIENT_GROUNDING
    citations: list[str]
    sources: list[dict]
    confidence: float | None = None
    consistency_issues: str = ""
    gates: dict = field(default_factory=dict)


def format_passages(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(
        f"[{c.chunk_id}] {c.doc_id} Art.{c.article or '-'} ({c.version}): {c.text}"
        for c, _ in hits)


def citation_gate(answer: str, hits: list[tuple[Chunk, float]]) -> tuple[bool, list[str], list[str]]:
    cited = sorted(set(re.findall(r"\[([0-9a-f]{16})\]", answer)))
    known = {c.chunk_id for c, _ in hits}
    bogus = [c for c in cited if c not in known]
    return (bool(cited) and not bogus), cited, bogus


class ComplianceRAG(dspy.Module):
    def __init__(self, embedder: Embedder | None, store: VectorStore, k: int = 6,
                 min_confidence: float = 0.6) -> None:
        super().__init__()
        self.embedder, self.store, self.k, self.min_conf = embedder, store, k, min_confidence
        self.answer = dspy.ChainOfThought(ComplianceAnswer)
        self.confidence = dspy.Predict(ConfidenceAssessment)
        self.consistency = dspy.Predict(ConsistencyCheck)

    def retrieve(self, question: str, filters: dict | None = None) -> list[tuple[Chunk, float]]:
        if self.embedder is None and hasattr(self.store, "search_text"):
            return self.store.search_text(question, self.k, filters)   # managed embeddings
        return self.store.search(self.embedder.embed([question])[0], self.k, filters)

    def forward(self, question: str, filters: dict | None = None,
                multi_agent: bool = True) -> RAGResult:
        hits = self.retrieve(question, filters)
        sources = [{**c.meta(), "score": round(s, 4)} for c, s in hits]
        if not hits:
            return RAGResult(question, "No relevant passages found.", "INSUFFICIENT_GROUNDING",
                             [], [], gates={"retrieval": False})

        passages = format_passages(hits)
        ans = self.answer(question=question, passages=passages)
        ok_cite, cited, bogus = citation_gate(ans.answer, hits)
        gates = {"citation": ok_cite, "bogus_citations": bogus}
        result = RAGResult(question, ans.answer, "GROUNDED", cited, sources, gates=gates)

        if multi_agent and ok_cite:
            conf = self.confidence(question=question, passages=passages, answer=ans.answer)
            cons = self.consistency(passages=passages, answer=ans.answer)
            result.confidence = float(conf.confidence)
            result.consistency_issues = cons.issues
            gates["confidence"] = result.confidence >= self.min_conf
            gates["consistency"] = bool(cons.consistent)

        if not all(v for k, v in gates.items() if k != "bogus_citations"):
            result.status = "INSUFFICIENT_GROUNDING"
        return result

    def regulatory_context(self, query: str, k: int = 4) -> str:
        """Retrieved obligations as plain text for the other agents' prompts."""
        return format_passages(self.retrieve(query)[:k]) or "No regulatory passages retrieved."
