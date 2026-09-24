"""FastAPI service wrapping the DSPy agents.

Same shape as the cash agent: the agent functionality is exposed as endpoints
consumed by the Streamlit front end. Endpoints:

  POST /investigate   NL question over a case -> code + evidence
  POST /summarize     result -> insight + DRAFT SAR narrative + action
  POST /screen        real-time-ish name screen against watchlists (stub)
  POST /policy/map    guidance excerpt -> proposed control change (suggestion)
  POST /feedback      thumbs up/down + rationale -> Delta feedback table
  GET  /patterns      the pattern catalogue
  POST /rag/query     grounded compliance Q&A over the UAE regulatory corpus
  POST /ars/score     score alerts with the EBM ARS model -> route + explanation
  GET  /ars/global    global term importances (for the validation view)
  GET  /health
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import SETTINGS
from src.agents.agents import Investigator, PolicyMapper, Summarizer
from src.llm.gateway import configure_dspy
from src.patterns.detection_patterns import PATTERN_CATALOGUE, catalogue_for_prompt

app = FastAPI(title="AML/CFT/CPF Agent", version="0.1.0")

# Configure models once at startup. use_gateway=True for a corporate LLM gateway.
_LMS = configure_dspy(use_gateway=False)
_investigator = Investigator()
_summarizer = Summarizer(summarizer_lm=_LMS["summarizer"])
_policy = PolicyMapper()

# Lazy singletons so the API imports cleanly without live Databricks creds.
_sql = None
_rag = None
_ars = None


def _rag_engine():
    """ComplianceRAG over the configured vector backend (databricks | qdrant)."""
    global _rag
    if _rag is None:
        from src.rag import ingest
        from src.rag.agents import ComplianceRAG

        backend = os.environ.get("AML_VECTOR_BACKEND", "databricks")
        if backend == "databricks":
            store = ingest.DatabricksVectorSearchStore(
                os.environ["AML_VS_ENDPOINT"], os.environ["AML_VS_INDEX"])
            embedder = None       # managed-embedding index: query by text
        else:
            store = ingest.QdrantStore(os.environ.get("QDRANT_URL", "http://localhost:6333"))
            embedder = ingest.DatabricksEmbedder()
        _rag = ComplianceRAG(embedder, store)
    return _rag


def _ars_model():
    global _ars
    if _ars is None:
        from src.tm import registry

        path = os.environ.get("AML_ARS_MODEL_PATH")
        if path:
            _ars = registry.load_local(path)
        else:
            _ars = registry.load_from_mlflow(os.environ["AML_ARS_MODEL_URI"],
                                             cutoff=float(os.environ["AML_ARS_CUTOFF"]))
    return _ars


def _client():
    global _sql
    if _sql is None:
        from src.data.lakehouse import SqlClient
        _sql = SqlClient()
    return _sql


# --------------------------------- models ---------------------------------- #
class InvestigateReq(BaseModel):
    question: str
    party_id: str
    entity_context: str = "None."
    regulatory_context: str = ""            # empty -> retrieved from the RAG corpus


class SummarizeReq(BaseModel):
    question: str
    code: str
    result_preview: str
    regulatory_context: str = "General UAE AML/CFT/CPF obligations apply."
    model_explanation: str = "No ARS explanation supplied."


class RAGReq(BaseModel):
    question: str
    multi_agent: bool = True
    jurisdiction: str | None = None      # e.g. "UAE" or "UAE-DIFC"


class ARSScoreReq(BaseModel):
    alerts: list[dict[str, Any]]         # rows with the ARS_FEATURES columns
    explain_top_k: int = 5


class ScreenReq(BaseModel):
    full_name: str
    country: str | None = None


class PolicyMapReq(BaseModel):
    guidance_excerpt: str
    guidance_source: str
    control_catalogue: str


class FeedbackReq(BaseModel):
    session_id: str
    question: str
    code: str
    verdict: str            # "up" | "down"
    rationale: str = ""


# --------------------------------- routes ---------------------------------- #
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/patterns")
def patterns() -> dict[str, Any]:
    return {"catalogue": catalogue_for_prompt(), "count": len(PATTERN_CATALOGUE)}


@app.post("/investigate")
def investigate(req: InvestigateReq) -> dict[str, Any]:
    sql = _client()
    frames = sql.load_case_frames(req.party_id)
    reg_ctx = req.regulatory_context or _rag_engine().regulatory_context(req.question)
    out = _investigator.forward(
        question=req.question,
        frames=frames,
        schema=sql.schema_string(),
        entity_context=req.entity_context,
        regulatory_context=reg_ctx,
    )
    return {
        "python_code": out.python_code,
        "explanation": out.explanation,
        "ok": out.exec_result.ok,
        "kind": out.exec_result.kind,
        "payload": out.exec_result.payload,
        "stdout": out.exec_result.stdout,
    }


@app.post("/summarize")
def summarize(req: SummarizeReq) -> dict[str, str]:
    return _summarizer.forward(
        question=req.question, code=req.code,
        result_preview=req.result_preview, regulatory_context=req.regulatory_context,
        model_explanation=req.model_explanation,
    )


@app.post("/rag/query")
def rag_query(req: RAGReq) -> dict[str, Any]:
    filters = {"jurisdiction": req.jurisdiction} if req.jurisdiction else None
    return _rag_engine().forward(req.question, filters, req.multi_agent).__dict__


@app.post("/ars/score")
def ars_score(req: ARSScoreReq) -> dict[str, Any]:
    from src.tm.ars import ARS_FEATURES
    from src.tm.explain import explanation_text

    model = _ars_model()
    df = pd.DataFrame(req.alerts)
    missing = [c for c in ARS_FEATURES + ["typology"] if c not in df]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing columns: {missing}")
    scored = model.score(df)
    out = []
    for i, r in enumerate(scored.itertuples()):
        out.append({
            "ars_score": round(float(r.ars_score), 6),
            "route": r.route,
            "hard_override": bool(r.hard_override),
            "explanation": explanation_text(model.ebm, scored[ARS_FEATURES], i,
                                            float(r.ars_score), model.cutoff,
                                            top_k=req.explain_top_k),
        })
    return {"cutoff": model.cutoff, "results": out}


@app.get("/ars/global")
def ars_global() -> dict[str, Any]:
    from src.tm.explain import global_importance

    return {"importance": global_importance(_ars_model().ebm).to_dict("records")}


@app.post("/screen")
def screen(req: ScreenReq) -> dict[str, Any]:
    """Placeholder for the real-time screening plane. In production this calls
    the dedicated screening service (transliteration-aware fuzzy match against
    the versioned watchlists) — kept separate from the behavioural agent."""
    # TODO(FAB): call the screening microservice; return hits + list version ids.
    return {
        "name": req.full_name,
        "watchlists_checked": list(SETTINGS.watchlists),
        "hits": [],
        "note": "Stub — wire to the real-time screening service.",
    }


@app.post("/policy/map")
def policy_map(req: PolicyMapReq) -> dict[str, str]:
    return _policy.forward(
        guidance_excerpt=req.guidance_excerpt,
        guidance_source=req.guidance_source,
        control_catalogue=req.control_catalogue,
    )


@app.post("/feedback")
def feedback(req: FeedbackReq) -> dict[str, str]:
    try:
        _client().log_feedback(req.model_dump())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"feedback log failed: {e}") from e
    return {"status": "logged"}
