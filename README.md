# Extended AML Agent

AML / CFT / CPF transaction monitoring, explainable alert scoring and compliance RAG for Databricks + Dataiku.

A DSPy-based analytics agent for anti-money-laundering, counter-terrorist-
financing, and counter-proliferation-financing investigations, structured the
same way as the Walmart **cash agent** but re-platformed onto **Databricks**
(Delta / Unity Catalog) with **Dataiku** for orchestration and governance.

It is decision-support for a human investigator. It never determines guilt and
never files an STR/SAR — every narrative it produces is a **draft for review**.

## How it maps to the cash agent

| Cash agent | This project |
|---|---|
| DSPy ReAct code-gen agent (`Master_Agent_Prompt`) | `AMLInvestigator` signature + `Investigator` module |
| Gemini summarizer (`Summarizer_Agent_Prompt`) | `AMLSummarizer` + `Summarizer` module (own LM) |
| Walmart LLM Gateway via custom LiteLLM provider + JWT | `src/llm/gateway.py` (`GatewayTokenProvider`) |
| 15 pre-computed fraud pattern flags | 16 AML/CFT/CPF typology flags in `patterns/` |
| Code executed in sandboxed namespace | `execution/sandbox.py` (AST check + subprocess + timeout) |
| BigQuery feedback log | Delta `agent_feedback` table |
| Streamlit: Overview / Chatbot / Patterns / Investigation | Same four pages + a Policy panel |
| 44 code-gen + 12 summarization guardrails | `config/guardrails.py` (starter set — extend) |
| Zero-shot (no optimizer) | Same; `investigation_metric` + feedback table are the hooks to compile later |

New vs the cash agent: a **`RegulatoryMapper`** signature / `PolicyMapper`
module that turns a piece of new guidance into a *proposed* control change for
human approval — the policy-update loop — and a **`/screen`** endpoint that
stands in for the separate real-time sanctions-screening plane.

## What was added in v0.2

**1. TM process layer (`src/tm/`, `notebooks/01–05`).** The full monitoring
lifecycle, runnable locally on synthetic data and on Databricks against the gold
`cust_month` table:

| Step | Module | What it does |
|---|---|---|
| Segmentation | `segmentation.py` | IND/CORP split → KMeans with silhouette-chosen k; plain-number profile for sign-off; migration matrix + stability |
| Initial thresholds | `thresholds.initial_thresholds` | P95 per (scenario, segment), floored by policy minimums |
| SAR-driven tuning | `thresholds.tune` | Highest threshold keeping SAR capture ≥ recall floor; too-few-SAR cells keep P95 and are flagged |
| BTL plan | `thresholds.btl_sample_plan` | Zero-failure sample size per cell to evidence the miss rate below the line |
| ARS | `ars.py` | InterpretML **EBM**, time-based validation, auto-close cutoff set by SAR-leakage tolerance, hard overrides (CFT/CPF, HIGH CRR, sanctions/PEP) |
| Explainability | `explain.py` | Global importances, shape functions, per-alert contributions, text for the LLM summarizer, LIME for vendor/black-box models |
| KPIs & drift | `kpis.py` | Scenario productivity, SAR coverage, auto-close rate/leakage, PSI drift |
| Persistence | `registry.py` | EBM + cutoff + metrics to MLflow / Unity Catalog; thresholds versioned in Delta |

Thresholds are only ever written as `PROPOSED`. Activation happens through the
Dataiku sign-off recipe (`dataiku/recipes.py::recipe_activate_thresholds`) — the
compute never promotes its own output.

**2. Compliance RAG (`src/rag/`, `notebooks/06`).** The AML-MultiAgent-RAG pattern
(RAG → confidence → consistency → orchestrator) ported from LangChain/Qdrant to
DSPy + Databricks Vector Search, with:
- article-aware chunking and document versioning for the UAE corpus
  (FDL 10/2025, CD 134/2025, CBUAE Apr-2026 guidance, DFSA module, FATF);
- a deterministic citation gate ahead of the LLM judges — every cited chunk id
  must have been retrieved;
- `regulatory_context()` feeding the investigator, summarizer and policy mapper
  from one grounded source.

**3. Explainability wired into the narrative.** `AMLSummarizer` now takes a
`model_explanation` input (the EBM contribution breakdown), and a guardrail
forbids attributing the score to anything not in it.

New endpoints: `POST /rag/query`, `POST /ars/score`, `GET /ars/global`.
New UI pages: Compliance Q&A, ARS explain.

## Why EBM for the alert score

An EBM is a generalised additive model with boosted shape functions:
score = intercept + Σ fᵢ(xᵢ) + Σ fᵢⱼ(xᵢ, xⱼ). The explanation *is* the model, so
there is no gap between what the validator reviews and what runs. The test suite
asserts this: intercept + contributions reconstructs every score exactly.
Accuracy on tabular data is generally close to gradient boosting.

## Layout

```
config/         settings + guardrails
src/llm/        DSPy model config + corporate-gateway provider
src/signatures/ DSPy signatures (investigator, summarizer, policy mapper)
src/agents/     the DSPy modules + eval metric
src/patterns/   PySpark typology flags (the pre-computed signals)
src/tm/         TM process: segmentation, thresholds, tuning, ARS (EBM), KPIs
src/rag/        regulatory ingestion + multi-agent compliance RAG
notebooks/      Databricks notebooks 00–06 (one per process step)
src/execution/  the sandboxed executor (enforcement boundary)
src/data/       Databricks lakehouse access + Delta feedback log
src/api/        FastAPI service
app/            Streamlit front end
dataiku/        Dataiku recipe / scenario scaffolds
tests/          sandbox, TM-process invariants, RAG gates (18 tests)
```

## Data model (gold tables the agent reads)

Conformed in the Silver→Gold layer; the agent binds four frames into the
sandbox: `tx` (fact_transaction, entity-resolved, with `flag_*` columns),
`party` (dim_party), `edges` (counterparty graph), `alerts`.

## Run locally

```bash
pip install -r requirements.txt

# 1. set env (LLM + Databricks) — see config/settings.py for the full list
export AML_LLM_API_BASE=...        AML_INVESTIGATOR_MODEL=azure/gpt-4o
export DATABRICKS_HOST=...         DATABRICKS_HTTP_PATH=...  DATABRICKS_TOKEN=...

# 2. API
uvicorn src.api.main:app --reload --port 8000

# 3. UI (separate shell)
export AML_API_BASE=http://localhost:8000
streamlit run app/streamlit_app.py
```

Try the process layer locally, no credentials needed:

```bash
python -c "from src.tm import synthetic, pipeline; a = pipeline.run(synthetic.generate()); print(a.programme_kpis)"
pytest -q
```

The pattern-scoring job runs *inside* Databricks (or as a Dataiku Spark recipe):
`python -c "from src.data.lakehouse import spark_apply_patterns; spark_apply_patterns()"`.

## Wiring checklist for FAB

- `src/llm/gateway.py::GatewayTokenProvider._mint_token` — FAB identity provider.
- `src/api/main.py::/screen` — connect the real-time screening microservice.
- `patterns/detection_patterns.py` — replace placeholder country lists and move
  thresholds under the versioned policy store (never hardcode in prod).
- Point Unity Catalog names in `config/settings.py::LakehouseConfig` at the real
  catalog/schema; put secrets in a Databricks secret scope / Dataiku variables.
- Replace synthetic data with the gold `cust_month` table; revisit the scenario
  registry and policy floors with compliance before any tuning run.
- Create the Vector Search endpoint + index (commented in `notebooks/06`) and set
  `AML_VS_ENDPOINT`, `AML_VS_INDEX`.
- Run the executor in an isolated container with no network before exposing it
  to real analyst traffic.

## Guardrail note

The prompt guardrails in `config/guardrails.py` are advisory. The real safety
boundary is `execution/sandbox.py`: static AST checks, a restricted builtins
set, a separate process, and a wall-clock timeout. Never rely on the prompt
alone to keep generated code safe.
```

## Acknowledgements

The compliance RAG layer (`src/rag/`) follows the multi-agent pattern of
[AML-MultiAgent-RAG](https://github.com/luuisotorres/AML-MultiAgent-RAG) by Luis
Fernando Torres (MIT) — RAG, confidence and consistency agents under an
orchestrator — reimplemented here in DSPy for Databricks Vector Search.
Alert risk scoring uses [InterpretML](https://github.com/interpretml/interpret).
