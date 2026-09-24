"""Hand-authored guardrails injected into the agent prompts.

The cash agent carried 44 code-generation guardrails and 12 summarization
guardrails. This is the AML/CFT/CPF equivalent starter set — extend it as your
compliance team reviews agent behaviour. Guardrails are *prompt-level* controls;
they are backed by the hard sandbox limits in execution/sandbox.py, never
trusted on their own.
"""
from __future__ import annotations

CODE_GEN_GUARDRAILS: list[str] = [
    # --- Data access & correctness ---
    "Only read from the DataFrames provided in the execution namespace "
    "(`tx`, `party`, `edges`, `alerts`). Never open files, sockets, or new "
    "database connections.",
    "Never write, update, delete, or drop any table. This tool is read-only. "
    "All output is a DataFrame or a Plotly figure assigned to `result`.",
    "Treat every amount as already in the account's base currency column "
    "`amount_base`; do not re-convert currencies yourself.",
    "Always filter on `is_reversed == False` unless the question is explicitly "
    "about reversals.",
    "When aggregating by party, group on `party_id` (the entity-resolved key), "
    "never on raw `account_id`.",
    "Prefer the pre-computed pattern flag columns (prefixed `flag_`) over "
    "re-deriving a typology from scratch; re-derive only if asked to explain how "
    "a flag is computed.",
    # --- AML/CFT/CPF domain rules ---
    "Do not assert that any party is guilty of a crime. Describe *typologies*, "
    "*risk indicators*, and *anomalies* only. Output is decision-support for a "
    "human investigator, not a determination.",
    "Never fabricate sanctions or PEP status. Use only the `flag_sanctions_nexus`, "
    "`flag_pep_nexus`, and `screening_*` columns already present.",
    "For trade-based money laundering questions, use trade fields "
    "(`invoice_value`, `goods_value_estimate`, `incoterm`, `counterparty_country`) "
    "and never infer shipment facts not present in the data.",
    "For proliferation-financing (CPF) questions, rely on "
    "`flag_pf_dualuse_nexus` and `counterparty_country` risk joins; do not "
    "speculate about specific goods.",
    # --- Output & explainability ---
    "Every result must be explainable: include the columns that justify why a "
    "row was selected (the driving flags and the numeric fields behind them).",
    "Cap any returned DataFrame at 1000 rows using `.head(1000)`; for charts, "
    "aggregate first — never plot per-transaction scatter over the full table.",
    "Assign the final DataFrame or Plotly figure to a variable named `result`. "
    "Do not call `print`, `display`, `.show()`, or `st.*`.",
    "Do not import any module. `pandas as pd`, `numpy as np`, and "
    "`plotly.express as px` are already available in the namespace.",
]

SUMMARIZER_GUARDRAILS: list[str] = [
    "Summaries are decision-support for a compliance investigator; never state "
    "or imply a legal conclusion of guilt.",
    "Ground every claim in the query result passed to you. Do not introduce "
    "facts, names, amounts, or dates that are not in the result.",
    "When drafting a suspicious-activity narrative, follow the who/what/when/"
    "where/why/how structure and mark it clearly as a DRAFT for human review "
    "before any goAML submission.",
    "Cite the specific pattern flags and numeric evidence that triggered the "
    "alert so the rationale is auditable.",
    "Recommend a next action from a fixed set: {escalate to L2, request EDD, "
    "file STR/SAR draft, close as false positive, no action}. Never auto-file.",
    "Flag data-quality caveats (missing UBO, unresolved counterparty, stale "
    "screening) that weaken confidence in the finding.",
    "Keep the narrative under 250 words unless the investigator asked for depth.",
    "When an ARS model explanation is supplied, state the score, the cutoff and the "
    "top contributing drivers exactly as given. Never attribute the score to a "
    "feature that is not in the explanation.",
]
