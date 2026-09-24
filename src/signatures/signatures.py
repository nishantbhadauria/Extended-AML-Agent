"""DSPy signatures.

The cash agent had two signatures: Master_Agent_Prompt (code generation, 8
inputs) and Summarizer_Agent_Prompt (deep insight). This mirrors that and adds
a third, RegulatoryMapper, that feeds the policy-update loop — mapping a piece
of new guidance to the scenarios/thresholds it affects (a *suggestion*, always
human-approved before anything changes).
"""
from __future__ import annotations

import dspy


class AMLInvestigator(dspy.Signature):
    """Generate Python (pandas) that answers an AML/CFT/CPF investigation
    question over the provided DataFrames, returning evidence a human can audit.

    You are a compliance analytics assistant. You do NOT determine guilt; you
    surface typologies, risk indicators and anomalies. Follow every guardrail.
    Assign the final DataFrame or Plotly figure to a variable named `result`.
    """

    question: str = dspy.InputField(desc="Investigator's natural-language question.")
    table_schemas: str = dspy.InputField(desc="Columns + dtypes for tx, party, edges, alerts.")
    available_patterns: str = dspy.InputField(desc="Names + descriptions of pre-computed flag_ columns.")
    entity_context: str = dspy.InputField(desc="Known context on the party/case under review, if any.")
    regulatory_context: str = dspy.InputField(desc="Relevant obligations/typologies (from the policy store).")
    risk_thresholds: str = dspy.InputField(desc="Current versioned thresholds (structuring window, etc.).")
    sample_rows: str = dspy.InputField(desc="A few example rows so the model sees realistic values.")
    guardrails: str = dspy.InputField(desc="Numbered code-generation guardrails that MUST be obeyed.")

    python_code: str = dspy.OutputField(desc="Executable pandas/plotly code assigning `result`. No imports, no prints.")
    explanation: str = dspy.OutputField(desc="Plain-language explanation of what the code does and why.")


class AMLSummarizer(dspy.Signature):
    """Turn a query result into an auditable investigator briefing and a DRAFT
    suspicious-activity narrative. Never auto-file; recommend a next action from
    the fixed action set. Ground everything in the result.
    """

    question: str = dspy.InputField(desc="Original investigation question.")
    generated_code: str = dspy.InputField(desc="Code that produced the result (for transparency).")
    result_preview: str = dspy.InputField(desc="Head of the result DataFrame / description of the figure.")
    regulatory_context: str = dspy.InputField(desc="Obligations/typologies relevant to the finding.")
    model_explanation: str = dspy.InputField(
        desc="EBM (InterpretML) contribution breakdown for the alert's ARS score. "
             "Cite these drivers; never invent a different rationale.")
    guardrails: str = dspy.InputField(desc="Numbered summarization guardrails.")

    insight: str = dspy.OutputField(desc="Key findings, grounded in the result, with the driving evidence.")
    sar_narrative_draft: str = dspy.OutputField(desc="DRAFT who/what/when/where/why/how narrative for human review.")
    recommended_action: str = dspy.OutputField(desc="One of: escalate to L2 | request EDD | file STR/SAR draft | close as false positive | no action.")
    data_quality_caveats: str = dspy.OutputField(desc="Gaps (missing UBO, unresolved counterparty, stale screening).")


class RegulatoryMapper(dspy.Signature):
    """Policy-loop hook: given a piece of new/changed regulatory guidance, map it
    to the detection scenarios, thresholds, watchlists, and controls it plausibly
    affects, and PROPOSE a change. Output is a suggestion for a compliance officer
    to review — it never mutates a live rule automatically.
    """

    guidance_excerpt: str = dspy.InputField(desc="Excerpt from CBUAE guidance / Cabinet Decision / FATF update.")
    guidance_source: str = dspy.InputField(desc="Document id + version for the audit trail.")
    control_catalogue: str = dspy.InputField(desc="Current scenarios, thresholds, watchlists and their ids.")

    affected_controls: str = dspy.OutputField(desc="Ids of scenarios/thresholds/lists this guidance touches.")
    proposed_change: str = dspy.OutputField(desc="Concrete proposed edit (new scenario / threshold delta / list add).")
    obligation_summary: str = dspy.OutputField(desc="What the guidance requires, in one or two sentences.")
    rationale: str = dspy.OutputField(desc="Why the mapping holds, citing the guidance_source.")
