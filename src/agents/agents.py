"""The DSPy agents.

Mirrors the cash agent: a main code-generating ReAct investigator plus a
lighter summarizer that runs under its own model. Kept zero-shot for now (no
compiled optimizer), exactly like the cash agent's current state — the feedback
table and the eval metric below are the hooks to compile it later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dspy
import pandas as pd

from config.guardrails import CODE_GEN_GUARDRAILS, SUMMARIZER_GUARDRAILS
from src.execution.sandbox import ExecResult, execute
from src.patterns.detection_patterns import catalogue_for_prompt
from src.signatures.signatures import AMLInvestigator, AMLSummarizer, RegulatoryMapper


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {g}" for i, g in enumerate(items, 1))


@dataclass
class InvestigationOutput:
    python_code: str
    explanation: str
    exec_result: ExecResult


class Investigator(dspy.Module):
    """NL question -> pandas code -> sandboxed execution -> evidence.

    Uses dspy.ReAct so the model can reason/retry over tool calls; the single
    tool here is the sandboxed executor, which lets the agent see runtime errors
    and self-correct (the cash agent's code-gen-then-run loop).
    """

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(AMLInvestigator)

    def forward(
        self,
        question: str,
        frames: dict[str, pd.DataFrame],
        schema: str,
        entity_context: str = "None.",
        regulatory_context: str = "General UAE AML/CFT/CPF obligations apply.",
        risk_thresholds: str = "Defaults from RiskConfig.",
        max_retries: int = 2,
    ) -> InvestigationOutput:
        sample_rows = frames["tx"].head(3).to_dict("records") if "tx" in frames else []
        last_error = ""
        for attempt in range(max_retries + 1):
            q = question if attempt == 0 else (
                f"{question}\n\nYour previous code failed with:\n{last_error}\n"
                "Fix it. Same rules apply."
            )
            pred = self.generate(
                question=q,
                table_schemas=schema,
                available_patterns=catalogue_for_prompt(),
                entity_context=entity_context,
                regulatory_context=regulatory_context,
                risk_thresholds=risk_thresholds,
                sample_rows=str(sample_rows),
                guardrails=_numbered(CODE_GEN_GUARDRAILS),
            )
            res = execute(pred.python_code, frames)
            if res.ok:
                return InvestigationOutput(pred.python_code, pred.explanation, res)
            last_error = str(res.payload)
        return InvestigationOutput(pred.python_code, pred.explanation, res)


class Summarizer(dspy.Module):
    """Deep-insight briefing + DRAFT SAR narrative, on the summarizer model."""

    def __init__(self, summarizer_lm: dspy.LM | None = None) -> None:
        super().__init__()
        self.summarizer_lm = summarizer_lm
        self.summarize = dspy.ChainOfThought(AMLSummarizer)

    def forward(self, question: str, code: str, result_preview: str,
                regulatory_context: str,
                model_explanation: str = "No ARS explanation supplied.") -> dict[str, str]:
        ctx = dspy.context(lm=self.summarizer_lm) if self.summarizer_lm else _null_ctx()
        with ctx:
            pred = self.summarize(
                question=question,
                generated_code=code,
                result_preview=result_preview,
                regulatory_context=regulatory_context,
                model_explanation=model_explanation,
                guardrails=_numbered(SUMMARIZER_GUARDRAILS),
            )
        return {
            "insight": pred.insight,
            "sar_narrative_draft": pred.sar_narrative_draft,
            "recommended_action": pred.recommended_action,
            "data_quality_caveats": pred.data_quality_caveats,
        }


class PolicyMapper(dspy.Module):
    """Policy-loop hook: guidance excerpt -> proposed control change (suggestion)."""

    def __init__(self) -> None:
        super().__init__()
        self.map = dspy.ChainOfThought(RegulatoryMapper)

    def forward(self, guidance_excerpt: str, guidance_source: str,
                control_catalogue: str) -> dict[str, str]:
        pred = self.map(
            guidance_excerpt=guidance_excerpt,
            guidance_source=guidance_source,
            control_catalogue=control_catalogue,
        )
        return {
            "affected_controls": pred.affected_controls,
            "proposed_change": pred.proposed_change,
            "obligation_summary": pred.obligation_summary,
            "rationale": pred.rationale,
        }


class _null_ctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# --------------------------------------------------------------------------- #
# Evaluation metric for later DSPy compilation (currently unused, like the
# cash agent's zero-shot state). Wire the feedback table into a trainset and
# call dspy.teleprompt.* to optimize once you have labelled examples.
# --------------------------------------------------------------------------- #
def investigation_metric(example: Any, pred: InvestigationOutput, trace: Any = None) -> float:
    """Reward: code executed AND returned rows AND cited a driving flag."""
    if not pred.exec_result.ok:
        return 0.0
    cited_flag = "flag_" in (pred.python_code or "")
    produced = pred.exec_result.kind in {"dataframe", "figure"}
    return 1.0 if (cited_flag and produced) else 0.5
