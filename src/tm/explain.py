"""Explainability via InterpretML.

Three consumers, three outputs:

  * Model validation / MRM  -> global term importances + per-feature shape
    functions (`global_importance`, `shape_function`). These go in the
    validation pack.
  * Investigator            -> per-alert contribution breakdown (`local_explanation`)
    shown in the case view: which features pushed this alert up or down.
  * LLM summarizer          -> `explanation_text`, a compact, factual string fed
    into AMLSummarizer so the draft narrative cites the *model's actual reasons*
    instead of the LLM inventing a rationale.

Also `explain_blackbox` for any non-EBM model you're asked to validate (e.g. a
vendor score) using InterpretML's model-agnostic explainers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier


def global_importance(ebm: ExplainableBoostingClassifier) -> pd.DataFrame:
    """Mean absolute contribution per term (features and interaction pairs)."""
    return (pd.DataFrame({"term": ebm.term_names_, "importance": ebm.term_importances()})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def shape_function(ebm: ExplainableBoostingClassifier, feature: str) -> pd.DataFrame:
    """The learned curve for one main-effect feature (bins -> log-odds score)."""
    g = ebm.explain_global()
    idx = list(ebm.term_names_).index(feature)
    d = g.data(idx)
    names, scores = d["names"], d["scores"]
    if len(names) == len(scores) + 1:           # continuous: bin edges
        labels = [f"[{names[i]:.6g}, {names[i+1]:.6g})" for i in range(len(scores))]
    else:                                        # categorical: levels
        labels = [str(n) for n in names]
    return pd.DataFrame({"bin": labels, "log_odds": np.round(scores, 4)})


def local_explanation(ebm: ExplainableBoostingClassifier, X: pd.DataFrame,
                      i: int = 0) -> pd.DataFrame:
    """Per-term contribution (log-odds) for row i, sorted by absolute impact."""
    d = ebm.explain_local(X.iloc[[i]]).data(0)
    row = X.iloc[i]
    values = []
    for term, v in zip(d["names"], d["values"]):
        if " & " in term:                       # interaction: show both inputs
            a, b = term.split(" & ")
            values.append(f"{_fmt(row[a])} & {_fmt(row[b])}")
        else:
            values.append(_fmt(v))
    df = pd.DataFrame({"term": d["names"], "value": values, "contribution": d["scores"]})
    df["direction"] = np.where(df.contribution >= 0, "raises risk", "lowers risk")
    df = df.reindex(df.contribution.abs().sort_values(ascending=False).index)
    intercept = float(np.ravel(d["extra"]["scores"])[0])
    df.attrs["intercept"] = intercept
    return df.reset_index(drop=True)


def _fmt(v) -> str:
    return f"{v:.4g}" if isinstance(v, (int, float, np.integer, np.floating)) else str(v)


def explanation_text(ebm: ExplainableBoostingClassifier, X: pd.DataFrame, i: int,
                     score: float, cutoff: float, top_k: int = 5) -> str:
    """Compact, factual explanation for the LLM summarizer and the case file."""
    loc = local_explanation(ebm, X, i).head(top_k)
    lines = [f"ARS score {score:.4g} (auto-close cutoff {cutoff:.4g}); "
             f"baseline log-odds {loc.attrs['intercept']:.3f}. Top drivers:"]
    for r in loc.itertuples():
        lines.append(f"- {r.term} = {r.value}: {r.contribution:+.3f} log-odds ({r.direction})")
    return "\n".join(lines)


def explain_blackbox(model, X: pd.DataFrame, feature_names: list[str] | None = None,
                     n_rows: int = 5):
    """Model-agnostic local explanations (LIME-style via InterpretML) for a
    black-box/vendor model under validation. Returns InterpretML explanation
    objects; render with `interpret.show(...)` in a notebook."""
    from interpret.blackbox import LimeTabular

    lime = LimeTabular(model, X, feature_names=feature_names or list(X.columns))
    return lime.explain_local(X.iloc[:n_rows])
