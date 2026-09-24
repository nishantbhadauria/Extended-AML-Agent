"""Step 4 — Alert Risk Scoring (ARS) with an Explainable Boosting Machine.

Why EBM (InterpretML) rather than XGBoost + SHAP: an EBM is a glassbox GAM —
the score is literally intercept + sum of per-feature (and pairwise) shape
functions. The explanation IS the model, not a post-hoc approximation of it, so
model validation can inspect every shape curve and a regulator can be shown the
exact contribution that pushed an alert over the auto-close line. Accuracy is
typically on par with gradient boosting on tabular data.

Design:
  * Time-based split (train on older months, validate on the latest) — random
    splits leak customer behaviour across the boundary and flatter the model.
  * Auto-close cutoff chosen on validation as the highest score below which the
    share of SARs lost stays within `miss_tolerance`. The constraint is SAR
    leakage, not alert reduction.
  * Hard overrides that no score can auto-close: sanctions/PEP nexus, HIGH CRR,
    CFT/CPF typologies. The model triages; policy rules still govern.
  * Everything not auto-closed goes to an investigator — nothing reaches the FIU
    without human sign-off.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ARS_FEATURES = [
    "scenario_id", "segment", "crr", "exceedance", "n_scenarios_same_month",
    "tenure_months", "cross_border_ratio", "rapid_in_out_ratio",
    "sub_threshold_cash_count", "prior_alerts",
]
NEVER_AUTO_CLOSE_TYPOLOGIES = {"CFT", "CPF"}


def build_ars_frame(alerts: pd.DataFrame) -> pd.DataFrame:
    """Add history features. prior_alerts = alerts for the customer in earlier months."""
    a = alerts.sort_values("month").copy()
    per_month = a.groupby(["customer_id", "month"]).size().rename("n").reset_index()
    per_month["prior_alerts"] = per_month.groupby("customer_id").n.cumsum() - per_month.n
    return a.merge(per_month[["customer_id", "month", "prior_alerts"]],
                   on=["customer_id", "month"])


def hard_override(df: pd.DataFrame) -> pd.Series:
    """True where policy forbids auto-closure regardless of score."""
    ov = df.crr.eq("HIGH") | df.typology.isin(NEVER_AUTO_CLOSE_TYPOLOGIES)
    for col in ("screening_sanctions_hit", "screening_pep_hit"):
        if col in df:
            ov |= df[col].fillna(False).astype(bool)
    return ov


@dataclass
class ARSModel:
    """EBM alert risk score with a SAR-leakage-bounded auto-close cutoff."""
    miss_tolerance: float = 0.01        # max share of SARs allowed below the cutoff
    random_state: int = 7
    ebm_params: dict = field(default_factory=lambda: {"interactions": 5, "n_jobs": -2})
    ebm: ExplainableBoostingClassifier | None = None
    cutoff: float | None = None
    metrics: dict = field(default_factory=dict)

    def fit(self, alerts: pd.DataFrame, valid_months: int = 1) -> "ARSModel":
        """Train on older months, validate on the latest, and set the cutoff."""
        months = sorted(alerts.month.unique())
        split = months[-valid_months]
        tr, va = alerts[alerts.month < split], alerts[alerts.month >= split]

        self.ebm = ExplainableBoostingClassifier(
            random_state=self.random_state, **self.ebm_params)
        self.ebm.fit(tr[ARS_FEATURES], tr.sar)

        p = self.ebm.predict_proba(va[ARS_FEATURES])[:, 1]
        self.cutoff = self._choose_cutoff(p, va.sar.to_numpy())
        closable = (p < self.cutoff) & ~hard_override(va).to_numpy()
        n_sar = int(va.sar.sum())
        self.metrics = {
            "train_alerts": len(tr), "valid_alerts": len(va), "valid_sars": n_sar,
            "roc_auc": round(roc_auc_score(va.sar, p), 4) if n_sar else None,
            "pr_auc": round(average_precision_score(va.sar, p), 4) if n_sar else None,
            "brier": round(brier_score_loss(va.sar, p), 4),
            "cutoff": float(f"{self.cutoff:.4g}"),
            "auto_close_rate": round(float(closable.mean()), 4),
            "sars_auto_closed": int((closable & (va.sar == 1)).sum()),
            "sar_leakage": round(float((closable & (va.sar == 1)).sum() / max(n_sar, 1)), 4),
        }
        return self

    def _choose_cutoff(self, p: np.ndarray, y: np.ndarray) -> float:
        """Highest cutoff where SARs scoring below it <= miss_tolerance * SARs."""
        if y.sum() == 0:
            return 0.0
        allowed = int(np.floor(self.miss_tolerance * y.sum()))
        sar_scores = np.sort(p[y == 1])
        # Everything strictly below the (allowed)-th SAR score can be closed.
        return float(sar_scores[allowed])

    def score(self, alerts: pd.DataFrame) -> pd.DataFrame:
        """Add ars_score, hard_override and route columns."""
        out = alerts.copy()
        out["ars_score"] = self.ebm.predict_proba(out[ARS_FEATURES])[:, 1]
        out["hard_override"] = hard_override(out)
        out["route"] = np.where(
            (out.ars_score < self.cutoff) & ~out.hard_override,
            "AUTO_CLOSE", "INVESTIGATE")
        return out

    def calibration_table(self, alerts: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
        """Mean score vs observed SAR rate per score decile."""
        s = self.score(alerts)
        s["bin"] = pd.qcut(s.ars_score, q=bins, duplicates="drop")
        return s.groupby("bin", observed=True).agg(
            alerts=("sar", "size"), mean_score=("ars_score", "mean"),
            sar_rate=("sar", "mean")).round(4).reset_index()
