"""Steps 2 & 3 — initial thresholds, then SAR-driven tuning.

Initial thresholds: per (scenario, segment), a percentile of the metric's
distribution (e.g. P95). Cheap, defensible starting point when no labels exist.

Tuning: sweep candidate thresholds (a percentile grid) against historical SAR
outcomes. For each candidate we know alerts and SARs captured *within the
alerted population*. Pick the HIGHEST threshold (fewest alerts) that still keeps
SAR capture at or above a recall floor relative to the most sensitive candidate.

Effectiveness beats efficiency: the recall floor is the constraint, productivity
is what you optimise. And because you can't see SARs you never alerted on,
every tuned threshold ships with a below-the-line (BTL) sampling plan — the
evidence a supervisor will ask for that the new threshold isn't hiding cases.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    metric: str
    description: str
    typology: str            # AML | CFT | CPF | TBML
    min_floor: float = 0.0   # regulatory/policy floor the threshold may never drop below


SCENARIO_REGISTRY: list[Scenario] = [
    Scenario("SC01_HV_CASH", "cash_deposit_total", "High-value cash deposits", "AML"),
    Scenario("SC02_STRUCT", "sub_threshold_cash_count", "Structuring below reporting figure", "AML", 2),
    Scenario("SC03_RAPID", "rapid_in_out_ratio", "Rapid in/out pass-through", "AML"),
    Scenario("SC04_HRGEO", "high_risk_geo_wire_total", "Wires to high-risk jurisdictions", "CFT"),
    Scenario("SC05_FAN", "distinct_counterparties", "Fan-in / fan-out counterparties", "AML"),
]
PERCENTILE_GRID = [80, 85, 90, 92.5, 95, 97, 98, 99, 99.5]


def initial_thresholds(cm: pd.DataFrame, pct: float = 95.0,
                       registry: list[Scenario] = SCENARIO_REGISTRY) -> pd.DataFrame:
    """Percentile-rule thresholds per (scenario, segment)."""
    rows = []
    for sc in registry:
        for seg, grp in cm.groupby("segment"):
            t = max(float(np.percentile(grp[sc.metric], pct)), sc.min_floor)
            rows.append({"scenario_id": sc.scenario_id, "segment": seg,
                         "metric": sc.metric, "threshold": t,
                         "method": f"P{pct:g}", "typology": sc.typology})
    return pd.DataFrame(rows)


def sweep(cm: pd.DataFrame, sc: Scenario, segment: str,
          grid: list[float] = PERCENTILE_GRID) -> pd.DataFrame:
    """ATL sweep: alerts, SARs captured and productivity for each candidate."""
    g = cm[cm.segment == segment]
    out = []
    for pct in grid:
        t = max(float(np.percentile(g[sc.metric], pct)), sc.min_floor)
        hit = g[sc.metric] >= t
        alerts, sars = int(hit.sum()), int((hit & (g.sar == 1)).sum())
        out.append({"pct": pct, "threshold": t, "alerts": alerts, "sars": sars,
                    "productivity": sars / alerts if alerts else 0.0})
    return pd.DataFrame(out)


def tune(cm: pd.DataFrame, recall_floor: float = 0.95,
         registry: list[Scenario] = SCENARIO_REGISTRY,
         min_sars_to_tune: int = 5) -> pd.DataFrame:
    """Pick the highest threshold keeping SAR capture >= recall_floor.

    Recall is relative to the most sensitive candidate (lowest percentile). If a
    segment has too few SARs to tune on, keep the conservative initial P95 and
    flag it — don't tune on noise.
    """
    rows = []
    for sc in registry:
        for seg in sorted(cm.segment.unique()):
            s = sweep(cm, sc, seg)
            base = s.sars.iloc[0]
            if base < min_sars_to_tune:
                pick = s[s.pct == 95].iloc[0]
                status = "INSUFFICIENT_SARS_KEPT_P95"
            else:
                ok = s[s.sars >= recall_floor * base]
                pick = ok.iloc[-1]        # highest threshold still meeting the floor
                status = "TUNED"
            rows.append({
                "scenario_id": sc.scenario_id, "segment": seg, "metric": sc.metric,
                "typology": sc.typology, "threshold": pick.threshold,
                "pct": pick.pct, "alerts": int(pick.alerts), "sars": int(pick.sars),
                "productivity": round(pick.productivity, 4),
                "relative_recall": round(pick.sars / base, 4) if base else None,
                "status": status,
            })
    return pd.DataFrame(rows)


def btl_sample_plan(cm: pd.DataFrame, tuned: pd.DataFrame, band_pct: float = 10.0,
                    confidence: float = 0.95, max_miss_rate: float = 0.01) -> pd.DataFrame:
    """Below-the-line plan: how many sub-threshold cases to review per cell.

    Band = customers between the (threshold - band_pct%) and the threshold.
    Sample size uses the zero-failure rule n = ln(1-conf)/ln(1-p): if n reviewed
    BTL cases yield zero SARs, you can claim the miss rate is below p at that
    confidence. Cap at the band population.
    """
    n_req = int(np.ceil(np.log(1 - confidence) / np.log(1 - max_miss_rate)))
    rows = []
    for r in tuned.itertuples():
        g = cm[cm.segment == r.segment]
        lo = r.threshold * (1 - band_pct / 100)
        band = int(((g[r.metric] >= lo) & (g[r.metric] < r.threshold)).sum())
        rows.append({"scenario_id": r.scenario_id, "segment": r.segment,
                     "band_population": band, "sample_size": min(n_req, band)})
    return pd.DataFrame(rows)


def generate_alerts(cm: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """Fire alerts: one row per (customer-month, scenario) above threshold."""
    out = []
    for r in thresholds.itertuples():
        g = cm[(cm.segment == r.segment) & (cm[r.metric] >= r.threshold)].copy()
        if g.empty:
            continue
        g["scenario_id"] = r.scenario_id
        g["typology"] = r.typology
        g["exceedance"] = g[r.metric] / max(r.threshold, 1e-9)
        out.append(g)
    alerts = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if not alerts.empty:
        alerts["n_scenarios_same_month"] = alerts.groupby(
            ["customer_id", "month"]).scenario_id.transform("nunique")
    return alerts
