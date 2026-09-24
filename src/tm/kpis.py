"""Step 5 — KPIs and monitoring.

Two families:

  Effectiveness / efficiency KPIs (the TM programme's MI pack)
    - alert volume, SAR conversion (productivity) by scenario and segment
    - scenarios that never convert (candidates for retirement or redesign)
    - auto-close rate and SAR leakage through auto-close
    - SAR coverage: share of SARs caught by at least one scenario

  Model/population drift (for MRM ongoing monitoring)
    - PSI of the ARS score and key features, current vs reference period
      (< 0.1 stable, 0.1–0.25 watch, > 0.25 investigate/retune)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def scenario_kpis(alerts: pd.DataFrame) -> pd.DataFrame:
    k = alerts.groupby(["scenario_id", "segment"]).agg(
        alerts=("sar", "size"), sars=("sar", "sum")).reset_index()
    k["productivity"] = (k.sars / k.alerts).round(4)
    k["flag"] = np.where((k.alerts >= 50) & (k.sars == 0), "NON_PRODUCTIVE", "")
    return k.sort_values(["scenario_id", "segment"]).reset_index(drop=True)


def programme_kpis(cm: pd.DataFrame, alerts: pd.DataFrame,
                   scored: pd.DataFrame | None = None) -> dict:
    total_sars = int(cm.sar.sum())
    alerted_keys = alerts[["customer_id", "month"]].drop_duplicates()
    caught = cm.merge(alerted_keys, on=["customer_id", "month"]).sar.sum()
    out = {
        "customer_months": len(cm),
        "alerts": len(alerts),
        "alerted_customer_months": len(alerted_keys),
        "alert_rate": round(len(alerted_keys) / len(cm), 4),
        "sars_total": total_sars,
        "sar_coverage": round(float(caught / total_sars), 4) if total_sars else None,
        "overall_productivity": round(float(alerts.sar.mean()), 4) if len(alerts) else None,
    }
    if scored is not None and len(scored):
        ac = scored.route.eq("AUTO_CLOSE")
        out.update({
            "auto_close_rate": round(float(ac.mean()), 4),
            "sars_auto_closed": int((ac & scored.sar.eq(1)).sum()),
            "investigator_queue": int((~ac).sum()),
            "queue_productivity": round(float(scored.loc[~ac, "sar"].mean()), 4),
        })
    return out


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population stability index on quantile bins of the reference."""
    ref, cur = ref.dropna().to_numpy(), cur.dropna().to_numpy()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        v = psi(ref[c], cur[c])
        rows.append({"feature": c, "psi": round(v, 4),
                     "status": "STABLE" if v < 0.1 else "WATCH" if v < 0.25 else "INVESTIGATE"})
    return pd.DataFrame(rows)
