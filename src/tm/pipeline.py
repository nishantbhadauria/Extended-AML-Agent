"""End-to-end TM process: segment -> initial thresholds -> SAR tuning -> alerts
-> ARS (EBM) -> KPIs. Each step returns plain DataFrames so the same functions
back the Databricks notebooks, the Dataiku recipes and the API.

Segmentation is fit on the reference window (all but the last month) and then
applied, so the latest month behaves like an out-of-time period for tuning and
ARS validation — the same discipline you'd apply in production.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.tm import ars, kpis, segmentation, thresholds


@dataclass
class TMArtifacts:
    """Everything one TM process run produces."""
    segment_model: segmentation.SegmentModel
    segment_profile: pd.DataFrame
    segment_stability: float
    initial_thresholds: pd.DataFrame
    tuned_thresholds: pd.DataFrame
    btl_plan: pd.DataFrame
    alerts: pd.DataFrame
    ars_model: ars.ARSModel
    scored_alerts: pd.DataFrame
    scenario_kpis: pd.DataFrame
    programme_kpis: dict
    drift: pd.DataFrame


def run(cm: pd.DataFrame, recall_floor: float = 0.95,
        miss_tolerance: float = 0.01, ebm_params: dict | None = None) -> TMArtifacts:
    """Run the full TM process on customer-month data."""
    months = sorted(cm.month.unique())
    ref_cm = cm[cm.month < months[-1]]

    # 1. Segmentation (fit on reference window, assign on full history)
    feats_ref = segmentation.customer_features(ref_cm)
    seg = segmentation.SegmentModel().fit(feats_ref)
    feats_ref["segment"] = seg.assign(feats_ref)
    feats_cur = segmentation.customer_features(cm[cm.month == months[-1]])
    feats_cur["segment"] = seg.assign(feats_cur)
    cm = cm.merge(feats_ref[["customer_id", "segment"]], on="customer_id", how="left")

    # 2–3. Thresholds: percentile init, then SAR-driven tuning + BTL plan
    init = thresholds.initial_thresholds(cm)
    tuned = thresholds.tune(cm, recall_floor=recall_floor)
    btl = thresholds.btl_sample_plan(cm, tuned)

    # 4. Alerts + ARS
    alerts = ars.build_ars_frame(thresholds.generate_alerts(cm, tuned))
    model = ars.ARSModel(miss_tolerance=miss_tolerance,
                         **({"ebm_params": ebm_params} if ebm_params else {})).fit(alerts)
    scored = model.score(alerts)

    # 5. KPIs + drift (reference months vs latest month)
    last = scored.month == months[-1]
    drift = kpis.drift_report(scored[~last], scored[last],
                              ["ars_score", "exceedance", "cross_border_ratio"])

    return TMArtifacts(
        segment_model=seg,
        segment_profile=segmentation.segment_profile(feats_ref),
        segment_stability=segmentation.stability(feats_ref, feats_cur),
        initial_thresholds=init, tuned_thresholds=tuned, btl_plan=btl,
        alerts=alerts, ars_model=model, scored_alerts=scored,
        scenario_kpis=kpis.scenario_kpis(alerts),
        programme_kpis=kpis.programme_kpis(cm, alerts, scored),
        drift=drift,
    )
