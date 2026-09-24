"""Synthetic customer-month data for developing and regression-testing the TM
process end to end (segmentation -> thresholds -> tuning -> ARS -> KPIs).

Shape mirrors the gold `cust_month` aggregate the notebooks read in Databricks:
one row per customer per month with scenario metrics and a SAR label. A small
latent "risky" population drives higher metrics and SAR probability so the
tuning and the EBM have real signal to find. Never use this for calibration of
production thresholds — it exists to prove the pipeline runs and behaves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIO_METRICS = [
    "cash_deposit_total",          # high-value cash
    "sub_threshold_cash_count",    # structuring
    "rapid_in_out_ratio",          # pass-through
    "high_risk_geo_wire_total",    # geography
    "distinct_counterparties",     # fan-in / fan-out
]


def generate(n_customers: int = 3000, n_months: int = 6, risky_share: float = 0.04,
             seed: int = 7) -> pd.DataFrame:
    """Customer-month rows with scenario metrics and a SAR label."""
    rng = np.random.default_rng(seed)
    cust = pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(n_customers)],
        "entity_type": rng.choice(["IND", "CORP"], n_customers, p=[0.8, 0.2]),
        "tenure_months": rng.integers(1, 240, n_customers),
        "crr": rng.choice(["LOW", "MEDIUM", "HIGH"], n_customers, p=[0.6, 0.3, 0.1]),
        "risky": rng.random(n_customers) < risky_share,
    })
    # Size factor: corporates and a long tail of wealthy individuals transact more.
    cust["size"] = np.where(cust.entity_type == "CORP",
                            rng.lognormal(12.5, 0.8, n_customers),
                            rng.lognormal(9.5, 1.0, n_customers))

    rows = []
    for m in range(n_months):
        month = pd.Timestamp("2026-01-01") + pd.DateOffset(months=m)
        noise = rng.lognormal(0, 0.35, n_customers)
        r = cust.risky.values
        total_credit = cust["size"].values * noise
        cash_share = np.clip(rng.beta(1.5, 8, n_customers) + r * 0.35, 0, 1)
        rows.append(pd.DataFrame({
            "customer_id": cust.customer_id,
            "month": month,
            "total_credit": total_credit,
            "txn_count": rng.poisson(8 + 20 * (cust.entity_type == "CORP").values + r * 15),
            "cash_deposit_total": total_credit * cash_share,
            "sub_threshold_cash_count": rng.poisson(0.3 + r * 3.5),
            "rapid_in_out_ratio": np.clip(rng.beta(2, 6, n_customers) + r * 0.4, 0, 1),
            "high_risk_geo_wire_total": total_credit * rng.beta(0.3, 20, n_customers)
                                        * (1 + r * 8),
            "distinct_counterparties": rng.poisson(4 + r * 12),
            "cross_border_ratio": np.clip(rng.beta(1, 6, n_customers) + r * 0.2, 0, 1),
        }))
    df = pd.concat(rows, ignore_index=True).merge(
        cust[["customer_id", "entity_type", "tenure_months", "crr", "risky"]],
        on="customer_id")

    # SAR label: mostly the risky population, occasionally elsewhere (noise).
    p = np.where(df.risky, 0.35, 0.002)
    p = p * np.where(df.crr == "HIGH", 1.5, 1.0)
    df["sar"] = (rng.random(len(df)) < np.clip(p, 0, 1)).astype(int)
    return df.drop(columns="risky")
