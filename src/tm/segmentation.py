"""Step 1 — customer segmentation.

Thresholds are set per peer segment, so segmentation comes first. Approach:

  1. Hard business split by entity type (IND vs CORP never share thresholds).
  2. Within each split, cluster on behavioural features (log-scaled volume,
     velocity, cash share, cross-border share) using KMeans with k chosen by
     silhouette score inside a bounded range, so segments stay few and explainable.
  3. Profile every segment in plain numbers so compliance can name and sign off
     on it — an unexplainable segment will not survive a model review.
  4. Track segment migration between periods; heavy churn means thresholds tuned
     on last quarter's segments will misfire this quarter.

Pandas/sklearn here; at bank scale, compute the features in Spark and fit on a
stratified sample, then assign with the fitted model in a Spark UDF.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["log_credit", "log_txn", "cash_share", "cross_border_ratio"]


def customer_features(cust_month: pd.DataFrame) -> pd.DataFrame:
    """Aggregate customer-month rows into one behavioural profile per customer."""
    g = cust_month.groupby(["customer_id", "entity_type"], as_index=False).agg(
        total_credit=("total_credit", "mean"),
        txn_count=("txn_count", "mean"),
        cash_deposit_total=("cash_deposit_total", "mean"),
        cross_border_ratio=("cross_border_ratio", "mean"),
    )
    g["log_credit"] = np.log1p(g.total_credit)
    g["log_txn"] = np.log1p(g.txn_count)
    g["cash_share"] = (g.cash_deposit_total / g.total_credit.clip(lower=1)).clip(0, 1)
    return g


@dataclass
class SegmentModel:
    k_by_entity: dict[str, int] = field(default_factory=dict)
    scalers: dict[str, StandardScaler] = field(default_factory=dict)
    models: dict[str, KMeans] = field(default_factory=dict)
    silhouettes: dict[str, float] = field(default_factory=dict)

    def fit(self, feats: pd.DataFrame, k_range: range = range(2, 7),
            sample: int = 20_000, seed: int = 7) -> "SegmentModel":
        for ent, grp in feats.groupby("entity_type"):
            X = grp[FEATURES].to_numpy()
            if len(X) > sample:
                X = X[np.random.default_rng(seed).choice(len(X), sample, replace=False)]
            scaler = StandardScaler().fit(X)
            Xs = scaler.transform(X)
            best = (None, -1.0, None)
            for k in k_range:
                if k >= len(Xs):
                    break
                km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
                s = silhouette_score(Xs, km.labels_,
                                     sample_size=min(5000, len(Xs)), random_state=seed)
                if s > best[1]:
                    best = (k, s, km)
            self.k_by_entity[ent], self.silhouettes[ent], self.models[ent] = best
            self.scalers[ent] = scaler
        return self

    def assign(self, feats: pd.DataFrame) -> pd.Series:
        out = pd.Series(index=feats.index, dtype="object")
        for ent, grp in feats.groupby("entity_type"):
            Xs = self.scalers[ent].transform(grp[FEATURES].to_numpy())
            # Order clusters by mean volume so labels are stable and readable.
            order = np.argsort(self.models[ent].cluster_centers_[:, 0])
            rank = {c: i for i, c in enumerate(order)}
            labels = [rank[c] for c in self.models[ent].predict(Xs)]
            out.loc[grp.index] = [f"{ent}_S{l}" for l in labels]
        return out


def segment_profile(feats: pd.DataFrame) -> pd.DataFrame:
    """Plain-number profile per segment for compliance sign-off."""
    return feats.groupby("segment").agg(
        customers=("customer_id", "count"),
        median_monthly_credit=("total_credit", "median"),
        median_txn=("txn_count", "median"),
        median_cash_share=("cash_share", "median"),
        median_cross_border=("cross_border_ratio", "median"),
    ).round(3).reset_index()


def migration_matrix(prev: pd.DataFrame, curr: pd.DataFrame) -> pd.DataFrame:
    """Row-normalised transition matrix between two periods' segment labels."""
    m = prev[["customer_id", "segment"]].merge(
        curr[["customer_id", "segment"]], on="customer_id", suffixes=("_prev", "_curr"))
    return pd.crosstab(m.segment_prev, m.segment_curr, normalize="index").round(3)


def stability(prev: pd.DataFrame, curr: pd.DataFrame) -> float:
    """Share of customers who stayed in their segment (diagonal mass)."""
    m = prev[["customer_id", "segment"]].merge(
        curr[["customer_id", "segment"]], on="customer_id", suffixes=("_prev", "_curr"))
    return float((m.segment_prev == m.segment_curr).mean())
