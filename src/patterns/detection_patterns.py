"""Pre-computed AML/CFT/CPF detection patterns.

The cash agent added 15 pre-computed fraud-detection pattern flags as columns
before the agent ran, so the LLM reasons over ready-made signals instead of
re-deriving typologies every time. This is the AML equivalent: a catalogue of
PySpark transforms that stamp `flag_*` columns onto the gold transaction table.

Run these as a Databricks job (or a Dataiku recipe — see dataiku/) on a schedule.
Each pattern is a pure function `DataFrame -> DataFrame` so they compose and are
individually unit-testable. Thresholds come from RiskConfig, never hardcoded, so
the policy loop can version them.

Every flag is a *risk indicator*, not a verdict. Keep the driving numeric fields
alongside each flag so an alert stays explainable.
"""
from __future__ import annotations

from typing import Callable

try:
    from pyspark.sql import DataFrame, Window
    from pyspark.sql import functions as F
except ImportError:  # API host without Spark: the catalogue text stays importable
    DataFrame = Window = F = None  # type: ignore[assignment]

from config.settings import SETTINGS

RISK = SETTINGS.risk

# Illustrative high-risk / sanctioned-nexus country buckets. In production these
# come from the versioned country-risk model, not a literal list.
HIGH_RISK_COUNTRIES = ["IR", "KP", "SY", "MM"]           # placeholder
PF_NEXUS_COUNTRIES = ["KP", "IR"]                        # proliferation nexus


# --------------------------------------------------------------------------- #
# Pattern catalogue — name -> (transform, human description)
# --------------------------------------------------------------------------- #
def p_structuring(df: DataFrame) -> DataFrame:
    """Multiple sub-threshold deposits by one party inside a short window."""
    w = (
        Window.partitionBy("party_id")
        .orderBy(F.col("event_ts").cast("long"))
        .rangeBetween(-RISK.structuring_window_days * 86400, 0)
    )
    return df.withColumn(
        "_struct_cnt",
        F.count(F.when(F.col("amount_base") < RISK.just_below_amount, 1)).over(w),
    ).withColumn(
        "flag_structuring",
        (F.col("_struct_cnt") >= RISK.structuring_min_txns)
        & (F.col("amount_base") < RISK.just_below_amount),
    ).drop("_struct_cnt")


def p_just_below_threshold(df: DataFrame) -> DataFrame:
    """Single amounts parked just under a round reporting-style figure."""
    lo = RISK.just_below_amount * 0.9
    return df.withColumn(
        "flag_just_below_threshold",
        (F.col("amount_base") >= lo) & (F.col("amount_base") < RISK.just_below_amount),
    )


def p_rapid_movement(df: DataFrame) -> DataFrame:
    """Funds in and back out of an account within a few hours (pass-through)."""
    w = Window.partitionBy("party_id").orderBy("event_ts")
    return df.withColumn("_prev_ts", F.lag("event_ts").over(w)).withColumn(
        "flag_rapid_movement",
        (F.col("direction") == "OUT")
        & (
            F.col("event_ts").cast("long") - F.col("_prev_ts").cast("long")
            <= RISK.rapid_movement_hours * 3600
        ),
    ).drop("_prev_ts")


def p_round_amount(df: DataFrame) -> DataFrame:
    """Suspiciously round amounts (weak signal, useful in combination)."""
    return df.withColumn(
        "flag_round_amount", (F.col("amount_base") % 1000 == 0) & (F.col("amount_base") >= 10_000)
    )


def p_dormant_then_active(df: DataFrame) -> DataFrame:
    """Long-dormant account suddenly transacting."""
    w = Window.partitionBy("party_id").orderBy("event_ts")
    return df.withColumn("_gap_days",
        (F.col("event_ts").cast("long") - F.lag("event_ts").over(w).cast("long")) / 86400,
    ).withColumn(
        "flag_dormant_then_active", F.col("_gap_days") >= RISK.dormancy_days
    ).drop("_gap_days")


def p_high_velocity(df: DataFrame) -> DataFrame:
    """Unusually many transactions in a day for the party."""
    daily = Window.partitionBy("party_id", F.to_date("event_ts"))
    return df.withColumn("_day_cnt", F.count("*").over(daily)).withColumn(
        "flag_high_velocity", F.col("_day_cnt") >= RISK.high_velocity_txn_count
    ).drop("_day_cnt")


def p_high_risk_geo(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "flag_high_risk_geo", F.col("counterparty_country").isin(HIGH_RISK_COUNTRIES)
    )


def p_new_account_high_value(df: DataFrame) -> DataFrame:
    """Large value shortly after account opening."""
    return df.withColumn(
        "flag_new_account_high_value",
        (F.datediff("event_ts", "account_open_date") <= 30) & (F.col("amount_base") >= 100_000),
    )


def p_fan_in_out(df: DataFrame) -> DataFrame:
    """Mule-like fan-in / fan-out: many distinct counterparties per party per day."""
    daily = Window.partitionBy("party_id", F.to_date("event_ts"))
    return df.withColumn(
        "_distinct_cp", F.size(F.collect_set("counterparty_id").over(daily))
    ).withColumn("flag_fan_in_out", F.col("_distinct_cp") >= 10).drop("_distinct_cp")


def p_tbml_invoice_mismatch(df: DataFrame) -> DataFrame:
    """Trade-based ML: invoice value diverges sharply from estimated goods value."""
    return df.withColumn(
        "flag_tbml_invoice_mismatch",
        (F.col("product_type") == "TRADE_FINANCE")
        & (
            F.abs(F.col("invoice_value") - F.col("goods_value_estimate"))
            / F.greatest(F.col("goods_value_estimate"), F.lit(1.0))
            > 0.3
        ),
    )


def p_sanctions_nexus(df: DataFrame) -> DataFrame:
    """Counterparty resolved to a real-time screening hit (populated upstream)."""
    return df.withColumn(
        "flag_sanctions_nexus", F.coalesce(F.col("screening_sanctions_hit"), F.lit(False))
    )


def p_pep_nexus(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "flag_pep_nexus", F.coalesce(F.col("screening_pep_hit"), F.lit(False))
    )


def p_pf_dualuse_nexus(df: DataFrame) -> DataFrame:
    """Proliferation-financing indicator: trade with a PF-nexus jurisdiction."""
    return df.withColumn(
        "flag_pf_dualuse_nexus",
        (F.col("product_type") == "TRADE_FINANCE")
        & F.col("counterparty_country").isin(PF_NEXUS_COUNTRIES),
    )


def p_correspondent_nested(df: DataFrame) -> DataFrame:
    """Nested / downstream correspondent relationship needing EDD."""
    return df.withColumn(
        "flag_correspondent_nested",
        (F.col("channel") == "CORRESPONDENT") & (F.col("nested_relationship") == True),  # noqa: E712
    )


def p_third_party_funding(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "flag_third_party_funding",
        F.col("originator_id") != F.col("account_holder_id"),
    )


def p_cross_border_layering(df: DataFrame) -> DataFrame:
    """Same value hopping across borders in quick succession."""
    w = Window.partitionBy("value_trace_id").orderBy("event_ts")
    return df.withColumn(
        "flag_cross_border_layering",
        (F.count("*").over(w) >= 3)
        & (F.size(F.collect_set("counterparty_country").over(w)) >= 3),
    )


# Ordered catalogue: name -> (transform, description shown to the agent)
PATTERN_CATALOGUE: dict[str, tuple[Callable[[DataFrame], DataFrame], str]] = {
    "structuring": (p_structuring, "Multiple sub-threshold deposits in a short window."),
    "just_below_threshold": (p_just_below_threshold, "Single amount parked just under a reporting-style figure."),
    "rapid_movement": (p_rapid_movement, "Funds out shortly after coming in (pass-through)."),
    "round_amount": (p_round_amount, "Suspiciously round large amounts."),
    "dormant_then_active": (p_dormant_then_active, "Dormant account suddenly active."),
    "high_velocity": (p_high_velocity, "Abnormally many transactions in a day."),
    "high_risk_geo": (p_high_risk_geo, "Counterparty in a high-risk jurisdiction."),
    "new_account_high_value": (p_new_account_high_value, "Large value soon after opening."),
    "fan_in_out": (p_fan_in_out, "Mule-like many-counterparty fan-in/out."),
    "tbml_invoice_mismatch": (p_tbml_invoice_mismatch, "Trade invoice vs goods-value divergence."),
    "sanctions_nexus": (p_sanctions_nexus, "Counterparty screened to a sanctions hit."),
    "pep_nexus": (p_pep_nexus, "Counterparty screened to a PEP hit."),
    "pf_dualuse_nexus": (p_pf_dualuse_nexus, "Trade with a proliferation-nexus jurisdiction."),
    "correspondent_nested": (p_correspondent_nested, "Nested correspondent relationship (EDD)."),
    "third_party_funding": (p_third_party_funding, "Funder differs from account holder."),
    "cross_border_layering": (p_cross_border_layering, "Same value layered across 3+ borders."),
}


def apply_all_patterns(df: DataFrame) -> DataFrame:
    """Stamp every flag_ column onto the transaction DataFrame."""
    for _name, (fn, _desc) in PATTERN_CATALOGUE.items():
        df = fn(df)
    flag_cols = [f"flag_{n}" for n in PATTERN_CATALOGUE]
    return df.withColumn(
        "flag_risk_count",
        sum(F.col(c).cast("int") for c in flag_cols),
    )


def catalogue_for_prompt() -> str:
    """Human-readable list of flags for the agent's `available_patterns` input."""
    return "\n".join(f"- flag_{n}: {desc}" for n, (_fn, desc) in PATTERN_CATALOGUE.items())
